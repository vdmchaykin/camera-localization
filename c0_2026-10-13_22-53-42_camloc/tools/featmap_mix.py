#!/usr/bin/env python3
"""
Method 3a: 3D feature map from the training run + camera relocalization (v7 pipeline at inference).

Build the map (offline; GT poses of the selected laps are allowed here) -- unchanged:
    python featmap.py build --data data/ --laps 1 2 3 5 6 --cams fl fr r --out maps/v1
    python colmap_map.py build ...                   # COLMAP / hloc map, same format
    python dino_desc.py --data data/ --map maps/v1    # DINOv2 descriptors of the map keyframes

Validate on held-out laps (causal: GT only for the first pose of each lap and for metrics):
    python featmap.py validate --data data/ --map maps/colmap_7cam_kf2 --laps 3 6 --out val.csv \
        --state data/state.parquet --dino
    python featmap.py validate ... --prior gt        # DEBUG ONLY: tests matching in isolation

Inference = the v7 localizer of the guided-keyframe-ekf branch (v7 -> v6 -> v5 -> v3 -> v2), on this map:
  * filter: TrackEKF [x, y, yaw, speed scale k, yaw-rate bias bw], predicted with every car-state message
    through the calibrated motion model (motion_calib.json: vy_eff = a*vy + b*wz + c, wz_eff = s*wz + d,
    vx_eff = k*vx); measurement noise scaled by sqrt(100 / inliers), chi2 gate, recovery when two rejected
    fixes ask for the same correction;
  * a correction on every frame (in corners and on straights);
  * map keyframes: DINOv2 near the prior (as before, --dino), else the nearest ones, one per pass;
  * guided matching from the filter prior: map points projected with the prior, SuperPoint keypoints matched
    only inside a window around the projection (mutual best descriptor), then a dense search in SuperPoint's
    descriptor map for the map points left unmatched (sub-pixel), no LightGlue, no RANSAC;
  * pose: robust Gauss-Newton over all cameras starting from the prior (two inlier thresholds);
  * SuperPoint: one FP16 batch for all cameras, 1024 keypoints per camera;
  * daylight defences (v6): exposure normalisation to the map's median grey, saturated pixels masked,
    a relaxed retry when the guided fix fails, fixes off the drivable area rejected (track_bounds);
  * v7: map points on bright blobs (5x5 peak of the keyframe grey >= 200) are dropped;
  * global fallback (filter lost: sigma > 1.5 m or 10 failures in a row, at most once per second): the
    previous path, DINOv2 candidates + LightGlue + PnP-RANSAC + joint refinement, near the prior.
v7 pixel constants are for 753 px wide images; here they are scaled to the full-resolution camera images.
"""
import argparse, copy, json, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from lightglue import SuperPoint, LightGlue
from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints

from lap_index import load_frames

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- geometry
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def T_from(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def T_base_cam(calib, cam):
    """Compose /tf_static edges from the camera frame up to base_link: p_base = T @ p_cam."""
    start = f = calib["cameras"][cam]["frame_id"]
    T, seen = np.eye(4), set()
    while not f.endswith("base_link"):
        if f not in calib["tf_static"] or f in seen:
            raise KeyError(f"no /tf_static chain from {start} to base_link (stuck at {f})")
        seen.add(f)
        e = calib["tf_static"][f]
        T = T_from(e["t"], e["q_xyzw"]) @ T
        f = e["parent"]
    return T


def T_map_base_row(r):
    return T_from([r.x, r.y, r.z], [r.qx, r.qy, r.qz, r.qw])


def T_from_xyyaw(x, y, yaw, z=0.0):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def xyyaw(T):
    return float(T[0, 3]), float(T[1, 3]), float(np.arctan2(T[1, 0], T[0, 0]))


def optical_heading(T_map_cam):
    z = T_map_cam[:3, 2]                      # camera z axis = viewing direction
    return float(np.arctan2(z[1], z[0]))


class Cam:
    def __init__(self, calib, name):
        c = calib["cameras"][name]
        self.name = name
        self.K = np.array(c["K"], dtype=np.float64).reshape(3, 3)
        self.D = np.array(c["D"], dtype=np.float64)
        self.f = float(self.K[0, 0])
        self.width, self.height = int(c["width"]), int(c["height"])
        self.T_bc = T_base_cam(calib, name)       # cam -> base_link
        self.T_cb = np.linalg.inv(self.T_bc)

    def norm(self, uv):
        """pixels -> undistorted normalized image coordinates"""
        if len(uv) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64),
                                   self.K, self.D).reshape(-1, 2)

    def project(self, pc):
        """points in the camera frame (N,3) -> pixels (N,2), nan behind / far outside the image"""
        uv = np.full((len(pc), 2), np.nan)
        z = pc[:, 2]
        ok = z > 0.5
        zz = np.where(ok, z, 1.0)
        x, y = pc[:, 0] / zz, pc[:, 1] / zz
        ok &= ((np.abs(x) < 1.15 * self.width / 2 / self.K[0, 0])
               & (np.abs(y) < 1.15 * self.height / 2 / self.K[1, 1]))
        if ok.any():
            uv[ok] = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                       self.K, self.D)[0].reshape(-1, 2)
        return uv


def load_mask(data, cam):
    p = Path(data) / "masks" / f"{cam}.png"    # optional: white = usable pixels
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None


# ----------------------------------------------------------------------------- features
FP16 = False                    # set by --fast-gpu: SuperPoint (LightGlue path) / DINOv2 under fp16


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=FP16 and DEV == "cuda")


class Features:
    """SuperPoint + LightGlue: map building and the global fallback at inference."""

    def __init__(self, max_kp=1024, mp=False):
        self.sp = SuperPoint(max_num_keypoints=max_kp).eval().to(DEV)
        try:                           # mp = mixed precision (fp16) inside LightGlue
            self.lg = LightGlue(features="superpoint", mp=mp).eval().to(DEV)
        except TypeError:
            self.lg = LightGlue(features="superpoint").eval().to(DEV)
        self.resize = 1024             # SuperPoint input: long image side in px

    @torch.no_grad()
    def extract(self, path, mask=None):
        img = path if isinstance(path, np.ndarray) else cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        t = torch.from_numpy(img[:, :, ::-1].copy()).to(DEV).permute(2, 0, 1).float().div(255)
        with autocast():
            f = self.sp.extract(t, resize=self.resize)  # keypoints in original pixels
        f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
        kp = f["keypoints"][0].cpu().numpy()
        de = f["descriptors"][0].cpu().numpy()
        sc = f["keypoint_scores"][0].cpu().numpy()
        if mask is not None:
            u = kp[:, 0].astype(int).clip(0, mask.shape[1] - 1)
            v = kp[:, 1].astype(int).clip(0, mask.shape[0] - 1)
            keep = mask[v, u] > 0
            kp, de, sc = kp[keep], de[keep], sc[keep]
        return {"kp": kp.astype(np.float32), "desc": de.astype(np.float16),
                "score": sc.astype(np.float32), "size": np.array([img.shape[1], img.shape[0]])}

    @torch.no_grad()
    def match(self, a, b):
        if len(a["kp"]) < 8 or len(b["kp"]) < 8:
            return np.zeros((0, 2), dtype=int)

        def pack(x):
            return {"keypoints": torch.from_numpy(x["kp"])[None].to(DEV),
                    "descriptors": torch.from_numpy(x["desc"].astype(np.float32))[None].to(DEV),
                    "image_size": torch.from_numpy(x["size"]).float()[None].to(DEV)}

        out = self.lg({"image0": pack(a), "image1": pack(b)})
        return out["matches"][0].cpu().numpy()


# ----------------------------------------------------------------------------- build
def select_keyframes(f, dist_m, yaw_deg):
    keep, last = [], None
    for i, r in enumerate(f.itertuples()):
        if (last is None or np.hypot(r.x - last.x, r.y - last.y) >= dist_m
                or abs(wrap(r.yaw - last.yaw)) >= np.deg2rad(yaw_deg)):
            keep.append(i)
            last = r
    return f.iloc[keep].reset_index(drop=True)


def triangulate_pair(cam, fa, fb, Tma, Tmb, m, o):
    """Two views of one camera with known map poses -> 3D points + validity mask."""
    ua, ub = fa["kp"][m[:, 0]], fb["kp"][m[:, 1]]
    moving = np.linalg.norm(ua - ub, axis=1) >= o.min_flow_px   # drops car body / glare
    na, nb = cam.norm(ua), cam.norm(ub)
    Pa, Pb = np.linalg.inv(Tma)[:3], np.linalg.inv(Tmb)[:3]      # map -> cam, 3x4
    Xh = cv2.triangulatePoints(Pa, Pb, na.T, nb.T)
    X = (Xh[:3] / Xh[3]).T

    def proj(P):
        Xc = X @ P[:, :3].T + P[:, 3]
        z = Xc[:, 2]
        return z, Xc[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)[:, None]

    za, pa = proj(Pa)
    zb, pb = proj(Pb)
    err_px = np.maximum(np.linalg.norm(pa - na, axis=1), np.linalg.norm(pb - nb, axis=1)) * cam.f
    ra, rb = X - Tma[:3, 3], X - Tmb[:3, 3]
    cos = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-9)
    par = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ok = (moving & (za > o.min_depth) & (zb > o.min_depth) & (za < o.max_depth)
          & (zb < o.max_depth) & (err_px < o.max_reproj_px) & (par > o.min_parallax_deg))
    return X, ok, par


def consolidate(obs, kp_cam_center, o):
    """obs: {kp_idx: [(X, parallax), ...]} -> {kp_idx: X} with a consistency check."""
    out = {}
    for k, lst in obs.items():
        Xs = np.array([x for x, _ in lst])
        if len(lst) == 1:
            if lst[0][1] >= 2 * o.min_parallax_deg:
                out[k] = Xs[0]
            continue
        Xm = np.median(Xs, axis=0)
        depth = np.linalg.norm(Xm - kp_cam_center)
        if np.max(np.linalg.norm(Xs - Xm, axis=1)) < max(0.3, 0.03 * depth):
            out[k] = Xm
    return out


def build(a):
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = json.loads((data / "calib.json").read_text())
    feats = Features(a.max_kp)

    for cam_name in a.cams:
        cam = Cam(calib, cam_name)
        mask = load_mask(data, cam_name)
        f = load_frames(data, laps=a.laps, cams=[cam_name])
        kfs = select_keyframes(f, a.kf_dist, a.kf_yaw)
        if a.limit:
            kfs = kfs.iloc[:a.limit]
        print(f"[{cam_name}] {len(f)} frames -> {len(kfs)} keyframes")

        Tmc = [T_map_base_row(r) @ cam.T_bc for r in kfs.itertuples()]
        stamps = kfs.stamp_ns.values
        window = deque()                       # (kf_idx, feats, obs)
        store = {"kp": [], "desc": [], "xyz": [], "score": [], "kf": []}
        n_pts, t0 = 0, time.time()

        def finalize(item):
            nonlocal n_pts
            i, fi, obs = item
            pts = consolidate(obs, Tmc[i][:3, 3], a)
            if not pts:
                return
            idx = np.array(sorted(pts, key=lambda k: -fi["score"][k]))[:a.max_pts_per_kf]
            store["kp"].append(fi["kp"][idx]); store["desc"].append(fi["desc"][idx])
            store["score"].append(fi["score"][idx])
            store["xyz"].append(np.array([pts[k] for k in idx], dtype=np.float32))
            store["kf"].append(np.full(len(idx), i))
            n_pts += len(idx)

        for i in range(len(kfs)):
            fi = feats.extract(data / kfs.path[i], mask)
            cur = (i, fi, {})
            for j, fj, obs_j in window:
                if (stamps[i] - stamps[j]) * 1e-9 > a.max_gap_s:
                    continue                    # e.g. jump from lap 3 to lap 5
                m = feats.match(fj, fi)
                if len(m) < 8:
                    continue
                X, ok, par = triangulate_pair(cam, fj, fi, Tmc[j], Tmc[i], m, a)
                for (kj, ki), x, p in zip(m[ok], X[ok], par[ok]):
                    obs_j.setdefault(kj, []).append((x, p))
                    cur[2].setdefault(ki, []).append((x, p))
            window.append(cur)
            if len(window) > a.pair_span:
                finalize(window.popleft())
            if i % 200 == 0:
                print(f"  kf {i}/{len(kfs)}  points {n_pts}  {time.time() - t0:.0f}s")
        while window:
            finalize(window.popleft())

        kf_ids = np.concatenate(store["kf"])
        np.savez(out / f"{cam_name}.npz",
                 kp=np.concatenate(store["kp"]), desc=np.concatenate(store["desc"]),
                 xyz=np.concatenate(store["xyz"]), score=np.concatenate(store["score"]),
                 kf=kf_ids)
        kt = kfs[["stamp_ns", "lap", "x", "y", "z", "yaw"]].copy()
        kt["cx"] = [T[0, 3] for T in Tmc]
        kt["cy"] = [T[1, 3] for T in Tmc]
        kt["cam_heading"] = [optical_heading(T) for T in Tmc]
        kt["n_pts"] = np.bincount(kf_ids, minlength=len(kfs))
        kt.to_parquet(out / f"{cam_name}_kf.parquet")
        print(f"[{cam_name}] saved {n_pts} points, "
              f"{(kt.n_pts > 0).mean():.0%} keyframes have points")

    (out / "meta.json").write_text(json.dumps(
        {"laps": a.laps, "cams": a.cams, "args": {k: v for k, v in vars(a).items()
                                                  if isinstance(v, (int, float, str, list))}},
        indent=2))


# ----------------------------------------------------------------------------- map
class CamMap:
    def __init__(self, map_dir, cam):
        z = np.load(Path(map_dir) / f"{cam}.npz")
        self.kt = pd.read_parquet(Path(map_dir) / f"{cam}_kf.parquet")
        order = np.argsort(z["kf"], kind="stable")
        self.kp, self.desc, self.xyz = z["kp"][order], z["desc"][order], z["xyz"][order]
        self.kf = z["kf"][order]
        self._index()
        self.tree = cKDTree(self.kt[["cx", "cy"]].values)
        self.base_xy = self.kt[["x", "y"]].values
        self.base_yaw = self.kt.yaw.values
        self.stamp = self.kt.stamp_ns.values.astype(np.int64)
        self.size = None
        self.dino = None

    def _index(self):
        self.start = np.searchsorted(self.kf, np.arange(len(self.kt)))
        self.end = np.searchsorted(self.kf, np.arange(len(self.kt)), side="right")

    def prune(self, keep):
        """drop map points (bool mask over the points, in this object's order)"""
        self.kp, self.desc, self.xyz, self.kf = self.kp[keep], self.desc[keep], self.xyz[keep], self.kf[keep]
        self._index()

    def keyframe(self, i):
        s, e = self.start[i], self.end[i]
        return {"kp": self.kp[s:e], "desc": self.desc[s:e], "size": self.size}, self.xyz[s:e]


def keyframe_stats(data, map_dir, cam, cm, mask):
    """v6/v7 training-time artefacts from the map keyframe images (cached in the map folder):
    med  = per keyframe, median raw grey inside the body mask (target of the exposure normalisation),
    peak = per map point, max raw grey in a 5x5 patch at the keypoint (half resolution, as v7)."""
    p = Path(map_dir) / f"light_{cam}.npz"
    if p.exists():
        z = np.load(p)
        if len(z["peak"]) == len(cm.kp) and len(z["med"]) == len(cm.kt):
            return z["med"], z["peak"]
    t0 = time.time()
    fr = pd.read_parquet(Path(data) / "frames.parquet", columns=["stamp_ns", "cam", "path"])
    path = dict(zip(fr[fr.cam == cam].stamp_ns.values, fr[fr.cam == cam].path.values))
    med = np.full(len(cm.kt), np.nan, np.float32)
    peak = np.zeros(len(cm.kp), np.uint8)
    mh = None if mask is None else cv2.resize(mask, (cm.size[0] // 2, cm.size[1] // 2),
                                              interpolation=cv2.INTER_NEAREST) > 0

    def work(i):
        q = path.get(int(cm.stamp[i]))
        g = None if q is None else cv2.imread(str(Path(data) / q), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if g is None:
            return
        med[i] = float(np.median(g[mh] if mh is not None and mh.shape == g.shape else g))
        s, e = cm.start[i], cm.end[i]
        if e > s:
            mx = cv2.dilate(g, np.ones((5, 5), np.uint8))
            k = np.round((cm.kp[s:e] + 0.5) * 0.5 - 0.5).astype(int)
            peak[s:e] = mx[k[:, 1].clip(0, g.shape[0] - 1), k[:, 0].clip(0, g.shape[1] - 1)]

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(work, range(len(cm.kt))))
    np.savez(p, med=med, peak=peak)
    print(f"[{cam}] keyframe stats from {np.isfinite(med).sum()} images in {time.time() - t0:.0f}s -> {p}")
    return med, peak


class Drivable:
    """v6: raster of the drivable area, between track_left / track_right (closed loops) + pit lane."""

    def __init__(self, bounds, res=0.25, margin=1.5):
        pts = np.concatenate([b[:, :2] for b in bounds])
        self.lo, hi, self.res = pts.min(0) - 20, pts.max(0) + 20, res
        self.shape = tuple(np.ceil((hi - self.lo) / res).astype(int)[::-1])        # rows (y), cols (x)
        px = lambda v: np.round((v[:, :2] - self.lo) / res).astype(np.int32)
        a = np.zeros(self.shape, np.uint8); b = np.zeros(self.shape, np.uint8)
        cv2.fillPoly(a, [px(bounds[0])], 1)
        cv2.fillPoly(b, [px(bounds[1])], 1)
        m = a ^ b
        if len(bounds) >= 4:
            cv2.fillPoly(m, [np.concatenate([px(bounds[2]), px(bounds[3])[::-1]])], 1)
        k = int(np.ceil(margin / res)) * 2 + 1
        self.m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    def inside(self, x, y):
        c, r = np.round((np.array([x, y]) - self.lo) / self.res).astype(int)
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1] and bool(self.m[r, c])


def load_bounds(data):
    """track_bounds.json: [track_left, track_right, pitlane_left, pitlane_right] line strips"""
    p = Path(data) / "track_bounds.json"
    return [np.asarray(q, dtype=np.float64) for _, q in json.loads(p.read_text())] if p.exists() else None


# ----------------------------------------------------------------------------- filter
class TrackEKF:
    """v2 CalibEKF: X = [x, y, yaw, k, bw]; speed scale k (v_true = k * v) and yaw-rate bias bw."""
    Q_POS_T = 0.10        # m / sqrt(s)
    Q_POS_D = 0.010       # fraction of distance travelled (1 sigma)
    Q_YAW_T = 0.004       # rad / sqrt(s)
    Q_K = 2e-4            # 1 / sqrt(s)
    Q_BW = 2e-5           # rad/s / sqrt(s)
    R_LAT, R_LON, R_YAW = 0.12, 0.30, np.radians(0.30)   # camera pose with 100 inliers (body frame)
    MIN_SCALE = 0.7       # sqrt(100 / inliers) floored: more than ~200 inliers are not more certain
    GATE_CHI2 = 16.0      # 3 dof, ~99.9 %
    MIN_INLIERS = 30
    RECOVER_POS, RECOVER_YAW = 0.3, np.radians(0.3)

    def __init__(self, x, y, yaw, calib=None):
        self.cal = calib
        self.X = np.array([x, y, yaw, 1.0, 0.0])
        self.P = np.diag([0.1 ** 2, 0.1 ** 2, np.radians(0.3) ** 2, 0.01 ** 2, 0.002 ** 2])
        self.n_upd = self.n_rej = self.n_recover = 0
        self.since_upd = 0.0
        self.last_rej = None      # innovation of the previous measurement if it was rejected

    def predict(self, vx, vy, wz, dt):
        """vx, vy: body-frame velocity, wz: yaw rate (car state); calibrated to base_link first."""
        if dt <= 0:
            return
        c = self.cal
        if c:
            vx, vy, wz = c["vx_k"] * vx, c["vy_a"] * vy + c["vy_b"] * wz + c["vy_c"], c["wz_s"] * wz + c["wz_d"]
        x, y, yaw, k, bw = self.X
        w = wz - bw
        ym = yaw + 0.5 * w * dt
        cs, sn = np.cos(ym), np.sin(ym)
        ux, uy = (vx * cs - vy * sn) * dt, (vx * sn + vy * cs) * dt
        dx, dy = k * ux, k * uy
        d = np.hypot(dx, dy)
        self.X = np.array([x + dx, y + dy, wrap(yaw + w * dt), k, bw])
        Fm = np.eye(5)
        Fm[0, 2] = -dy; Fm[0, 3] = ux; Fm[0, 4] = 0.5 * dt * dy
        Fm[1, 2] = dx;  Fm[1, 3] = uy; Fm[1, 4] = -0.5 * dt * dx
        Fm[2, 4] = -dt
        s2 = self.Q_POS_T ** 2 * dt + (self.Q_POS_D * abs(d)) ** 2
        Q = np.diag([s2, s2, self.Q_YAW_T ** 2 * dt, self.Q_K ** 2 * dt, self.Q_BW ** 2 * dt])
        self.P = Fm @ self.P @ Fm.T + Q
        self.since_upd += dt

    def update(self, zx, zy, zyaw, inliers):
        """-> (accepted, Mahalanobis d2)"""
        if inliers < self.MIN_INLIERS:
            self.n_rej += 1
            return False, np.nan
        c, s = np.cos(zyaw), np.sin(zyaw)
        sc = max(self.MIN_SCALE, np.sqrt(100.0 / inliers))
        Rot = np.array([[c, -s], [s, c]])
        Rm = np.zeros((3, 3))
        Rm[:2, :2] = Rot @ np.diag([(self.R_LON * sc) ** 2, (self.R_LAT * sc) ** 2]) @ Rot.T
        Rm[2, 2] = (self.R_YAW * sc) ** 2
        H = np.zeros((3, 5)); H[0, 0] = H[1, 1] = H[2, 2] = 1
        r = np.array([zx - self.X[0], zy - self.X[1], wrap(zyaw - self.X[2])])
        S = H @ self.P @ H.T + Rm
        d2 = float(r @ np.linalg.solve(S, r))
        if d2 > self.GATE_CHI2:
            prev, self.last_rej = self.last_rej, r
            if prev is not None and np.hypot(*(r[:2] - prev[:2])) < self.RECOVER_POS \
                    and abs(wrap(r[2] - prev[2])) < self.RECOVER_YAW:
                # the filter, not the measurements, is off: widen the pose covariance to cover the offset
                self.P[:3, :3] += np.diag(r ** 2) * 4
                S = H @ self.P @ H.T + Rm
                self.n_recover += 1
            else:
                self.n_rej += 1
                if self.since_upd > 5.0:          # long without corrections: let the covariance grow
                    self.P[:3, :3] *= 2.0
                return False, d2
        self.last_rej = None
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ r
        self.X[2] = wrap(self.X[2])
        self.P = (np.eye(5) - K @ H) @ self.P
        self.n_upd += 1
        self.since_upd = 0.0
        return True, d2

    @property
    def pose(self):
        return float(self.X[0]), float(self.X[1]), float(wrap(self.X[2]))

    def prior(self):
        return dict(x=float(self.X[0]), y=float(self.X[1]), yaw=float(self.X[2]),
                    sig_xy=float(np.sqrt(max(self.P[0, 0], self.P[1, 1]))), sig_yaw=float(np.sqrt(self.P[2, 2])))


class CarState:
    """car state (allowed input): each message holds until the next one (strictly causal)."""

    def __init__(self, path, vcol, vycol, rcol, rscale=1.0):
        st = pd.read_parquet(path).sort_values("stamp_ns")
        self.ts = st.stamp_ns.values.astype(np.int64)
        self.v = np.c_[st[vcol].values, st[vycol].values if vycol else np.zeros(len(st)),
                       st[rcol].values * rscale].astype(float)

    def propagate(self, f, t0, t1):
        if t1 <= t0:
            return
        ts = self.ts
        i = int(np.searchsorted(ts, t0, side="right")) - 1
        hold = self.v[max(i, 0)]                   # before the first message: hold the first one
        tc, j = t0, i + 1
        while j < len(ts) and ts[j] <= t1:
            f.predict(*hold, (ts[j] - tc) * 1e-9)
            tc, hold = ts[j], self.v[j]
            j += 1
        f.predict(*hold, (t1 - tc) * 1e-9)

    def yaw_rate(self, t):
        i = int(np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.ts) - 1))
        return float(self.v[i, 2])


# ----------------------------------------------------------------------------- localize
REF_W = 753.0               # v7 pixel constants are for 753 px wide images
N_KF_DIST, KF_MAX_DIST, KF_MAX_DYAW = 3, 8.0, np.radians(45)   # nearest keyframes (no DINO), one per pass
PASS_GAP_S = 20.0
SIM_MIN, RELAX_SIM_MIN = 0.70, 0.60          # SuperPoint descriptor cosine similarity
WIN_MIN, WIN_MAX, WIN_ADD = 5.0, 40.0, 3.0   # px @ REF_W, search window around a projected map point
THR_PX = 3.0                                 # px @ REF_W, inlier threshold of the pose
MIN_INLIERS_GUIDED = 40
DENSE_R = 8                                  # px @ REF_W, max radius of the dense search
DENSE_SIM, RELAX_DENSE_SIM, DENSE_MARGIN, DENSE_FAR = 0.75, 0.65, 0.03, 2.5
SAT_GREY, SAT_DILATE, SAT_CAM_FRAC = 250, 11, 0.60   # dilation px @ REF_W
GAMMA_MIN, GAMMA_MAX = 0.35, 2.5
LOST_SIGMA, LOST_FAILS, GLOBAL_MIN_S, GLOBAL_RADIUS_MIN = 1.5, 10, 1.0, 30.0


def _backbone(s, x):
    """SuperPoint encoder + heads -> (score logits (b, 65, h/8, w/8), dense descriptors (b, 256, h/8, w/8))"""
    y = s.relu(s.conv1a(x)); y = s.relu(s.conv1b(y)); y = s.pool(y)
    y = s.relu(s.conv2a(y)); y = s.relu(s.conv2b(y)); y = s.pool(y)
    y = s.relu(s.conv3a(y)); y = s.relu(s.conv3b(y)); y = s.pool(y)
    y = s.relu(s.conv4a(y)); y = s.relu(s.conv4b(y))
    return s.convPb(s.relu(s.convPa(y))), s.convDb(s.relu(s.convDa(y)))


class Localizer:
    def __init__(self, data, map_dir, cams=None, o=None):
        meta = json.loads((Path(map_dir) / "meta.json").read_text())
        calib = json.loads((Path(data) / "calib.json").read_text())
        self.o = o
        self.names = cams or meta["cams"]
        self.cams = {c: Cam(calib, c) for c in self.names}
        self.maps = {c: CamMap(map_dir, c) for c in self.names}
        for c in self.names:
            self.maps[c].size = np.array([self.cams[c].width, self.cams[c].height])
        self.masks = {c: load_mask(data, c) for c in self.names}
        if o.no_query_masks:
            self.masks = {c: None for c in self.names}
        offs = o.cam_yaw_offsets
        if offs:
            offs = json.loads(Path(offs).read_text())
            for c, deg in offs.items():
                if c in self.cams:      # camera c reports base yaw too large by deg -> rotate it
                    Rz = T_from_xyyaw(0, 0, np.radians(deg))
                    self.cams[c].T_bc = Rz @ self.cams[c].T_bc
                    self.cams[c].T_cb = np.linalg.inv(self.cams[c].T_bc)
            print("camera yaw offsets applied:", {c: round(v, 4) for c, v in offs.items()})
        sizes = {(c.width, c.height) for c in self.cams.values()}
        assert len(sizes) == 1, f"one SuperPoint batch needs equal image sizes, got {sizes}"
        self.W, self.H = sizes.pop()
        self.px = self.W / REF_W                               # v7 px -> full-resolution px
        s = o.sp_resize / max(self.W, self.H)
        self.s_in = min(s, 1.0)                                # full-res px -> SuperPoint input px
        self.W_in, self.H_in = int(round(self.W * self.s_in)), int(round(self.H * self.s_in))
        self.dense_r = int(round(DENSE_R * self.W_in / REF_W))
        g = np.arange(-self.dense_r, self.dense_r + 1)
        self.grid = torch.from_numpy(np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)).to(DEV)
        self.clahe = cv2.createCLAHE(2.0, (8, 8)) if o.clahe else None

        # map keyframe base poses (z / roll / pitch for the prior) from the map laps' GT
        fr = pd.read_parquet(Path(data) / "frames.parquet")
        ref = self.names[0]
        cm = self.maps[ref]
        q = cm.kt[["stamp_ns"]].merge(fr[fr.cam == ref][["stamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"]],
                                      on="stamp_ns", how="left")
        self.kf_T = np.stack([T_map_base_row(r) if np.isfinite(r.qw) else T_from_xyyaw(r.x, r.y, 0)
                              for r in q.itertuples()])
        self.kf_ref = ref
        self.kf_tree = cKDTree(cm.base_xy)

        # v6 exposure target, v7 bright map points
        self.target, self.pruned = {}, {}
        if o.expo or o.prune_peak > 0:
            for c in self.names:
                med, peak = keyframe_stats(data, map_dir, c, self.maps[c], self.masks[c])
                self.target[c] = float(np.nanmedian(med))
                if o.prune_peak > 0:
                    keep = peak < o.prune_peak
                    self.maps[c].prune(keep)
                    self.pruned[c] = round(float(1 - keep.mean()) * 100, 1)
            if self.pruned:
                print(f"v7: map points with 5x5 peak >= {o.prune_peak} dropped, % per camera: {self.pruned}")
            if o.expo:
                print("v6 exposure target (median grey of the map keyframes):",
                      {c: round(v, 1) for c, v in self.target.items()})
        bounds = load_bounds(data) if o.bounds else None
        self.drivable = Drivable(bounds) if bounds else None
        if o.bounds and bounds is None:
            print("WARNING: data/track_bounds.json missing: drivable-area check off")

        # SuperPoint for the guided path: FP16, channels_last, one batch for all cameras
        sp = SuperPoint(max_num_keypoints=None, nms_radius=o.nms_radius).eval().to(DEV)
        self.sp_conf = sp.conf
        self.sp16 = sp.half().to(memory_format=torch.channels_last) if DEV == "cuda" else sp
        fn = lambda x: _backbone(self.sp16, x)
        self.bb = torch.compile(fn, dynamic=False) if o.sp_compile else fn
        # global fallback: SuperPoint + LightGlue + PnP-RANSAC
        self.feats = Features(o.max_kp, mp=o.lg_mp)
        self.feats.resize = o.sp_resize
        self.pool = ThreadPoolExecutor(8)       # JPEG decoding of all cameras in parallel
        self.prof = {} if o.profile else None
        self.dino = None
        if o.dino:
            try:                                  # dino_desc.py with DINOv3 support
                from dino_desc import Dino, TAG
                self.dino = Dino(o.dino_model)
            except ImportError:                   # older dino_desc.py: DINOv2 only
                from dino_desc import Dino
                if o.dino_model != "dinov2":
                    raise SystemExit("this dino_desc.py supports only DINOv2")
                TAG = {"dinov2": "dino"}
                self.dino = Dino()
            for c in self.names:
                p = Path(map_dir) / f"{c}_{TAG[o.dino_model]}.npy"
                if not p.exists():
                    raise SystemExit(f"{p} missing: run dino_desc.py --map {map_dir} "
                                     f"--model {o.dino_model}")
                self.maps[c].dino = np.load(p).astype(np.float32)
            print(f"{o.dino_model} keyframe selection: radius {o.dino_radius} m, top {o.k_cand} per camera")
        self.transform = None                     # optional image degradation (stress tests)
        self.reset()
        self.st = dict(guided=0, guided_fail=0, relax=0, relax_ok=0, global_=0, global_ok=0, global_skip=0,
                       bounds_rej=0, cam_skip=0, n_fix=0)
        self.gammas = []
        self._warmup()

    def reset(self):
        """per run / lap: failure counter of the guided path, last global attempt"""
        self.fails, self.last_global = 0, None

    def _warmup(self):
        z = {c: np.zeros((self.H, self.W, 3), np.uint8) for c in self.names}
        self._features(z)
        if DEV == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------ profiling
    def _tick(self, name, t0):
        if self.prof is None:
            return time.perf_counter()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.prof[name] = self.prof.get(name, 0.0) + (t1 - t0) * 1000
        return t1

    def profile_report(self):
        n = self.st["n_fix"]
        if not self.prof or not n:
            return
        tot = sum(self.prof.values())
        print(f"time per camera fix ({n} fixes), ms:")
        for k, v in sorted(self.prof.items(), key=lambda x: -x[1]):
            print(f"  {k:12s} {v / n:7.1f}   ({v / tot:.0%})")
        print(f"  {'total':12s} {tot / n:7.1f}")

    # ------------------------------------------------------------ features (v6: exposure + saturation, v5: dense maps)
    @torch.no_grad()
    def _features(self, imgs):
        o = self.o
        names, grays = [], []
        self._ok = {}
        k_sat = max(3, int(round(SAT_DILATE * self.px)) | 1)
        for c in self.names:
            if c not in imgs:
                continue
            raw = cv2.cvtColor(imgs[c], cv2.COLOR_BGR2GRAY)
            mask = self.masks[c]
            ok = mask > 0 if mask is not None else np.ones(raw.shape, bool)
            if o.sat:
                sat = cv2.dilate((raw >= SAT_GREY).astype(np.uint8), np.ones((k_sat, k_sat), np.uint8)) > 0
                if (sat & ok).sum() > SAT_CAM_FRAC * ok.sum():
                    self.st["cam_skip"] += 1
                    continue
                ok = ok & ~sat
            if o.expo and c in self.target:
                med = float(np.median(raw[mask > 0] if mask is not None else raw))
                if 5 < med < 250:
                    gm = float(np.clip(np.log(self.target[c] / 255) / np.log(med / 255), GAMMA_MIN, GAMMA_MAX))
                    if abs(gm - 1) > 0.05:
                        raw = cv2.LUT(raw, np.clip(255 * (np.arange(256) / 255) ** gm, 0, 255).astype(np.uint8))
                    self.gammas.append(gm)
            if self.clahe is not None:
                raw = self.clahe.apply(raw)
            if (self.W_in, self.H_in) != (self.W, self.H):
                raw = cv2.resize(raw, (self.W_in, self.H_in), interpolation=cv2.INTER_AREA)
            self._ok[c] = ok
            grays.append(raw)
            names.append(c)
        self._dense = {}
        if not names:
            return {}
        n = len(names)
        nb = len(self.names) if o.sp_compile else n                  # fixed batch shape for the compiled graph
        g = np.zeros((nb, self.H_in, self.W_in), np.uint8)
        g[:n] = np.stack(grays)
        x = torch.from_numpy(g).to(DEV)[:, None]
        x = (x.half() if DEV == "cuda" else x.float()).div_(255).contiguous(memory_format=torch.channels_last)
        logits, dense = self.bb(x)
        sc = F.softmax(logits[:n].float(), 1)[:, :-1]
        b, _, h, w = sc.shape
        sc = sc.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        sc = simple_nms(sc, self.sp_conf.nms_radius)
        p = self.sp_conf.remove_borders
        if p:
            sc[:, :p] = -1; sc[:, :, :p] = -1; sc[:, -p:] = -1; sc[:, :, -p:] = -1
        dense = F.normalize(dense[:n].float(), p=2, dim=1)
        out = {}
        for i, c in enumerate(names):
            self._dense[c] = dense[i:i + 1]
            ij = torch.nonzero(sc[i] > self.sp_conf.detection_threshold)
            s = sc[i][ij[:, 0], ij[:, 1]]
            ij, s = top_k_keypoints(ij, s, o.guided_kp)
            k = torch.flip(ij, [1]).float()
            d = sample_descriptors(k[None], dense[i:i + 1], 8)[0].T
            k = (k.cpu().numpy() + 0.5) / self.s_in - 0.5               # -> full-resolution pixels
            s, d = s.cpu().numpy(), d.cpu().numpy()
            ki = np.round(k).astype(int)
            keep = self._ok[c][ki[:, 1].clip(0, self.H - 1), ki[:, 0].clip(0, self.W - 1)]
            out[c] = dict(kps=k[keep].astype(np.float32), desc=d[keep].astype(np.float16))
        return out

    # ------------------------------------------------------------ keyframes
    def candidates(self, cam, T_prior, radius, qdesc=None):
        """map keyframes of camera `cam` to match: DINOv2 near the prior, else the nearest, one per pass"""
        T = T_prior @ self.cams[cam].T_bc
        cm = self.maps[cam]
        if qdesc is not None:
            radius = max(radius, self.o.dino_radius)
            idx = cm.tree.query_ball_point(T[:2, 3], radius)
            h = optical_heading(T)
            idx = [i for i in idx if cm.end[i] > cm.start[i]
                   and abs(wrap(cm.kt.cam_heading.values[i] - h)) < np.deg2rad(self.o.max_dyaw_deg)]
            if not idx:
                return []
            sim = cm.dino[idx] @ qdesc               # most similar looking keyframes near the prior
            return [idx[k] for k in np.argsort(-sim)[:self.o.k_cand]]
        bx, by, byaw = xyyaw(T_prior)
        d = np.hypot(*(cm.base_xy - [bx, by]).T)
        ok = (d < KF_MAX_DIST) & (np.abs(wrap(cm.base_yaw - byaw)) < KF_MAX_DYAW) & (cm.end > cm.start)
        idx = np.where(ok)[0]
        out = []
        for i in idx[np.argsort(d[idx])]:
            if all(abs(int(cm.stamp[i]) - int(cm.stamp[j])) > PASS_GAP_S * 1e9 for j in out):
                out.append(i)
                if len(out) == N_KF_DIST:
                    break
        return out

    def T_prior(self, prior):
        """prior x, y, yaw with z / roll / pitch of the nearest map keyframe"""
        d, i = self.kf_tree.query([prior["x"], prior["y"]])
        if not np.isfinite(d) or d > 2 * KF_MAX_DIST:
            return T_from_xyyaw(prior["x"], prior["y"], prior["yaw"], z=float(np.median(self.kf_T[:, 2, 3])))
        T = self.kf_T[i].copy()
        T[:3, :3] = R.from_euler("z", prior["yaw"] - xyyaw(T)[2]).as_matrix() @ T[:3, :3]
        T[0, 3], T[1, 3] = prior["x"], prior["y"]
        return T

    # ------------------------------------------------------------ dense guided search (v5)
    @torch.no_grad()
    def _dense_match(self, c, uv, D, win, sim_min):
        """uv (n,2) projected map points (full-res px), D (n,256), win (n,) px -> (uv_found (m,2), index (m,))"""
        dm = self._dense.get(c)
        if dm is None or not len(uv):
            return np.zeros((0, 2), np.float32), np.zeros(0, int)
        G, S = self.grid, 2 * self.dense_r + 1
        far = DENSE_FAR * self.W_in / REF_W
        found_all, ok_all = [], []
        for a in range(0, len(uv), 256):                              # chunks: bounded GPU memory
            u = torch.from_numpy(((uv[a:a + 256] + 0.5) * self.s_in - 0.5).astype(np.float32)).to(DEV)
            n = len(u)
            r = torch.from_numpy(np.clip(win[a:a + 256] * self.s_in, 3, self.dense_r).astype(np.float32)).to(DEV)
            pos = u[:, None, :] + G[None]                             # (n, g, 2) in SuperPoint input px
            inside = ((pos[..., 0] >= 4) & (pos[..., 0] <= self.W_in - 5)
                      & (pos[..., 1] >= 4) & (pos[..., 1] <= self.H_in - 5))
            inwin = inside & (G.norm(dim=1)[None] <= r[:, None])
            desc = sample_descriptors(pos.reshape(1, -1, 2), dm, 8)[0]          # (256, n*g)
            Dm = torch.from_numpy(D[a:a + 256].astype(np.float32)).to(DEV)
            sim = torch.einsum("nc,cng->ng", Dm, desc.reshape(-1, n, len(G)))
            sim[~inwin] = -2
            best, bi = sim.max(1)
            farm = (G[None] - G[bi][:, None]).norm(dim=2) > far
            second = torch.where(farm, sim, torch.full_like(sim, -2)).max(1).values
            ok = (best > sim_min) & (best - second > DENSE_MARGIN)
            # sub-pixel: parabola through the neighbours on the grid (x and y separately)
            by, bx = bi // S, bi % S
            s3 = sim.reshape(n, S, S)
            ar = torch.arange(n, device=DEV)

            def para(m, c0, p):
                den = m - 2 * c0 + p
                return torch.where(den < -1e-6, 0.5 * (m - p) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
            c0 = s3[ar, by, bx]
            dx = torch.where((bx > 0) & (bx < S - 1),
                             para(s3[ar, by, (bx - 1).clamp(0)], c0, s3[ar, by, (bx + 1).clamp(max=S - 1)]), 0 * c0)
            dy = torch.where((by > 0) & (by < S - 1),
                             para(s3[ar, (by - 1).clamp(0), bx], c0, s3[ar, (by + 1).clamp(max=S - 1), bx]), 0 * c0)
            found = pos[ar, bi] + torch.stack([dx, dy], 1)
            found_all.append(found.cpu().numpy()); ok_all.append(ok.cpu().numpy())
        found = (np.concatenate(found_all) + 0.5) / self.s_in - 0.5          # -> full-resolution px
        ok = np.concatenate(ok_all)
        fi = np.round(found).astype(int)
        ok &= self._ok[c][fi[:, 1].clip(0, self.H - 1), fi[:, 0].clip(0, self.W - 1)]
        return found[ok].astype(np.float32), np.where(ok)[0]

    # ------------------------------------------------------------ guided correction (v2 + v5)
    @torch.no_grad()
    def _guided(self, qf, prior, qd, sim_min=SIM_MIN, dense_sim=DENSE_SIM):
        T0 = self.T_prior(prior)
        px = self.px
        per, n_kp = {}, 0
        t = time.perf_counter()
        for c, f in qf.items():
            cm, cam = self.maps[c], self.cams[c]
            kfs = self.candidates(c, T0, self.o.radius, qd.get(c))
            if not kfs:
                continue
            X = np.concatenate([cm.keyframe(i)[1] for i in kfs])
            D = np.concatenate([cm.keyframe(i)[0]["desc"] for i in kfs])
            if len(X) < 8:
                continue
            Tc = T0 @ cam.T_bc
            pc = (X - Tc[:3, 3]) @ Tc[:3, :3]
            uv = cam.project(pc)
            vis = np.isfinite(uv[:, 0])
            if vis.sum() < 8:
                continue
            X, D, uv, depth = X[vis], D[vis], uv[vis], pc[vis, 2]
            win = np.clip(cam.f * 2.5 * prior["sig_xy"] / depth + cam.f * 2.5 * prior["sig_yaw"] + WIN_ADD * px,
                          WIN_MIN * px, WIN_MAX * px)
            used = np.zeros(len(X), bool)
            kp_uv, kp_X = np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32)
            if len(f["kps"]) >= 8:
                uv_t = torch.from_numpy(uv.astype(np.float32)).to(DEV)
                k_t = torch.from_numpy(f["kps"]).to(DEV)
                inwin = torch.cdist(uv_t, k_t) < torch.from_numpy(win.astype(np.float32)).to(DEV)[:, None]
                Sm = torch.from_numpy(D).to(DEV).float() @ torch.from_numpy(f["desc"]).to(DEV).float().T
                Sm[~inwin] = -1
                bj = Sm.argmax(1); bi = Sm.argmax(0)          # mutual best: map point <-> query keypoint
                i = torch.arange(len(X), device=DEV)
                ok = ((Sm[i, bj] > sim_min) & (bi[bj] == i)).cpu().numpy()
                bj = bj.cpu().numpy()
                if ok.sum() >= 4:
                    used = ok
                    kp_uv, kp_X = f["kps"][bj[ok]], X[ok]
            n_kp += len(kp_uv)
            per[c] = (X, D, uv, win, used, kp_uv, kp_X)
        t = self._tick("match", t)
        corr, n_dense = {}, 0
        for c, (X, D, uv, win, used, kp_uv, kp_X) in per.items():
            U, XX = [kp_uv], [kp_X]
            if (~used).any():                         # dense search for the map points left unmatched
                rest = np.where(~used)[0]
                fu, j = self._dense_match(c, uv[rest], D[rest], win[rest], dense_sim)
                U.append(fu); XX.append(X[rest[j]]); n_dense += len(fu)
            U, XX = np.concatenate(U), np.concatenate(XX).astype(np.float64)
            if len(U) >= 4:
                corr[c] = (self.cams[c].norm(U), XX)
        t = self._tick("dense", t)
        info = dict(n_kp=n_kp, n_dense=n_dense)
        if sum(len(v[0]) for v in corr.values()) < MIN_INLIERS_GUIDED:
            return None, info
        Tb, thr = T0, THR_PX * px
        for th in (4 * thr, thr):
            r = self.residuals(Tb, corr)
            sel = {c: e < th for c, e in r.items()}
            if sum(s.sum() for s in sel.values()) < MIN_INLIERS_GUIDED:
                return None, info
            Tb = self._refine(Tb, corr, sel, thr)
        r = self.residuals(Tb, corr)
        n_inl = int(sum((e < thr).sum() for e in r.values()))
        self._tick("refine", t)
        if n_inl < MIN_INLIERS_GUIDED:
            return None, info
        return (xyyaw(Tb), n_inl), info

    def residuals(self, Tb, corr):
        """pixel reprojection residual norms per camera"""
        out = {}
        for c, (n, X) in corr.items():
            T = Tb @ self.cams[c].T_bc
            pc = (X - T[:3, 3]) @ T[:3, :3]
            with np.errstate(all="ignore"):
                e = np.linalg.norm(pc[:, :2] / pc[:, 2:] - n, axis=1) * self.cams[c].f
            e[pc[:, 2] <= 0.5] = np.inf
            out[c] = e
        return out

    def _refine(self, Tb, corr, sel, thr, iters=8):
        """robust Gauss-Newton on T_map_base, analytic Jacobian, Huber weights (pixel residuals, all
        cameras). Perturbation: R <- exp([dth]x) R, t <- t + dt (map frame)."""
        n_l, X_l, f_l, tbc_l, Rbc_l = [], [], [], [], []
        for c, (n, X) in corr.items():
            m = sel[c]
            if m.any():
                k, cam = int(m.sum()), self.cams[c]
                n_l.append(n[m]); X_l.append(X[m]); f_l.append(np.full(k, cam.f))
                tbc_l.append(np.repeat(cam.T_bc[None, :3, 3], k, 0))
                Rbc_l.append(np.repeat(cam.T_bc[None, :3, :3], k, 0))
        n, X, f = np.concatenate(n_l), np.concatenate(X_l), np.concatenate(f_l)
        tbc, Rbc = np.concatenate(tbc_l), np.concatenate(Rbc_l)
        Rm, t = Tb[:3, :3].copy(), Tb[:3, 3].copy()
        delta = thr / 2
        for _ in range(iters):
            d = X - t
            q = d @ Rm                                             # R^T d (base frame)
            pc = np.einsum("kji,kj->ki", Rbc, q - tbc)             # Rbc^T (q - tbc)
            z = np.maximum(pc[:, 2], 0.1)
            r = (pc[:, :2] / z[:, None] - n) * f[:, None]
            A = np.einsum("kji,jl->kil", Rbc, Rm.T)                # Rbc^T R^T
            sk = np.zeros((len(d), 3, 3))
            sk[:, 0, 1], sk[:, 0, 2], sk[:, 1, 2] = -d[:, 2], d[:, 1], -d[:, 0]
            sk[:, 1, 0], sk[:, 2, 0], sk[:, 2, 1] = d[:, 2], -d[:, 1], d[:, 0]
            Jp = np.zeros((len(d), 2, 3))
            Jp[:, 0, 0] = Jp[:, 1, 1] = f / z
            Jp[:, 0, 2] = -f * pc[:, 0] / z ** 2
            Jp[:, 1, 2] = -f * pc[:, 1] / z ** 2
            J = np.concatenate([Jp @ A @ sk, -Jp @ A], axis=2)     # (k, 2, 6)
            e = np.linalg.norm(r, axis=1)
            w = np.where(e < delta, 1.0, delta / np.maximum(e, 1e-9))
            Hm = np.einsum("k,kai,kaj->ij", w, J, J)
            g = np.einsum("k,kai,ka->i", w, J, r)
            dx = -np.linalg.solve(Hm + 1e-9 * np.eye(6), g)
            Rm = R.from_rotvec(dx[:3]).as_matrix() @ Rm
            t = t + dx[3:]
            if np.abs(dx).max() < 1e-6:
                break
        T = np.eye(4); T[:3, :3], T[:3, 3] = Rm, t
        return T

    # ------------------------------------------------------------ global fallback (previous path)
    def _global(self, imgs, prior, qd):
        o = self.o
        T_prior = self.T_prior(prior)
        radius = max(GLOBAL_RADIUS_MIN, 4 * prior["sig_xy"])
        per_cam = {}
        for cam, img in imgs.items():
            c = self.cams[cam]
            q = self.feats.extract(img, self.masks[cam])
            uv, X = [], []
            for i in self.candidates(cam, T_prior, radius, qd.get(cam)) if qd else \
                    self._near(cam, T_prior, radius):
                kf, xyz = self.maps[cam].keyframe(i)
                m = self.feats.match(q, kf)
                uv.append(q["kp"][m[:, 0]]); X.append(xyz[m[:, 1]])
            if not uv:
                continue
            uv, X = np.concatenate(uv), np.concatenate(X).astype(np.float64)
            if len(X) < o.min_inliers:
                continue
            n = c.norm(uv)
            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    X, n, np.eye(3), None, iterationsCount=o.pnp_iters,
                    reprojectionError=o.pnp_px / c.f, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:          # degenerate sample (e.g. near-duplicate points)
                continue
            if not ok or inl is None or len(inl) < o.min_inliers:
                continue
            inl = inl.ravel()
            T_cm = np.eye(4)
            T_cm[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cm[:3, 3] = tvec.ravel()
            per_cam[cam] = (np.linalg.inv(T_cm) @ c.T_cb, n[inl], X[inl])
        if not per_cam:
            return None
        best = max(per_cam, key=lambda k: len(per_cam[k][1]))     # joint refinement (rig PnP)
        T0 = per_cam[best][0]
        p0 = np.r_[R.from_matrix(T0[:3, :3]).as_rotvec(), T0[:3, 3]]

        def resid(p):
            T = np.eye(4)
            T[:3, :3] = R.from_rotvec(p[:3]).as_matrix()
            T[:3, 3] = p[3:]
            T_bm = np.linalg.inv(T)
            res = []
            for cam, (_, n, X) in per_cam.items():
                c = self.cams[cam]
                T_cm = c.T_cb @ T_bm
                Xc = X @ T_cm[:3, :3].T + T_cm[:3, 3]
                z = np.maximum(Xc[:, 2:3], 0.1)
                res.append(((Xc[:, :2] / z) - n).ravel() * c.f)
            return np.concatenate(res)

        sol = least_squares(resid, p0, loss="huber", f_scale=o.pnp_px)
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(sol.x[:3]).as_matrix()
        T[:3, 3] = sol.x[3:]
        return xyyaw(T), sum(len(v[1]) for v in per_cam.values())

    def _near(self, cam, T_prior, radius):
        T = T_prior @ self.cams[cam].T_bc
        cm = self.maps[cam]
        idx = cm.tree.query_ball_point(T[:2, 3], radius)
        h = optical_heading(T)
        idx = [i for i in idx if cm.end[i] > cm.start[i]
               and abs(wrap(cm.kt.cam_heading.values[i] - h)) < np.deg2rad(self.o.max_dyaw_deg)]
        d = [np.hypot(*(cm.kt[["cx", "cy"]].values[i] - T[:2, 3])) for i in idx]
        return [idx[k] for k in np.argsort(d)[:self.o.k_cand]]

    # ------------------------------------------------------------ one camera fix
    def localize(self, t, images, prior):
        """images: {cam: path}, prior: TrackEKF.prior(). -> ((x, y, yaw), n_inliers) or None, info"""
        self.st["n_fix"] += 1
        t0 = time.perf_counter()
        cams = [c for c in self.names if c in images]
        imgs = dict(zip(cams, self.pool.map(lambda c: cv2.imread(str(images[c])), cams)))
        imgs = {c: im for c, im in imgs.items() if im is not None}
        if self.transform is not None:
            imgs = {c: self.transform(im) for c, im in imgs.items()}
        t0 = self._tick("read jpeg", t0)
        qd = {}
        if self.dino is not None and imgs:       # one batch for all cameras of this set
            with autocast():
                qd = dict(zip(imgs, self.dino(list(imgs.values()))))
            t0 = self._tick("dino", t0)
        lost = prior["sig_xy"] > LOST_SIGMA or self.fails >= LOST_FAILS
        if lost and self.last_global is not None and t - self.last_global < GLOBAL_MIN_S * 1e9:
            # too soon for another global attempt: keep trying the guided path from the (wider) prior
            self.st["global_skip"] += 1
            res, info = self._guided_relaxed(imgs, dict(prior, sig_xy=min(prior["sig_xy"], 0.99 * LOST_SIGMA)), qd)
            info["mode"] = "guided-wide"
        elif lost:
            self.last_global = t
            self.st["global_"] += 1
            res, info = self._global(imgs, prior, qd), dict(mode="global", n_kp=0, n_dense=0)
            self._tick("global", t0)
            if res is not None and not self._on_track(res):
                res = None
            self.st["global_ok"] += res is not None
            if res is not None:
                self.fails = 0
            return res, info
        else:
            res, info = self._guided_relaxed(imgs, prior, qd)
        self.fails = 0 if res is not None else self.fails + 1
        return res, info

    def _guided_relaxed(self, imgs, prior, qd):
        """guided fix (v5), retried once with lower similarity thresholds (v6), drivable-area check"""
        t = time.perf_counter()
        qf = self._features(imgs)
        self._tick("superpoint", t)
        self.st["guided"] += 1
        res, info = self._guided(qf, prior, qd)
        info["mode"] = "guided"
        if res is None and self.o.relax and qf:
            self.st["relax"] += 1
            res, info = self._guided(qf, prior, qd, RELAX_SIM_MIN, RELAX_DENSE_SIM)
            info["mode"] = "relax"
            self.st["relax_ok"] += res is not None
        if res is not None and not self._on_track(res):
            res = None
        self.st["guided_fail"] += res is None
        return res, info

    def _on_track(self, res):
        if self.drivable is None or self.drivable.inside(*res[0][:2]):
            return True
        self.st["bounds_rej"] += 1
        return False

    def summary(self):
        g = np.array(self.gammas) if self.gammas else np.array([1.0])
        return dict(self.st, gamma_median=round(float(np.median(g)), 3), pruned_pct=self.pruned)


# ----------------------------------------------------------------------------- validate
def make_degrade(gain, gamma, noise, blur, seed=0):
    """Photometric stress test on query images only (the map is untouched).
    gain: brightness multiplier, gamma: >1 darker / <1 brighter mid-tones,
    noise: Gaussian sigma in 0..255 units, blur: horizontal motion-blur length in px."""
    if gain == 1 and gamma == 1 and noise == 0 and blur <= 1:
        return None
    rng = np.random.default_rng(seed)
    kernel = np.full((1, int(blur)), 1.0 / int(blur), np.float32) if blur > 1 else None

    def f(img):
        x = img.astype(np.float32) / 255.0
        x = np.clip(x * gain, 0, 1) ** gamma
        if kernel is not None:
            x = cv2.filter2D(x, -1, kernel)
        if noise > 0:
            x = x + rng.normal(0, noise / 255.0, x.shape).astype(np.float32)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return f


def row(t, pose, gt, lap, **kw):
    ex, ey = pose[0] - gt[0], pose[1] - gt[1]
    return dict(timestamp_ns=t, x=pose[0], y=pose[1], yaw=pose[2], lap=lap, **kw,
                err=np.hypot(ex, ey), err_lat=-np.sin(gt[2]) * ex + np.cos(gt[2]) * ey,
                err_yaw=np.degrees(wrap(pose[2] - gt[2])))


def due_fix(a, car, t, since):
    """v2/v3 scheduling: a correction every --fix-every frames, every --fix-every-corner in corners"""
    need = a.fix_every
    if a.fix_every_corner and abs(car.yaw_rate(t)) > a.corner_wz:
        need = a.fix_every_corner
    return since + 1 >= need


def sync_lap(a, loc, car, cal, g, data, lap):
    """camera fix inside the frame (deterministic): the pose of frame k waits for its own fix"""
    rows, ekf, t_last, since = [], None, None, 10 ** 6
    lap_t0 = int(g.stamp_ns.min())
    for k, (_, s) in enumerate(g.groupby("t_bin")):
        t_frame = time.perf_counter()
        last = s.loc[s.stamp_ns.idxmax()]               # pose is stamped at the newest image
        t = int(last.stamp_ns)
        gt = (last.x, last.y, last.yaw)
        if k == 0:
            ekf = TrackEKF(*gt, calib=cal)              # first pose of the "run" is allowed
        else:
            car.propagate(ekf, t_last, t)
        t_last = t
        t_lap = (t - lap_t0) * 1e-9
        dropped = a.drop_every > 0 and t_lap >= a.drop_every and (t_lap % a.drop_every) < a.drop_len
        prior = ekf.prior()
        if a.prior == "gt":
            prior.update(x=gt[0], y=gt[1], yaw=gt[2])
        due = due_fix(a, car, t, since)
        res, info, d2, ok, dt = None, {}, np.nan, False, 0.0
        if due and not dropped and not a.odom_only:
            since = 0
            t0 = time.perf_counter()
            res, info = loc.localize(t, {r.cam: data / r.path for r in s.itertuples() if r.cam in loc.cams}, prior)
            dt = (time.perf_counter() - t0) * 1000
            if res is not None:
                ok, d2 = ekf.update(*res[0], res[1])
        else:
            since += 1
        rows.append(row(t, ekf.pose, gt, lap, ok=ok, dropped=dropped, n_inl=res[1] if res else 0, ms=dt,
                        total_ms=(time.perf_counter() - t_frame) * 1000, mode=info.get("mode", ""),
                        n_kp=info.get("n_kp", 0), n_dense=info.get("n_dense", 0), sig_xy=prior["sig_xy"],
                        speed_scale=float(ekf.X[3]), gyro_bias=float(ekf.X[4]), kf_d2=d2))
        if k % 100 == 0:
            print(f"lap {lap} frame {k}: err {rows[-1]['err']:.2f} m, ok {ok}, {info.get('mode', '-')}, "
                  f"inl {rows[-1]['n_inl']}, {dt:.0f} ms")
    return rows


def async_lap(a, loc, car, cal, g, data, lap):
    """Live-like: the camera fix runs in a background worker (v1 delayed measurement).

    Frame k arrives at its stamp t_k; its pose is output at once from the filter (car state + the fixes
    FINISHED by t_k). A fix started on frame m finishes at t_m + (measured wall time of the fix); the filter
    is then rewound to its snapshot at t_m, updated, and propagated again to the current frame. Poses already
    output are never changed. One worker: a new fix starts on the first frame after the previous one finished."""
    rows, hist, pending, since = [], [], None, 10 ** 6
    for k, (_, s) in enumerate(g.groupby("t_bin")):
        t_wall = time.perf_counter()
        last = s.loc[s.stamp_ns.idxmax()]
        t = int(last.stamp_ns)
        gt = (last.x, last.y, last.yaw)
        if k == 0:
            ekf = TrackEKF(*gt, calib=cal)
        else:
            car.propagate(ekf, hist[-1][0], t)
        hist.append((t, copy.deepcopy(ekf)))
        applied, d2, n_inl, delay = False, np.nan, 0, np.nan
        if pending is not None and pending["ready"] <= t:          # a fix has finished
            km, res = pending["k"], pending["res"]
            delay = (t - hist[km][0]) / 1e6
            if res is not None:
                f = copy.deepcopy(hist[km][1])
                applied, d2 = f.update(*res[0], res[1])
                hist[km] = (hist[km][0], copy.deepcopy(f))
                for j in range(km + 1, k + 1):                    # replay the car state to "now"
                    car.propagate(f, hist[j - 1][0], hist[j][0])
                    hist[j] = (hist[j][0], copy.deepcopy(f))
                ekf, n_inl = f, res[1]
            pending = None
        pose = ekf.pose                                            # output now, before any new fix
        main_ms = (time.perf_counter() - t_wall) * 1000
        job_ms, info = 0.0, {}
        if pending is None and due_fix(a, car, t, since):          # worker is free: start a new fix
            since = 0
            tw = time.perf_counter()
            res, info = loc.localize(t, {r.cam: data / r.path for r in s.itertuples() if r.cam in loc.cams},
                                     ekf.prior())
            job_ms = (time.perf_counter() - tw) * 1000
            pending = {"k": k, "res": res, "ready": t + int(job_ms * 1e6)}
        else:
            since += 1
        rows.append(row(t, pose, gt, lap, ok=applied, dropped=False, n_inl=n_inl, ms=job_ms,
                        total_ms=main_ms + job_ms, main_ms=main_ms, fix_delay_ms=delay,
                        mode=info.get("mode", ""), n_kp=info.get("n_kp", 0), n_dense=info.get("n_dense", 0),
                        sig_xy=ekf.prior()["sig_xy"], speed_scale=float(ekf.X[3]), gyro_bias=float(ekf.X[4]),
                        kf_d2=d2))
        if k % 100 == 0:
            print(f"lap {lap} frame {k}: err {rows[-1]['err']:.2f} m, fix applied {applied}"
                  + (f" (delay {delay:.0f} ms)" if applied else "")
                  + f", worker {job_ms:.0f} ms, output {main_ms:.1f} ms")
    return rows


def validate(a):
    global FP16
    if a.fast_gpu and DEV == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True        # picks the fastest conv kernels per shape
        FP16 = True
        print("fast GPU: TF32 + cudnn.benchmark, DINOv2 / fallback SuperPoint in fp16")
    data = Path(a.data)
    if not a.state:
        raise SystemExit("the v7 filter needs the car state: --state data/state.parquet")
    car = CarState(a.state, a.speed_col, a.vy_col, a.yawrate_col, a.yawrate_scale)
    calp = a.motion_calib or (data / "motion_calib.json")
    cal = None
    if not a.no_motion_calib and Path(calp).exists():
        cal = json.loads(Path(calp).read_text())
        print(f"motion calibration: vx*{cal['vx_k']:.4f}, vy = {cal['vy_a']:.3f}*vy {cal['vy_b']:+.3f}*wz "
              f"{cal['vy_c']:+.3f}, wz*{cal['wz_s']:.4f}{cal['wz_d']:+.5f}")
    else:
        print("motion calibration: off (car state taken as is)")
    loc = Localizer(data, a.map, a.cams, o=a)
    print("query body masks: " + ", ".join(c for c, m in loc.masks.items() if m is not None)
          if any(m is not None for m in loc.masks.values()) else "query body masks: off")
    print(f"guided: SuperPoint {loc.W_in}x{loc.H_in}, {a.guided_kp} kp/camera, nms {a.nms_radius}, "
          f"window {WIN_MIN * loc.px:.0f}-{WIN_MAX * loc.px:.0f} px, inlier {THR_PX * loc.px:.1f} px, "
          f"dense radius {loc.dense_r} px (SuperPoint input); expo {a.expo}, sat {a.sat}, relax {a.relax}, "
          f"bounds {loc.drivable is not None}, clahe {a.clahe}")
    loc.transform = make_degrade(a.gain, a.gamma, a.noise, a.blur)
    stress = []
    if loc.transform is not None:
        stress.append(f"images: gain {a.gain}, gamma {a.gamma}, noise {a.noise}, blur {a.blur}px")
    if a.drop_every > 0:
        stress.append(f"no camera fixes for {a.drop_len}s every {a.drop_every}s")
    print("STRESS TEST: " + "; ".join(stress) if stress else "no stress test")
    print(f"camera fix every {a.fix_every} frame(s)"
          + (f", every {a.fix_every_corner} in corners" if a.fix_every_corner else "")
          + (", async (background worker)" if a.async_fix else ", sync"))
    if a.prior == "gt":
        print("WARNING: --prior gt uses ground truth at inference. Debug only, never for results.")
    f = load_frames(data, laps=a.laps, cams=sorted(loc.names))
    f["t_bin"] = (f.t_rel * 10).round().astype(int)
    rows = []
    for lap in a.laps:
        loc.reset()
        g = f[f.lap == lap]
        rows += (async_lap if a.async_fix else sync_lap)(a, loc, car, cal, g, data, lap)

    r = pd.DataFrame(rows)
    for l in a.laps:
        e = r[r.lap == l].iloc[-1]
        print(f"lap {l}: speed scale k {e.speed_scale:.4f}, yaw-rate bias {np.degrees(e.gyro_bias):+.4f} deg/s")
    # runtime report: whole per-frame step (images, features, matching, pose, filter)
    tm = r.total_ms.values
    warm = tm[1:] if len(tm) > 1 else tm                  # first frame = GPU warm-up
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    period = float(np.median(np.concatenate([np.diff(np.sort(r[r.lap == l].timestamp_ns.values))
                                             for l in a.laps])) / 1e6)
    lag, lags = 0.0, []
    for p in warm:                              # live playback: frames queue while we are busy
        lag = max(0.0, lag - period) + p
        lags.append(lag)
    lags = np.array(lags)
    print(f"runtime on {gpu}, camera period {period:.1f} ms (first frame = warm-up, excluded):")
    print(f"  per frame: mean {warm.mean():.1f} ms | median {np.median(warm):.1f} | "
          f"p95 {np.percentile(warm, 95):.1f} | max {warm.max():.1f} | "
          f"{(warm > period).mean():.1%} of frames longer than one period")
    print(f"  REAL-TIME (mean <= period): {'YES' if warm.mean() <= period else 'NO'} "
          f"({warm.mean() / period:.0%} of the period)")
    print(f"  live playback lag: mean {lags.mean():.1f} ms, p95 {np.percentile(lags, 95):.1f}, "
          f"max {lags.max():.1f} ms -> {'bounded, keeps up' if lags[-200:].mean() < 3 * period else 'GROWING, cannot keep up'}")
    loc.profile_report()
    if a.async_fix:
        dl = r.fix_delay_ms.dropna()
        print(f"  ASYNC: pose output latency mean {r.main_ms.mean():.2f} ms, max {r.main_ms.max():.1f} ms | "
              f"camera fixes applied {r.ok.sum()} ({r.ok.mean():.0%} of frames), "
              f"delay image->applied mean {dl.mean():.0f} ms, max {dl.max():.0f} ms")
    print("v7 localizer:", json.dumps(loc.summary()))
    used = r[r.n_inl > 0]
    if len(used):
        print(f"  median inliers {int(used.n_inl.median())}, keypoint matches {int(used.n_kp.median())}, "
              f"dense matches {int(used.n_dense.median())}")
    for l in a.laps:
        t = np.sort(r[r.lap == l].timestamp_ns.values)
        sec = (t - t[0]) // 1_000_000_000
        print(f"lap {l}: {len(t)} poses, median step {np.median(np.diff(t)) / 1e6:.1f} ms, "
              f"1 s bins with a pose {len(np.unique(sec)) / (sec.max() + 1):.1%}")
    r[["timestamp_ns", "x", "y", "yaw"]].to_csv(a.out, index=False)
    r.to_csv(Path(a.out).with_suffix(".diag.csv"), index=False)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    for name, d in [("all", r)] + [(f"lap {l}", r[r.lap == l]) for l in a.laps]:
        print(f"{name:>6}: pos RMSE {rms(d.err):.3f} m | lat RMSE {rms(d.err_lat):.3f} m | "
              f"yaw RMSE {rms(d.err_yaw):.3f} deg | <1m {(d.err < 1).mean():.0%} | "
              f"fix rate {d.ok.mean():.0%} | lost(>5m) {(d.err > 5).mean():.0%} | "
              f"max {d.err.max():.2f} m | {d.ms.mean():.0f} ms/frame")


# ----------------------------------------------------------------------------- cli
def common(p):
    p.add_argument("--data", required=True)
    p.add_argument("--laps", nargs="+", type=int, required=True)
    p.add_argument("--cams", nargs="+", default=None)
    p.add_argument("--max-kp", type=int, default=1024)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    common(b)
    b.add_argument("--out", required=True)
    b.add_argument("--kf-dist", type=float, default=3.0, help="m between keyframes")
    b.add_argument("--kf-yaw", type=float, default=5.0, help="deg between keyframes")
    b.add_argument("--pair-span", type=int, default=2, help="match each kf with N previous")
    b.add_argument("--max-gap-s", type=float, default=5.0)
    b.add_argument("--min-flow-px", type=float, default=2.0)
    b.add_argument("--min-depth", type=float, default=2.0)
    b.add_argument("--max-depth", type=float, default=100.0)
    b.add_argument("--max-reproj-px", type=float, default=2.0)
    b.add_argument("--min-parallax-deg", type=float, default=1.0)
    b.add_argument("--max-pts-per-kf", type=int, default=400)
    b.add_argument("--limit", type=int, default=0, help="debug: only first N keyframes")

    v = sub.add_parser("validate")
    common(v)
    v.add_argument("--map", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--prior", choices=["track", "gt"], default="track")
    # map keyframes
    v.add_argument("--k-cand", type=int, default=3, help="map keyframes per camera (DINOv2)")
    v.add_argument("--radius", type=float, default=20.0, help="m, keyframe search radius")
    v.add_argument("--max-dyaw-deg", type=float, default=45.0)
    v.add_argument("--dino", action="store_true",
                   help="choose map keyframes by DINOv2 similarity (run dino_desc.py first); "
                        "without it: the 3 nearest keyframes, one per pass (v5)")
    v.add_argument("--dino-model", choices=["dinov2", "dinov3"], default="dinov2")
    v.add_argument("--dino-radius", type=float, default=30.0,
                   help="m around the predicted pose in which DINOv2 picks keyframes")
    # guided path (v2 / v3 / v5)
    v.add_argument("--guided-kp", type=int, default=1024, help="SuperPoint keypoints per camera (v5)")
    v.add_argument("--nms-radius", type=int, default=3, help="SuperPoint NMS (hloc map: 3)")
    v.add_argument("--sp-resize", type=int, default=1024,
                   help="long side (px) of the image SuperPoint sees; the map was built at 1024")
    v.add_argument("--sp-compile", action="store_true", help="torch.compile of the SuperPoint backbone (v3)")
    v.add_argument("--clahe", action="store_true",
                   help="CLAHE on the query grey images (only if the map was built with CLAHE; hloc: no)")
    v.add_argument("--no-query-masks", action="store_true",
                   help="ignore data/masks/<cam>.png for query images")
    v.add_argument("--cam-yaw-offsets", default=None,
                   help="json {cam: deg} from cam_yaw_offsets.py, corrects camera extrinsic yaw")
    # v6 / v7
    v.add_argument("--no-expo", dest="expo", action="store_false", help="v6: no exposure normalisation")
    v.add_argument("--no-sat", dest="sat", action="store_false", help="v6: no saturation mask")
    v.add_argument("--no-relax", dest="relax", action="store_false", help="v6: no relaxed retry")
    v.add_argument("--no-bounds", dest="bounds", action="store_false", help="v6: no drivable-area check")
    v.add_argument("--prune-peak", type=int, default=200,
                   help="v7: drop map points whose 5x5 keyframe grey peak is >= this (0 = keep all, v6)")
    # global fallback (previous path: LightGlue + PnP-RANSAC)
    v.add_argument("--min-inliers", type=int, default=20)
    v.add_argument("--pnp-px", type=float, default=3.0)
    v.add_argument("--pnp-iters", type=int, default=300, help="RANSAC iterations per camera")
    v.add_argument("--lg-mp", action="store_true", help="LightGlue in mixed precision (fp16)")
    # filter / car state
    v.add_argument("--state", default=None, help="data/state.parquet from extract_state.py (required)")
    v.add_argument("--speed-col", default="vx_mps", help="body-frame forward speed (vx)")
    v.add_argument("--vy-col", default="vy_mps", help="body-frame lateral speed (sideslip)")
    v.add_argument("--yawrate-col", default="wz_radps")
    v.add_argument("--yawrate-scale", type=float, default=1.0, help="e.g. 0.01745 if deg/s")
    v.add_argument("--motion-calib", default=None, help="json from calib_motion.py (default: data/motion_calib.json)")
    v.add_argument("--no-motion-calib", action="store_true", help="car state taken as is")
    v.add_argument("--odom-only", action="store_true", help="no map fixes: pure dead reckoning")
    # scheduling
    v.add_argument("--fix-every", type=int, default=1, help="camera fix every N frames (v7: 1)")
    v.add_argument("--fix-every-corner", type=int, default=0,
                   help="in corners (|yaw rate| > --corner-wz) fix every N frames instead")
    v.add_argument("--corner-wz", type=float, default=0.10, help="rad/s")
    v.add_argument("--async-fix", action="store_true",
                   help="live-like: camera fixes run in a background worker and are applied "
                        "late (rewind + replay); poses are output immediately")
    v.add_argument("--profile", action="store_true", help="time per stage of a camera fix")
    v.add_argument("--fast-gpu", action="store_true",
                   help="TF32 + cudnn.benchmark, DINOv2 and the fallback SuperPoint under fp16 autocast")
    # stress tests (query images / fixes only; the map is never changed)
    v.add_argument("--drop-every", type=float, default=0, help="s; simulate camera outages")
    v.add_argument("--drop-len", type=float, default=2.0, help="s; length of each outage")
    v.add_argument("--gain", type=float, default=1.0, help="brightness multiplier")
    v.add_argument("--gamma", type=float, default=1.0, help=">1 darker, <1 brighter")
    v.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma, 0..255")
    v.add_argument("--blur", type=float, default=0.0, help="horizontal motion blur, px")

    a = ap.parse_args()
    if a.cmd == "build":
        a.cams = a.cams or ["fl", "fr", "r"]
        build(a)
    else:
        validate(a)