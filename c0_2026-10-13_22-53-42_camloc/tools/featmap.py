#!/usr/bin/env python3
"""
Method 3a: 3D feature map from the training run + multi-camera PnP relocalization.

Build the map (offline; GT poses of the selected laps are allowed here):
    python featmap.py build --data data/ --laps 1 2 3 5 6 --cams fl fr r --out maps/v1
    python featmap.py build ... --limit 300          # quick debug run

Validate on held-out laps (causal: GT only for the first pose of each lap and for metrics):
    python featmap.py validate --data data/ --map maps/v1 --laps 4 7 --out val_4_7.csv
    python featmap.py validate ... --prior gt        # DEBUG ONLY: tests matching in isolation

Design:
  * one map per camera: a query from camera "fr" is matched against keyframes from "fr",
    so viewpoints are similar and matching is easy;
  * every map keyframe stores its SuperPoint keypoints that were triangulated, their
    descriptors and 3D points in the map frame;
  * at query time, the prior pose selects nearby keyframes of the same camera, LightGlue
    gives 2D-3D matches, PnP-RANSAC runs per camera, then one joint refinement of the
    base_link pose uses the inliers of all cameras (known extrinsics = a camera rig).
"""
import argparse, json, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from lightglue import SuperPoint, LightGlue

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
        self.T_bc = T_base_cam(calib, name)       # cam -> base_link
        self.T_cb = np.linalg.inv(self.T_bc)

    def norm(self, uv):
        """pixels -> undistorted normalized image coordinates"""
        if len(uv) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64),
                                   self.K, self.D).reshape(-1, 2)


def load_mask(data, cam):
    p = Path(data) / "masks" / f"{cam}.png"    # optional: white = usable pixels
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None


# ----------------------------------------------------------------------------- features
class Features:
    def __init__(self, max_kp=1024, mp=False):
        self.sp = SuperPoint(max_num_keypoints=max_kp).eval().to(DEV)
        try:                           # mp = mixed precision (fp16) inside LightGlue
            self.lg = LightGlue(features="superpoint", mp=mp).eval().to(DEV)
        except TypeError:
            self.lg = LightGlue(features="superpoint").eval().to(DEV)
        self.transform = None          # optional image degradation for stress tests

    @torch.no_grad()
    def extract(self, path, mask=None):
        img = path if isinstance(path, np.ndarray) else cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if self.transform is not None:
            img = self.transform(img)
        t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(DEV)
        f = self.sp.extract(t)                   # keypoints are returned in original pixels
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
    def extract_batch(self, imgs, masks):
        """SuperPoint on all cameras in ONE forward pass. Same preprocessing as
        SuperPoint.extract (grayscale, long side resized to 1024, keypoints scaled back).
        Falls back to one-by-one if the batch cannot be stacked."""
        sizes = {im.shape[:2] for im in imgs}
        if len(imgs) == 1 or len(sizes) != 1 or self.transform is not None:
            return [self.extract(im, m) for im, m in zip(imgs, masks)]
        H, W = imgs[0].shape[:2]
        x = torch.from_numpy(np.stack(imgs)).to(DEV)              # B,H,W,3 uint8 BGR
        x = x.permute(0, 3, 1, 2).float().div(255)
        gray = (0.114 * x[:, 0] + 0.587 * x[:, 1] + 0.299 * x[:, 2])[:, None]
        s = 1024 / max(H, W)
        h2, w2 = (int(round(H * s)), int(round(W * s))) if s < 1 else (H, W)
        if (h2, w2) != (H, W):
            gray = torch.nn.functional.interpolate(gray, size=(h2, w2), mode="bilinear",
                                                   align_corners=False, antialias=True)
        try:
            f = self.sp({"image": gray})
        except RuntimeError:                   # different keypoint counts cannot be stacked
            return [self.extract(im, m) for im, m in zip(imgs, masks)]
        sc_xy = torch.tensor([w2 / W, h2 / H], device=gray.device)
        out = []
        for b in range(len(imgs)):
            kp = ((f["keypoints"][b] + 0.5) / sc_xy - 0.5).cpu().numpy()
            de = f["descriptors"][b].cpu().numpy()
            sc = f["keypoint_scores"][b].cpu().numpy()
            m = masks[b]
            if m is not None:
                u = kp[:, 0].astype(int).clip(0, m.shape[1] - 1)
                v = kp[:, 1].astype(int).clip(0, m.shape[0] - 1)
                keep = m[v, u] > 0
                kp, de, sc = kp[keep], de[keep], sc[keep]
            out.append({"kp": kp.astype(np.float32), "desc": de.astype(np.float16),
                        "score": sc.astype(np.float32), "size": np.array([W, H])})
        return out

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


# ----------------------------------------------------------------------------- localize
class CamMap:
    def __init__(self, map_dir, cam):
        z = np.load(Path(map_dir) / f"{cam}.npz")
        self.kt = pd.read_parquet(Path(map_dir) / f"{cam}_kf.parquet")
        order = np.argsort(z["kf"], kind="stable")
        self.kp, self.desc, self.xyz = z["kp"][order], z["desc"][order], z["xyz"][order]
        kf = z["kf"][order]
        self.start = np.searchsorted(kf, np.arange(len(self.kt)))
        self.end = np.searchsorted(kf, np.arange(len(self.kt)), side="right")
        self.tree = cKDTree(self.kt[["cx", "cy"]].values)
        self.size = None
        self.dino = None

    def keyframe(self, i):
        s, e = self.start[i], self.end[i]
        return {"kp": self.kp[s:e], "desc": self.desc[s:e], "size": self.size}, self.xyz[s:e]


class Localizer:
    def __init__(self, data, map_dir, cams=None, feats=None, o=None):
        meta = json.loads((Path(map_dir) / "meta.json").read_text())
        calib = json.loads((Path(data) / "calib.json").read_text())
        self.o = o
        self.names = cams or meta["cams"]
        self.cams = {c: Cam(calib, c) for c in self.names}
        self.maps = {c: CamMap(map_dir, c) for c in self.names}
        for c in self.names:
            cc = calib["cameras"][c]
            self.maps[c].size = np.array([cc["width"], cc["height"]])
        self.masks = {c: load_mask(data, c) for c in self.names}
        offs = getattr(o, "cam_yaw_offsets", None)
        if offs:
            offs = json.loads(Path(offs).read_text())
            for c, deg in offs.items():
                if c in self.cams:      # camera c reports base yaw too large by deg -> rotate it
                    Rz = T_from_xyyaw(0, 0, np.radians(deg))
                    self.cams[c].T_bc = Rz @ self.cams[c].T_bc
                    self.cams[c].T_cb = np.linalg.inv(self.cams[c].T_bc)
            print("camera yaw offsets applied:", {c: round(v, 4) for c, v in offs.items()})
        self.z0 = float(np.median(self.maps[self.names[0]].kt.z))
        self.feats = feats or Features(o.max_kp, mp=getattr(o, "lg_mp", False))
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(8)       # JPEG decoding of all cameras in parallel
        self.prof = {} if getattr(o, "profile", False) else None
        self.n_fix = 0
        self.dino = None
        if getattr(o, "dino", False):
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
            print(f"{o.dino_model} candidate selection: radius {o.dino_radius} m, "
                  f"top {o.k_cand} per camera")

    def candidates(self, cam, T_prior, radius, qdesc=None):
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
        if qdesc is not None:                   # most similar looking keyframes near the prior
            sim = cm.dino[idx] @ qdesc
            return [idx[k] for k in np.argsort(-sim)[:self.o.k_cand]]
        d = [np.hypot(*(cm.kt[["cx", "cy"]].values[i] - T[:2, 3])) for i in idx]
        return [idx[k] for k in np.argsort(d)[:self.o.k_cand]]

    def _tick(self, name, t0):
        """accumulate stage time (ms) when profiling; GPU work is synchronised first"""
        if self.prof is None:
            return time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.prof[name] = self.prof.get(name, 0.0) + (t1 - t0) * 1000
        return t1

    def profile_report(self):
        if not self.prof or not self.n_fix:
            return
        tot = sum(self.prof.values())
        print(f"time per camera fix ({self.n_fix} fixes), ms:")
        for k, v in sorted(self.prof.items(), key=lambda x: -x[1]):
            print(f"  {k:12s} {v / self.n_fix:7.1f}   ({v / tot:.0%})")
        print(f"  {'total':12s} {tot / self.n_fix:7.1f}")

    def localize(self, images, prior_xyyaw, radius):
        """images: {cam: path}. Returns (x, y, yaw), n_inliers, info  or  None."""
        self.n_fix += 1
        t = time.perf_counter()
        T_prior = T_from_xyyaw(*prior_xyyaw, z=self.z0)
        per_cam = {}
        cams = [c for c in images if c in self.cams]
        imgs = dict(zip(cams, self.pool.map(lambda c: cv2.imread(str(images[c])), cams)))
        t = self._tick("read jpeg", t)
        qd = {}
        if self.dino is not None and cams:      # one batch for all cameras of this set
            qd = dict(zip(cams, self.dino([imgs[c] for c in cams])))
            t = self._tick("dino", t)
        if getattr(self.o, "sp_batch", False):
            qs = dict(zip(cams, self.feats.extract_batch([imgs[c] for c in cams],
                                                        [self.masks[c] for c in cams])))
            t = self._tick("superpoint", t)
        for cam in cams:
            c = self.cams[cam]
            if getattr(self.o, "sp_batch", False):
                q = qs[cam]
            else:
                q = self.feats.extract(imgs[cam], self.masks[cam])
                t = self._tick("superpoint", t)
            uv, X = [], []
            for i in self.candidates(cam, T_prior, radius, qd.get(cam)):
                kf, xyz = self.maps[cam].keyframe(i)
                m = self.feats.match(q, kf)
                uv.append(q["kp"][m[:, 0]]); X.append(xyz[m[:, 1]])
            t = self._tick("lightglue", t)
            if not uv:
                continue
            uv, X = np.concatenate(uv), np.concatenate(X).astype(np.float64)
            if len(X) < self.o.min_inliers:
                continue
            n = c.norm(uv)
            gate = getattr(self.o, "pnp_prior_gate", 0.0)
            if gate > 0:
                # the filter prior is accurate to ~0.1 m: keep matches that agree with it and
                # skip RANSAC; the joint robust refinement below does the rest
                T_cm0 = c.T_cb @ np.linalg.inv(T_prior)
                Xc = X @ T_cm0[:3, :3].T + T_cm0[:3, 3]
                front = Xc[:, 2] > 0.5
                err = np.full(len(X), np.inf)
                err[front] = np.linalg.norm(Xc[front, :2] / Xc[front, 2:3] - n[front], axis=1) * c.f
                inl = np.flatnonzero(err < gate)
                if len(inl) >= 2 * self.o.min_inliers:
                    rmax = getattr(self.o, "refine_max_pts", 0)
                    if rmax and len(inl) > rmax:
                        inl = inl[np.linspace(0, len(inl) - 1, rmax).astype(int)]
                    per_cam[cam] = (T_prior, n[inl], X[inl])
                    t = self._tick("pnp gate", t)
                    continue                      # else: prior is off, fall back to RANSAC
            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    X, n, np.eye(3), None, iterationsCount=300,
                    reprojectionError=self.o.pnp_px / c.f, confidence=0.999,
                    flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:          # degenerate sample (e.g. near-duplicate points)
                t = self._tick("pnp ransac", t)
                continue
            t = self._tick("pnp ransac", t)
            if not ok or inl is None or len(inl) < self.o.min_inliers:
                continue
            inl = inl.ravel()
            rmax = getattr(self.o, "refine_max_pts", 0)
            if rmax and len(inl) > rmax:           # evenly spread subset for the refinement
                inl = inl[np.linspace(0, len(inl) - 1, rmax).astype(int)]
            T_cm = np.eye(4)
            T_cm[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cm[:3, 3] = tvec.ravel()
            T_mb = np.linalg.inv(T_cm) @ c.T_cb
            per_cam[cam] = (T_mb, n[inl], X[inl])
        if not per_cam:
            return None
        self._t_refine = t

        # joint refinement of base_link pose over all cameras (rig PnP)
        best = max(per_cam, key=lambda k: len(per_cam[k][1]))
        T0 = per_cam[best][0]                      # = prior pose for prior-gated cameras
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

        sol = least_squares(resid, p0, loss="huber", f_scale=self.o.pnp_px)
        if getattr(self.o, "pnp_prior_gate", 0.0) > 0:     # trim outliers, refine once more
            r_ = np.abs(resid(sol.x)).reshape(-1, 2).max(1)
            off = 0
            for cam in list(per_cam):
                T_, n_, X_ = per_cam[cam]
                keep = r_[off:off + len(n_)] < 2 * self.o.pnp_px
                off += len(n_)
                if keep.sum() >= self.o.min_inliers:
                    per_cam[cam] = (T_, n_[keep], X_[keep])
                else:
                    del per_cam[cam]
            if not per_cam:
                return None
            sol = least_squares(resid, sol.x, loss="huber", f_scale=self.o.pnp_px)
        self._tick("refine", self._t_refine)
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(sol.x[:3]).as_matrix()
        T[:3, 3] = sol.x[3:]
        n_inl = sum(len(v[1]) for v in per_cam.values())
        info = {k: {"inl": len(v[1]), "yaw": xyyaw(v[0])[2]} for k, v in per_cam.items()}
        return xyyaw(T), n_inl, info


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


def make_odometry(path, vcol, rcol, rscale, vycol=None, calib=None):
    """Dead reckoning from car-state speed and yaw rate (allowed inputs), strictly causal:
    each segment uses the last state sample at or before its start (sample-and-hold).
    calib (json from calib_motion.py): lever-arm / scale corrected motion model."""
    st = pd.read_parquet(path).sort_values("stamp_ns")
    ts = st.stamp_ns.values
    v = st[vcol].values.astype(float)
    r = st[rcol].values.astype(float) * rscale
    vy = st[vycol].values.astype(float) if vycol else np.zeros_like(v)   # body-frame lateral
    if calib:
        c = json.loads(Path(calib).read_text())
        wz_raw = r.copy()
        v = c["vx_k"] * v
        vy = c["vy_a"] * vy + c["vy_b"] * wz_raw + c["vy_c"]
        r = c["wz_s"] * wz_raw + c["wz_d"]
        print(f"motion calibration: vx*{c['vx_k']:.4f}, vy = {c['vy_a']:.3f}*vy "
              f"{c['vy_b']:+.3f}*wz {c['vy_c']:+.3f}, wz*{c['wz_s']:.4f}{c['wz_d']:+.5f}")

    def propagate(pose, t0, t1, bias=0.0, scale=1.0):
        """bias: yaw-rate offset to subtract (rad/s), scale: speed scale factor."""
        tt = np.r_[t0, ts[(ts > t0) & (ts < t1)], t1]
        i = np.clip(np.searchsorted(ts, tt[:-1], side="right") - 1, 0, len(ts) - 1)
        x, y, yaw = pose
        for vi, wi, ri, d in zip(v[i] * scale, vy[i] * scale, r[i] - bias, np.diff(tt) * 1e-9):
            ym = yaw + 0.5 * ri * d
            c, s_ = np.cos(ym), np.sin(ym)
            x, y, yaw = x + (vi * c - wi * s_) * d, y + (vi * s_ + wi * c) * d, yaw + ri * d
        return x, y, wrap(yaw)

    def yaw_rate(t):                       # last car-state yaw rate at or before t (causal)
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))
        return float(r[i])
    propagate.yaw_rate = yaw_rate
    return propagate


class PoseKF:
    """Extended Kalman filter on [x, y, yaw, yaw-rate bias, speed scale].

    predict: dead reckoning with vx, vy, wz (allowed car-state inputs), bias and scale applied;
             process noise grows per metre driven (along / across the car) and per second (yaw).
    update:  camera pose (x, y, yaw) with noise that shrinks with the number of PnP inliers;
             a Mahalanobis gate rejects fixes that disagree with the prediction.
    Causal by construction: only past state samples and the current fix are used."""

    def __init__(self, o):
        self.o = o

    def init(self, pose):
        self.x = np.array([pose[0], pose[1], pose[2], 0.0, 1.0])
        bias_var = np.radians(0.05) ** 2 if self.o.kf_bias else 0.0   # 0 = bias fixed at 0
        self.P = np.diag([0.1 ** 2, 0.1 ** 2, np.radians(0.2) ** 2, bias_var, 0.005 ** 2])

    @property
    def pose(self):
        return float(self.x[0]), float(self.x[1]), float(wrap(self.x[2]))

    def predict(self, odom, t0, t1):
        o = self.o
        x, y, yaw, b, sc = self.x
        nx, ny, nyaw = odom((x, y, yaw), t0, t1, b, sc)
        dt = (t1 - t0) * 1e-9
        dx, dy = nx - x, ny - y
        dist = float(np.hypot(dx, dy))
        F = np.eye(5)
        F[0, 2], F[1, 2] = -dy, dx          # position depends on heading
        F[2, 3] = -dt                       # heading depends on yaw-rate bias
        F[0, 4], F[1, 4] = dx / sc, dy / sc  # position depends on speed scale
        c, s_ = np.cos(yaw), np.sin(yaw)
        Rb = np.array([[c, -s_], [s_, c]])
        Qb = np.diag([o.q_lon ** 2 * dist + 1e-8, o.q_lat ** 2 * dist + 1e-8])
        Q = np.zeros((5, 5))
        Q[:2, :2] = Rb @ Qb @ Rb.T
        Q[2, 2] = np.radians(o.q_yaw) ** 2 * dt
        Q[3, 3] = np.radians(o.q_bias) ** 2 * dt if o.kf_bias else 0.0
        Q[4, 4] = o.q_scale ** 2 * dt
        self.P = F @ self.P @ F.T + Q
        self.x = np.array([nx, ny, nyaw, b, sc])

    def update(self, z, n_inl):
        o = self.o
        w = float(np.clip(np.sqrt(o.kf_ref_inl / max(n_inl, 1)), 0.5, 3.0))
        R = np.diag([(o.r_pos * w) ** 2, (o.r_pos * w) ** 2, np.radians(o.r_yaw * w) ** 2])
        H = np.zeros((3, 5)); H[:, :3] = np.eye(3)
        y = np.array([z[0] - self.x[0], z[1] - self.x[1], wrap(z[2] - self.x[2])])
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > o.kf_gate:
            return False, d2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        bmax = np.radians(o.kf_bias_max)
        self.x[3] = np.clip(self.x[3], -bmax, bmax)     # the gyro is good: keep the bias small
        self.x[4] = np.clip(self.x[4], 0.97, 1.03)
        self.P = (np.eye(5) - K @ H) @ self.P
        return True, d2

    def reset(self, z):
        """Lost: jump to the camera pose and widen the uncertainty."""
        self.x[:3] = z
        self.P[:3, :] = 0.0
        self.P[:, :3] = 0.0                             # forget correlations with the old state
        self.P[:3, :3] = np.diag([0.3 ** 2, 0.3 ** 2, np.radians(0.5) ** 2])


def async_lap(a, loc, odom, g, data, lap):
    """Live-like simulation of one lap: camera localization runs in a background worker.

    Frame k arrives at its stamp t_k. The pose for t_k is output immediately from the filter
    (odometry + all camera fixes that are FINISHED by t_k). A fix started on frame m with images
    of t_m finishes at t_m + (measured wall time of the fix); it is then applied as a delayed
    measurement: the filter is rewound to t_m, updated, and re-propagated with the odometry to
    the current frame. Poses already output are never changed. One worker: a new fix starts on
    the first frame after the previous one finished (and at least --fix-every frames apart)."""
    kf = PoseKF(a)
    hist, rows = [], []
    pending, since_req, fails = None, 10 ** 6, 0
    for k, (_, s) in enumerate(g.groupby("t_bin")):
        t_wall = time.perf_counter()
        last = s.loc[s.stamp_ns.idxmax()]
        t = int(last.stamp_ns)
        gt = (last.x, last.y, last.yaw)
        if k == 0:
            kf.init(gt)                                   # first pose of the run
        else:
            kf.predict(odom, hist[-1]["t"], t)
        hist.append({"t": t, "x": kf.x.copy(), "P": kf.P.copy()})

        applied, d2, n_inl, delay = False, np.nan, 0, np.nan
        if pending is not None and pending["ready"] <= t:          # a fix has finished
            km, res = pending["k"], pending["res"]
            delay = (t - hist[km]["t"]) / 1e6                     # ms from image to use
            if res:
                keep_x, keep_P = kf.x.copy(), kf.P.copy()
                kf.x, kf.P = hist[km]["x"].copy(), hist[km]["P"].copy()
                ok_, d2 = kf.update(res[0], res[1])
                if not ok_:
                    fails += 1
                    if fails >= a.kf_reset_after:
                        kf.reset(res[0]); ok_, fails = True, 0
                else:
                    fails = 0
                if ok_:                                   # replay odometry to "now"
                    hist[km]["x"], hist[km]["P"] = kf.x.copy(), kf.P.copy()
                    for j in range(km + 1, k + 1):
                        kf.predict(odom, hist[j - 1]["t"], hist[j]["t"])
                        hist[j]["x"], hist[j]["P"] = kf.x.copy(), kf.P.copy()
                    applied, n_inl = True, res[1]
                else:
                    kf.x, kf.P = keep_x, keep_P
            else:
                fails += 1
            pending = None

        pose = kf.pose                                    # output now, before any new fix
        main_ms = (time.perf_counter() - t_wall) * 1000

        job_ms = 0.0
        since_req += 1
        if pending is None and since_req >= a.fix_every:  # worker is free: start a new fix
            tw = time.perf_counter()
            res = loc.localize({r.cam: data / r.path for r in s.itertuples() if r.cam in loc.cams},
                               pose, a.radius)
            job_ms = (time.perf_counter() - tw) * 1000
            pending = {"k": k, "res": res, "ready": t + int(job_ms * 1e6)}
            since_req = 0

        ex, ey = pose[0] - gt[0], pose[1] - gt[1]
        rows.append(dict(timestamp_ns=t, x=pose[0], y=pose[1], yaw=pose[2], lap=lap,
                         ok=applied, dropped=False, n_inl=n_inl, ms=job_ms,
                         total_ms=main_ms + job_ms, main_ms=main_ms, fix_delay_ms=delay,
                         border_ok=False, border_dlat=0.0, border_ms=0.0,
                         gyro_bias=float(kf.x[3]), speed_scale=float(kf.x[4]),
                         kf_d2=d2, err=np.hypot(ex, ey),
                         err_lat=-np.sin(gt[2]) * ex + np.cos(gt[2]) * ey,
                         err_yaw=np.degrees(wrap(pose[2] - gt[2]))))
        if k % 100 == 0:
            print(f"lap {lap} frame {k}: err {rows[-1]['err']:.2f} m, fix applied {applied}"
                  + (f" (delay {delay:.0f} ms)" if applied else "")
                  + f", worker {job_ms:.0f} ms, output {main_ms:.1f} ms")
    return rows


def validate(a):
    data = Path(a.data)
    loc = Localizer(data, a.map, a.cams, o=a)
    if a.no_query_masks:
        loc.masks = {c: None for c in loc.names}
    print("query body masks: " + ", ".join(c for c, m in loc.masks.items() if m is not None)
          if any(m is not None for m in loc.masks.values()) else "query body masks: off")
    loc.feats.transform = make_degrade(a.gain, a.gamma, a.noise, a.blur)
    stress = []
    if loc.feats.transform is not None:
        stress.append(f"images: gain {a.gain}, gamma {a.gamma}, noise {a.noise}, blur {a.blur}px")
    if a.drop_every > 0:
        stress.append(f"no camera fixes for {a.drop_len}s every {a.drop_every}s")
    print("STRESS TEST: " + "; ".join(stress) if stress else "no stress test")
    odom = make_odometry(a.state, a.speed_col, a.yawrate_col, a.yawrate_scale, a.vy_col,
                         a.motion_calib) if a.state else None
    if a.filter == "kf" and odom is None:
        raise SystemExit("--filter kf needs odometry: pass --state and the speed / yaw-rate columns")
    if a.filter == "kf":
        print(f"Kalman filter: q_lon {a.q_lon} q_lat {a.q_lat} m/sqrt(m), q_yaw {a.q_yaw} deg/sqrt(s), "
              f"r_pos {a.r_pos} m, r_yaw {a.r_yaw} deg @ {a.kf_ref_inl} inliers, gate {a.kf_gate}")
    print(f"prior: {'GT (debug)' if a.prior == 'gt' else 'odometry' if odom else 'constant velocity'}"
          + (f", camera fix every {a.fix_every} frames" if a.fix_every > 1 else "")
          + (f", blend pos {a.alpha_pos} yaw {a.alpha_yaw}"
             if odom is not None and (a.alpha_pos < 1 or a.alpha_yaw < 1) else ""))
    border = None
    if a.borders:
        import borders as B
        bc = json.loads(Path(a.borders).read_text())
        bcams = a.border_cams or bc["cams"]
        bmask = a.border_masks or bc["masks"]
        calib = json.loads((data / "calib.json").read_text())
        sid, aid, stid = B.surface_and_avoid_ids(data, bmask, bc["surface"], bc["avoid"],
                                                 bc.get("structure", []))
        border = B.BorderCorrector({c: Cam(calib, c) for c in bcams}, B.BorderMap(B.load_bounds(data)),
                                   bc["ground_z"], sid, aid, {c: load_mask(data, c) for c in bcams},
                                   edge_mode=bc.get("edge_mode", "lateral"), structure_ids=stid)
        print(f"track borders: cams {bcams}, masks data/{bmask}, every {a.border_every} frames, "
              f"alpha lat {a.border_alpha} yaw {a.border_yaw_alpha} lon {a.border_lon_alpha}")
    groups = [g.split(",") for g in a.cam_groups.split(";")] if a.cam_groups else None
    if groups:
        unknown = {c for g in groups for c in g} - set(loc.names)
        if unknown:
            raise SystemExit(f"--cam-groups: cameras not in the map: {sorted(unknown)}")
        print(f"camera groups, one per frame in turn: {groups}")
    f = load_frames(data, laps=a.laps, cams=sorted(set(loc.names) | set(border.cams if border else [])))
    f["t_bin"] = (f.t_rel * 10).round().astype(int)
    rows = []
    if a.prior == "gt":
        print("WARNING: --prior gt uses ground truth at inference. Debug only, never for results.")

    if a.async_fix and (a.filter != "kf" or odom is None):
        raise SystemExit("--async-fix needs --filter kf and odometry (--state ...)")
    for lap in a.laps:
        g = f[f.lap == lap]
        if a.async_fix:
            rows += async_lap(a, loc, odom, g, data, lap)
            continue
        hist, fails = [], 0
        bias, scale, last_fix = 0.0, 1.0, None       # online yaw-rate bias / speed scale
        kf = PoseKF(a) if a.filter == "kf" else None
        since_fix = 10 ** 6
        kf_d2 = np.nan
        lap_t0 = int(g.stamp_ns.min())
        for k, (_, s) in enumerate(g.groupby("t_bin")):
            t_frame = time.perf_counter()                # full per-frame processing time
            last = s.loc[s.stamp_ns.idxmax()]            # pose is stamped at the newest image
            t_lap = (int(last.stamp_ns) - lap_t0) * 1e-9
            dropped = (a.drop_every > 0 and t_lap >= a.drop_every
                       and (t_lap % a.drop_every) < a.drop_len)
            gt = (last.x, last.y, last.yaw)
            if kf is not None:
                if k == 0:
                    kf.init(gt)                             # first pose of the "run" is allowed
                else:
                    kf.predict(odom, hist[-1][0], int(last.stamp_ns))
                prior = kf.pose
            elif k == 0 or a.prior == "gt":
                prior = gt                                  # first pose of the "run" is allowed
            elif odom is not None:
                prior = odom(tuple(hist[-1][1:]), hist[-1][0], int(last.stamp_ns), bias, scale)
            elif len(hist) >= 2:
                (t1, *p1), (t2, *p2) = hist[-2], hist[-1]
                r = (last.stamp_ns - t2) / max(t2 - t1, 1)
                prior = (p2[0] + r * (p2[0] - p1[0]), p2[1] + r * (p2[1] - p1[1]),
                         p2[2] + r * wrap(p2[2] - p1[2]))
            else:
                prior = tuple(hist[-1][1:])

            radius = min(a.radius * (1 + fails), 4 * a.radius)
            t0 = time.perf_counter()
            if a.fix_every_corner and odom is not None:
                # corners: correct more often (odometry drifts most there), straights: less
                corner = abs(odom.yaw_rate(int(last.stamp_ns))) > a.corner_wz
                need = a.fix_every_corner if corner else a.fix_every
                skip = since_fix + 1 < need
                since_fix = 0 if not skip else since_fix + 1
            else:
                skip = a.fix_every > 1 and k % a.fix_every != 0   # cheaper: fix every N frames
            use = loc.cams if not groups else set(groups[k % len(groups)])   # round-robin cameras
            res = None if (a.odom_only or dropped or skip) else loc.localize(
                {r.cam: data / r.path for r in s.itertuples() if r.cam in use}, prior, radius)
            dt = (time.perf_counter() - t0) * 1000
            gate = min(a.gate_m * (1 + fails), a.gate_max_m)
            if kf is not None:
                ok = False
                if res:
                    ok, kf_d2 = kf.update(res[0], res[1])
                    if ok:
                        fails = 0
                    else:
                        fails += 1
                        if fails >= a.kf_reset_after:          # many rejections in a row: lost
                            kf.reset(res[0]); ok, fails = True, 0
                elif not skip:
                    fails += 1
                pose = kf.pose
                bias, scale = float(kf.x[3]), float(kf.x[4])
            elif res and np.hypot(res[0][0] - prior[0], res[0][1] - prior[1]) < gate:
                # blend the camera fix with the odometry prediction (simple complementary
                # filter): yaw rate is very precise short-term, PnP yaw is noisier
                ap_, ay_ = (a.alpha_pos, a.alpha_yaw) if (odom is not None and k > 0) else (1, 1)
                px, py, pyaw = prior
                cx, cy, cyaw = res[0]
                pose = (px + ap_ * (cx - px), py + ap_ * (cy - py),
                        wrap(pyaw + ay_ * wrap(cyaw - pyaw)))
                ok, fails = True, 0
                # learn the odometry errors from the camera fix (causal: past fixes only)
                if odom is not None and last_fix is not None:
                    dt_fix = (int(last.stamp_ns) - last_fix[0]) * 1e-9
                    if dt_fix > 0:
                        e_yaw = wrap(cyaw - pyaw)              # >0: odometry turned too little
                        bias = float(np.clip(bias - a.bias_gain * e_yaw / dt_fix, -0.02, 0.02))
                    dist = np.hypot(px - last_fix[1], py - last_fix[2])
                    if dist > 1.0:
                        e_lon = np.cos(pyaw) * (cx - px) + np.sin(pyaw) * (cy - py)
                        scale = float(np.clip(scale * (1 + a.scale_gain * e_lon / dist), 0.97, 1.03))
                last_fix = (int(last.stamp_ns), pose[0], pose[1])
            elif skip:
                pose, ok = prior, False                         # planned skip is not a failure
            else:
                pose, ok, fails = prior, False, fails + 1
            b_ok, b_dlat, b_ms = False, 0.0, 0.0
            if border is not None and not dropped and k % a.border_every == 0:
                tb = time.perf_counter()
                labs = {r.cam: B.load_labels(data, bmask, r.cam, r.stamp_ns)
                        for r in s.itertuples() if r.cam in border.cams}
                pts = border.ground_points(labs)
                bo = border.search(pose, pts)
                if bo and bo["n"] >= a.border_min_pts and bo["cost"] < a.border_max_cost:
                    x_, y_, yw = pose
                    dl, dn = a.border_alpha * bo["dlat"], a.border_lon_alpha * bo["dlon"]
                    pose = (x_ + dn * np.cos(yw) - dl * np.sin(yw),
                            y_ + dn * np.sin(yw) + dl * np.cos(yw),
                            wrap(yw + a.border_yaw_alpha * bo["dyaw"]))
                    b_ok, b_dlat = True, bo["dlat"]
                b_ms = (time.perf_counter() - tb) * 1000
            hist.append((int(last.stamp_ns), *pose))
            total_ms = (time.perf_counter() - t_frame) * 1000
            ex, ey = pose[0] - gt[0], pose[1] - gt[1]
            rows.append(dict(timestamp_ns=int(last.stamp_ns), x=pose[0], y=pose[1], yaw=pose[2],
                             lap=lap, ok=ok, dropped=dropped, n_inl=res[1] if res else 0, ms=dt,
                             total_ms=total_ms,
                             border_ok=b_ok, border_dlat=b_dlat, border_ms=b_ms,
                             gyro_bias=bias, speed_scale=scale, kf_d2=kf_d2 if res else np.nan,
                             **{f"yawerr_{c}": (np.degrees(wrap(v["yaw"] - gt[2])) if res else np.nan)
                                for c, v in (res[2].items() if res else [])},
                             err=np.hypot(ex, ey),
                             err_lat=-np.sin(gt[2]) * ex + np.cos(gt[2]) * ey,
                             err_yaw=np.degrees(wrap(pose[2] - gt[2]))))
            if k % 100 == 0:
                print(f"lap {lap} frame {k}: err {rows[-1]['err']:.2f} m, ok {ok}, "
                      f"inl {rows[-1]['n_inl']}, {dt:.0f} ms")

    r = pd.DataFrame(rows)
    if odom is not None and (a.bias_gain > 0 or a.scale_gain > 0 or a.filter == "kf"):
        for l in a.laps:
            e = r[r.lap == l].iloc[-1]
            print(f"lap {l}: learned yaw-rate bias {np.degrees(e.gyro_bias):+.4f} deg/s, "
                  f"speed scale {e.speed_scale:.4f}")
    # runtime report: whole per-frame step (images, features, matching, PnP, filter)
    tm = r.total_ms.values
    warm = tm[1:] if len(tm) > 1 else tm                  # first frame = GPU warm-up
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    period = float(np.median(np.concatenate([np.diff(np.sort(r[r.lap == l].timestamp_ns.values))
                                             for l in a.laps])) / 1e6)
    # live playback: frames arrive every `period` ms and wait in a queue while we are busy.
    # lag = how late the pose of a frame is ready after the frame arrived.
    lag, lags = 0.0, []
    for p in warm:
        lag = max(0.0, lag - period) + p        # queue left from earlier frames + this frame
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
    for l in a.laps:
        t = np.sort(r[r.lap == l].timestamp_ns.values)
        sec = (t - t[0]) // 1_000_000_000
        print(f"lap {l}: {len(t)} poses, median step {np.median(np.diff(t)) / 1e6:.1f} ms, "
              f"1 s bins with a pose {len(np.unique(sec)) / (sec.max() + 1):.1%}")
    r[["timestamp_ns", "x", "y", "yaw"]].to_csv(a.out, index=False)
    r.to_csv(Path(a.out).with_suffix(".diag.csv"), index=False)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    for name, d in [("all", r)] + [(f"lap {l}", r[r.lap == l]) for l in a.laps]:
        print(f"{name:>6}: pos RMSE {rms(d.err):.2f} m | lat RMSE {rms(d.err_lat):.2f} m | "
              f"yaw RMSE {rms(d.err_yaw):.2f} deg | <1m {(d.err < 1).mean():.0%} | "
              f"fix rate {d.ok.mean():.0%} | lost(>5m) {(d.err > 5).mean():.0%} | "
              f"max {d.err.max():.1f} m | {d.ms.mean():.0f} ms/frame"
              + (f" | border fixes {d.border_ok.mean():.0%}, +{d.border_ms.mean():.0f} ms"
                 if border is not None else ""))


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
    v.add_argument("--k-cand", type=int, default=3, help="map keyframes per camera")
    v.add_argument("--radius", type=float, default=20.0, help="m, keyframe search radius")
    v.add_argument("--max-dyaw-deg", type=float, default=45.0)
    v.add_argument("--min-inliers", type=int, default=20)
    v.add_argument("--pnp-px", type=float, default=3.0)
    v.add_argument("--gate-m", type=float, default=8.0, help="reject fixes this far from prior")
    v.add_argument("--gate-max-m", type=float, default=30.0, help="cap for the growing gate")
    v.add_argument("--state", default=None, help="data/state.parquet from extract_state.py")
    v.add_argument("--speed-col", default=None, help="body-frame forward speed (vx)")
    v.add_argument("--yawrate-col", default=None)
    v.add_argument("--vy-col", default=None, help="body-frame lateral speed (sideslip), optional")
    v.add_argument("--yawrate-scale", type=float, default=1.0, help="e.g. 0.01745 if deg/s")
    v.add_argument("--odom-only", action="store_true", help="no map fixes: pure dead reckoning")
    v.add_argument("--alpha-pos", type=float, default=1.0,
                   help="weight of the camera fix for x,y (1 = take it as is), needs --state")
    v.add_argument("--alpha-yaw", type=float, default=1.0,
                   help="weight of the camera fix for yaw (e.g. 0.2 = trust yaw rate more)")
    v.add_argument("--dino", action="store_true",
                   help="choose map keyframes by DINOv2 similarity (run dino_desc.py first)")
    v.add_argument("--dino-model", choices=["dinov2", "dinov3"], default="dinov2")
    v.add_argument("--dino-radius", type=float, default=30.0,
                   help="m around the predicted pose in which DINOv2 picks keyframes")
    v.add_argument("--cam-groups", default=None,
                   help='cameras per frame in turn, e.g. "fr,rl;cl,rr;cr,r,fl" (use with --filter kf)')
    v.add_argument("--filter", choices=["blend", "kf"], default="blend",
                   help="blend: fixed weights (--alpha-*); kf: Kalman filter (needs --state)")
    v.add_argument("--q-lon", type=float, default=0.01, help="KF along-track noise, m/sqrt(m)")
    v.add_argument("--q-lat", type=float, default=0.01, help="KF across-track noise, m/sqrt(m)")
    v.add_argument("--q-yaw", type=float, default=0.03, help="KF heading noise, deg/sqrt(s)")
    v.add_argument("--q-bias", type=float, default=0.005, help="KF yaw-rate bias walk, deg/s/sqrt(s)")
    v.add_argument("--q-scale", type=float, default=0.0005, help="KF speed-scale walk, 1/sqrt(s)")
    v.add_argument("--r-pos", type=float, default=0.08, help="KF camera position noise, m")
    v.add_argument("--r-yaw", type=float, default=0.10, help="KF camera heading noise, deg")
    v.add_argument("--kf-ref-inl", type=float, default=600, help="inliers at which r_* apply")
    v.add_argument("--kf-gate", type=float, default=16.0, help="chi2 gate (3 dof), 16 = 99.9%%")
    v.add_argument("--kf-reset-after", type=int, default=3, help="rejections in a row -> reset")
    v.add_argument("--kf-bias-max", type=float, default=0.1, help="limit of yaw-rate bias, deg/s")
    v.add_argument("--kf-no-bias", dest="kf_bias", action="store_false",
                   help="do not estimate the yaw-rate bias (gyro taken as is)")
    v.add_argument("--bias-gain", type=float, default=0.0,
                   help="learn yaw-rate bias from camera fixes (e.g. 0.05), needs --state")
    v.add_argument("--scale-gain", type=float, default=0.0,
                   help="learn speed scale from camera fixes (e.g. 0.05), needs --state")
    v.add_argument("--cam-yaw-offsets", default=None,
                   help="json {cam: deg} from cam_yaw_offsets.py, corrects camera extrinsic yaw")
    v.add_argument("--no-query-masks", action="store_true",
                   help="ignore data/masks/<cam>.png for query images")
    v.add_argument("--profile", action="store_true", help="time per stage of a camera fix")
    v.add_argument("--lg-mp", action="store_true", help="LightGlue in mixed precision (fp16)")
    v.add_argument("--sp-batch", action="store_true", help="SuperPoint on all cameras in one batch")
    v.add_argument("--pnp-prior-gate", type=float, default=0.0,
                   help="px; >0: keep matches consistent with the filter prior and skip RANSAC "
                        "(falls back to RANSAC if too few agree)")
    v.add_argument("--refine-max-pts", type=int, default=0,
                   help="max inliers per camera in the joint refinement (0 = all)")
    v.add_argument("--async-fix", action="store_true",
                   help="live-like: camera fixes run in a background worker and are applied "
                        "late (rewind + replay); poses are output immediately (needs --filter kf)")
    v.add_argument("--motion-calib", default=None, help="json from calib_motion.py")
    v.add_argument("--fix-every-corner", type=int, default=0,
                   help="in corners (|yaw rate| > --corner-wz) fix every N frames instead")
    v.add_argument("--corner-wz", type=float, default=0.10, help="rad/s")
    v.add_argument("--fix-every", type=int, default=1,
                   help="run camera localization only every N frames (needs --state)")
    # track-border correction (borders.py); road masks from data/<border-masks>/<cam>/<stamp>.png
    v.add_argument("--borders", default=None, help="data/borders_calib.json from borders.py")
    v.add_argument("--border-masks", default=None, help="mask folder under data/ (default: calib)")
    v.add_argument("--border-cams", nargs="+", default=None)
    v.add_argument("--border-every", type=int, default=1)
    v.add_argument("--border-alpha", type=float, default=0.5, help="weight of lateral correction")
    v.add_argument("--border-yaw-alpha", type=float, default=0.3)
    v.add_argument("--border-lon-alpha", type=float, default=0.0, help="along-track: weak, off")
    v.add_argument("--border-min-pts", type=int, default=100)
    v.add_argument("--border-max-cost", type=float, default=0.35, help="m, mean edge distance")
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