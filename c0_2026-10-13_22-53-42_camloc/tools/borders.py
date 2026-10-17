#!/usr/bin/env python3
"""
Track-border correction: road edges seen by the cameras vs. the track border map.

Idea (method 2 of the INTRO): pixels on the edge of the road mask are projected onto the
ground plane with the camera calibration; for the correct pose they lie on the map borders.
A small search around the current pose (lateral offset, along-track offset, yaw) finds the
pose where the projected edge points are closest to the borders. Lateral offset and yaw are
well observed; the along-track offset only in corners, so it is not used by default.

    # 1) once: ground height + per-camera check on a training lap with GT (offline analysis)
    python borders.py calibrate --data data/ --masks sam3 --laps 1 --cams fl fr cl cr

    # 2) use it in validation (featmap.py), e.g. odometry + borders only, no feature map:
    python featmap.py validate ... --odom-only --borders data/borders_calib.json --border-masks sam3

Road masks: here the SAM 3 masks (offline). At inference a fast segmentation network trained
on these masks has to produce them; SAM 3 itself is too slow for every frame.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np


def load_bounds(data):
    return [np.asarray(p, dtype=np.float64) for _, p in
            json.loads((Path(data) / "track_bounds.json").read_text())]


class BorderMap:
    """Truncated distance (m) to the nearest track / pit-lane border, bilinear lookup."""

    def __init__(self, bounds, res=0.2, pad=60.0, trunc=1.5):
        allp = np.concatenate(bounds)
        self.x0, self.y0 = allp.min(0) - pad
        x1, y1 = allp.max(0) + pad
        self.res, self.trunc = res, trunc
        self.W, self.H = int((x1 - self.x0) / res) + 2, int((y1 - self.y0) / res) + 2
        img = np.full((self.H, self.W), 255, np.uint8)
        for b in bounds:
            px = np.round((b - [self.x0, self.y0]) / res).astype(np.int32)
            cv2.polylines(img, [px.reshape(-1, 1, 2)], False, 0, 1)
        dt = cv2.distanceTransform(img, cv2.DIST_L2, 5) * res
        self.dt = np.minimum(dt, trunc).astype(np.float32)

    def dist(self, x, y):
        u = (x - self.x0) / self.res
        v = (y - self.y0) / self.res
        u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        ok = (u0 >= 0) & (v0 >= 0) & (u0 < self.W - 1) & (v0 < self.H - 1)
        out = np.full(u.shape, self.trunc, np.float32)
        u0, v0, du, dv = u0[ok], v0[ok], (u - np.floor(u))[ok], (v - np.floor(v))[ok]
        d = self.dt
        out[ok] = ((1 - du) * (1 - dv) * d[v0, u0] + du * (1 - dv) * d[v0, u0 + 1]
                   + (1 - du) * dv * d[v0 + 1, u0] + du * dv * d[v0 + 1, u0 + 1])
        return out


class BorderCorrector:
    def __init__(self, cams, bmap, ground_z, surface_ids, avoid_ids=(), body_masks=None,
                 rmin=4.0, rmax=35.0, max_pts=1500):
        self.cams = cams                       # {name: featmap.Cam}
        self.bmap = bmap
        self.g = ground_z                      # road plane height in base_link frame, m
        self.surface_ids = list(surface_ids)
        self.avoid_ids = list(avoid_ids)
        self.body = {}
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        for c, m in (body_masks or {}).items():
            if m is not None:
                self.body[c] = cv2.dilate((m == 0).astype(np.uint8), k) > 0
        self.rmin, self.rmax, self.max_pts = rmin, rmax, max_pts
        self.k3 = np.ones((3, 3), np.uint8)
        self.k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def edge_pixels(self, cam, lab):
        m = np.isin(lab, self.surface_ids).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k7)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k7)
        edge = (m - cv2.erode(m, self.k3)) > 0
        edge[:3], edge[-3:], edge[:, :3], edge[:, -3:] = False, False, False, False
        if cam in self.body:                       # road cut by our own car is not a border
            edge &= ~self.body[cam]
        if self.avoid_ids:                         # road cut by another car is not a border
            occ = cv2.dilate(np.isin(lab, self.avoid_ids).astype(np.uint8), self.k7, iterations=2)
            edge &= occ == 0
        v, u = np.nonzero(edge)
        return np.c_[u, v].astype(np.float64)

    def ground_points(self, labels):
        """labels: {cam: label image}. Returns (N, 2) points in base_link (x fwd, y left)."""
        out = []
        for cam, lab in labels.items():
            if cam not in self.cams or lab is None:
                continue
            c = self.cams[cam]
            uv = self.edge_pixels(cam, lab)
            if len(uv) == 0:
                continue
            n = c.norm(uv)
            d = np.c_[n, np.ones(len(n))] @ c.T_bc[:3, :3].T
            o = c.T_bc[:3, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (self.g - o[2]) / d[:, 2]
            ok = (d[:, 2] < -1e-3) & (lam > 0)
            P = o + lam[ok, None] * d[ok]
            r = np.hypot(P[:, 0], P[:, 1])
            out.append(P[(r > self.rmin) & (r < self.rmax), :2])
        pts = np.concatenate(out) if out else np.zeros((0, 2))
        if len(pts) > self.max_pts:
            pts = pts[np.random.default_rng(0).choice(len(pts), self.max_pts, replace=False)]
        return pts

    def cost(self, pose, pts):
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        return float(self.bmap.dist(x + c * pts[:, 0] - s * pts[:, 1],
                                    y + s * pts[:, 0] + c * pts[:, 1]).mean())

    def search(self, pose, pts, lat=0.6, lon=1.0, yaw_deg=1.0, refine=True):
        """Coarse grid search in the car frame around pose, then a finer grid around the best
        cell. Returns the total correction (dlat, dlon, dyaw) or None."""
        out = self._grid(pose, pts, lat, lon, yaw_deg, 13, 5, 11)
        if out is None or not refine:
            return out
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        p1 = (x + out["dlon"] * c - out["dlat"] * s, y + out["dlon"] * s + out["dlat"] * c,
              yaw + out["dyaw"])
        fine = self._grid(p1, pts, lat / 6, lon / 4, yaw_deg / 5, 11, 5, 11)
        c1, s1 = np.cos(out["dyaw"]), np.sin(out["dyaw"])
        fl, fn = fine["dlat"], fine["dlon"]
        fine["dlat"] = out["dlat"] + s1 * fn + c1 * fl
        fine["dlon"] = out["dlon"] + c1 * fn - s1 * fl
        fine["dyaw"] = out["dyaw"] + fine["dyaw"]
        fine["cost_prior"] = out["cost_prior"]
        return fine

    def _grid(self, pose, pts, lat, lon, yaw_deg, n_lat, n_lon, n_yaw):
        if len(pts) < 50:
            return None
        x, y, yaw = pose
        L, N, Y = [g.ravel() for g in np.meshgrid(
            np.linspace(-lat, lat, n_lat), np.linspace(-lon, lon, n_lon),
            np.deg2rad(np.linspace(-yaw_deg, yaw_deg, n_yaw)), indexing="ij")]
        c, s = np.cos(yaw), np.sin(yaw)
        X, Yp, T = x + N * c - L * s, y + N * s + L * c, yaw + Y
        ct, st = np.cos(T)[:, None], np.sin(T)[:, None]
        px = X[:, None] + ct * pts[None, :, 0] - st * pts[None, :, 1]
        py = Yp[:, None] + st * pts[None, :, 0] + ct * pts[None, :, 1]
        cost = self.bmap.dist(px, py).mean(1)
        b = int(np.argmin(cost))
        c0 = self.cost(pose, pts)
        # how sharp is the minimum across the track: cost rise at +-lat/2 lateral
        same = (N == N[b]) & (Y == Y[b])
        lat_profile = cost[same]
        sharp = float(np.max(lat_profile) - cost[b])
        return {"dlat": float(L[b]), "dlon": float(N[b]), "dyaw": float(Y[b]),
                "cost": float(cost[b]), "cost_prior": c0, "sharp": sharp, "n": len(pts)}


def load_labels(data, mask_dir, cam, stamp):
    p = Path(data) / mask_dir / cam / f"{stamp}.png"
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None


def surface_and_avoid_ids(data, mask_dir, surface, avoid):
    cls = json.loads((Path(data) / mask_dir / "classes.json").read_text())
    ids = {c["name"]: c["id"] for c in cls}
    return [ids[n] for n in surface if n in ids], [ids[n] for n in avoid if n in ids]


# ----------------------------------------------------------------------------- calibrate
def calibrate(a):
    from featmap import Cam, load_mask
    from lap_index import load_frames
    data = Path(a.data)
    calib = json.loads((data / "calib.json").read_text())
    cams = {c: Cam(calib, c) for c in a.cams}
    bmap = BorderMap(load_bounds(data))
    sid, aid = surface_and_avoid_ids(data, a.masks, a.surface, a.avoid)
    f = load_frames(data, laps=a.laps, cams=a.cams)
    stamps = sorted(f.stamp_ns.unique())[::a.every][:a.max_frames]
    f = f[f.stamp_ns.isin(stamps)]
    body = {c: load_mask(data, c) for c in a.cams}

    # cache label images once
    labs = {(r.cam, r.stamp_ns): load_labels(data, a.masks, r.cam, r.stamp_ns) for r in f.itertuples()}
    print(f"{len(stamps)} frame sets, masks found for "
          f"{sum(v is not None for v in labs.values())}/{len(labs)} images")

    def mean_cost(g, cam_list):
        corr = BorderCorrector({c: cams[c] for c in cam_list}, bmap, g, sid, aid, body)
        cs = []
        for r in f[f.cam.isin(cam_list)].itertuples():
            pts = corr.ground_points({r.cam: labs[(r.cam, r.stamp_ns)]})
            if len(pts) >= 50:
                cs.append(corr.cost((r.x, r.y, r.yaw), pts))
        return float(np.median(cs)) if cs else np.inf

    grid = np.round(np.arange(a.g_min, a.g_max + 1e-9, 0.05), 3)
    costs = [mean_cost(g, a.cams) for g in grid]
    g = float(grid[int(np.argmin(costs))])
    print("ground height in base_link [m] -> median distance of edge points to borders [m]")
    for gg, cc in zip(grid, costs):
        print(f"  {gg:+.2f}: {cc:.3f}" + ("   <- best" if gg == g else ""))

    # per camera: residual at GT and bias found by the search (should be ~0)
    print("\nper camera at GT pose (dlat/dyaw = correction the search would apply, should be ~0):")
    report = {}
    for cam in a.cams:
        corr = BorderCorrector({cam: cams[cam]}, bmap, g, sid, aid, body)
        res = []
        for r in f[f.cam == cam].itertuples():
            pts = corr.ground_points({cam: labs[(r.cam, r.stamp_ns)]})
            out = corr.search((r.x, r.y, r.yaw), pts)
            if out:
                res.append((out["cost_prior"], out["dlat"], np.degrees(out["dyaw"]), out["n"]))
        if not res:
            print(f"  {cam}: no usable frames")
            continue
        r_ = np.array(res)
        report[cam] = {"residual_m": float(np.median(r_[:, 0])), "dlat_m": float(np.median(r_[:, 1])),
                       "dyaw_deg": float(np.median(r_[:, 2])), "pts": int(np.median(r_[:, 3]))}
        print(f"  {cam}: residual {report[cam]['residual_m']:.3f} m, "
              f"dlat {report[cam]['dlat_m']:+.3f} m, dyaw {report[cam]['dyaw_deg']:+.2f} deg, "
              f"{report[cam]['pts']} pts/frame, {len(res)} frames")
    out = {"ground_z": g, "masks": a.masks, "surface": a.surface, "avoid": a.avoid,
           "cams": a.cams, "per_camera": report}
    (data / "borders_calib.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {data / 'borders_calib.json'}")
    print("Cameras with |dlat| > 0.15 m or |dyaw| > 0.3 deg are biased (calibration or mask);"
          " drop them with --border-cams.")


# ----------------------------------------------------------------------------- debug
def debug(a):
    """For a few frames with GT: image with SAM road mask, the edge pixels we use (red) and
    the map borders projected with the GT pose (green); plus a bird's-eye view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from featmap import Cam, load_mask
    from lap_index import load_frames
    data = Path(a.data)
    calib = json.loads((data / "calib.json").read_text())
    bc = json.loads((data / "borders_calib.json").read_text()) if (data / "borders_calib.json").exists() else {}
    g = a.ground_z if a.ground_z is not None else bc.get("ground_z", 0.0)
    cam = Cam(calib, a.cam)
    bounds = load_bounds(data)
    bmap = BorderMap(bounds)
    sid, aid = surface_and_avoid_ids(data, a.masks, a.surface, a.avoid)
    corr = BorderCorrector({a.cam: cam}, bmap, g, sid, aid, {a.cam: load_mask(data, a.cam)})
    f = load_frames(data, laps=a.laps, cams=[a.cam]).reset_index(drop=True)
    pick = f.iloc[np.linspace(0, len(f) - 1, a.n).astype(int)]
    out = data / "borders_debug"
    out.mkdir(exist_ok=True)
    T_cb = cam.T_cb
    for r in pick.itertuples():
        img = cv2.imread(str(data / r.path))
        lab = load_labels(data, a.masks, a.cam, r.stamp_ns)
        if lab is None:
            print(f"no mask for {r.stamp_ns}")
            continue
        vis = img.copy()
        road = np.isin(lab, sid)
        vis[road] = (0.6 * vis[road] + [80, 40, 0]).astype(np.uint8)          # road: blue tint
        uv = corr.edge_pixels(a.cam, lab).astype(int)
        vis[uv[:, 1], uv[:, 0]] = (0, 0, 255)                                 # edges: red
        # map borders -> base frame (2D, on the ground plane z = g) -> image
        c, s = np.cos(r.yaw), np.sin(r.yaw)
        for b in bounds:
            dx, dy = b[:, 0] - r.x, b[:, 1] - r.y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            m = np.hypot(bx, by) < 80
            if m.sum() < 2:
                continue
            P = np.c_[bx[m], by[m], np.full(m.sum(), g)]
            Pc = P @ T_cb[:3, :3].T + T_cb[:3, 3]
            k = Pc[:, 2] > 0.5
            if k.sum() < 2:
                continue
            px, _ = cv2.projectPoints(Pc[k], np.zeros(3), np.zeros(3), cam.K, cam.D)
            px = px.reshape(-1, 2)
            ok = (np.abs(px[:, 0]) < 1e5) & (np.abs(px[:, 1]) < 1e5)
            cv2.polylines(vis, [px[ok].astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 0), 2)
        cv2.putText(vis, f"{a.cam} lap {r.lap} t={r.t_rel:.0f}s  red=SAM edges  green=map borders (GT pose, z={g:+.2f})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{a.cam}_{r.stamp_ns}_img.jpg"), vis)

        pts = corr.ground_points({a.cam: lab})
        fig, ax = plt.subplots(figsize=(7, 7))
        for b in bounds:
            dx, dy = b[:, 0] - r.x, b[:, 1] - r.y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            m = (np.abs(bx) < 60) & (np.abs(by) < 60)
            ax.plot(bx[m], by[m], "g.", ms=2)
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], "r.", ms=2)
        ax.plot(0, 0, "k^", ms=10)
        ax.set_aspect("equal"); ax.set_xlim(-10, 50); ax.set_ylim(-30, 30)
        ax.set_title(f"car frame (x fwd, y left): green = map borders, red = projected SAM edges"
                     f"\ncost at GT {corr.cost((r.x, r.y, r.yaw), pts) if len(pts) else float('nan'):.2f} m",
                     fontsize=9)
        fig.savefig(out / f"{a.cam}_{r.stamp_ns}_bev.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {a.n} image + bird's-eye pairs to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--data", required=True)
    c.add_argument("--masks", default="sam3")
    c.add_argument("--laps", nargs="+", type=int, default=[1])
    c.add_argument("--cams", nargs="+", default=["fl", "fr", "cl", "cr", "r"])
    c.add_argument("--surface", nargs="+", default=["road", "line"],
                   help="classes that count as track surface")
    c.add_argument("--avoid", nargs="+", default=["car"], help="occluders: no edges next to them")
    c.add_argument("--every", type=int, default=10, help="use every N-th frame set")
    c.add_argument("--max-frames", type=int, default=150)
    c.add_argument("--g-min", type=float, default=-0.6)
    c.add_argument("--g-max", type=float, default=0.4)
    d = sub.add_parser("debug")
    d.add_argument("--data", required=True)
    d.add_argument("--masks", default="sam3")
    d.add_argument("--laps", nargs="+", type=int, default=[1])
    d.add_argument("--cam", default="fr")
    d.add_argument("--n", type=int, default=6)
    d.add_argument("--ground-z", type=float, default=None)
    d.add_argument("--surface", nargs="+", default=["road", "line"])
    d.add_argument("--avoid", nargs="+", default=["car"])
    a = ap.parse_args()
    calibrate(a) if a.cmd == "calibrate" else debug(a)