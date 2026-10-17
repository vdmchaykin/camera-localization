#!/usr/bin/env python3
"""
Look at the COLMAP track map.

3D, interactive (opens in a browser; rotate / zoom with the mouse):
    python viz_map.py 3d --data data/ --map maps/colmap_4cam
    python viz_map.py 3d --data data/ --map maps/colmap_4cam --center 120 -340 --radius 150
        points coloured by their image colour (or --color height / camera),
        black line = GT path of the map keyframes, grey = track borders

One camera frame with the map projected into it:
    python viz_map.py frame --data data/ --map maps/colmap_4cam --cam cl --kf 1200
        small dots  = all map points of this camera within --max-dist m, projected with the GT pose
                      (colour = distance, no occlusion handling)
        circles     = keypoints of this keyframe that have a 3D point (what PnP can use)
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_bounds(data):
    p = Path(data) / "track_bounds.json"
    return [np.asarray(pts) for _, pts in json.loads(p.read_text())] if p.exists() else []


# ----------------------------------------------------------------------------- 3d
def view_3d(a):
    import plotly.graph_objects as go
    import pycolmap
    from scipy.spatial import cKDTree

    mp = Path(a.map)
    rec = pycolmap.Reconstruction(mp / "sfm")
    pts = [(p.xyz, p.color, p.track.length(), p.error) for p in rec.points3D.values()]
    xyz = np.array([p[0] for p in pts])
    rgb = np.array([p[1] for p in pts])
    track = np.array([p[2] for p in pts])
    err = np.array([p[3] for p in pts])
    keep = (err < a.max_err) & (track >= a.min_track)
    index = pd.read_parquet(mp / "index.parquet")
    print(f"{len(xyz)} points, {keep.sum()} after filter (error < {a.max_err}px, track >= {a.min_track})")

    if a.center:
        cx, cy = a.center
        keep &= np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy) < a.radius
        index = index[np.hypot(index.x - cx, index.y - cy) < a.radius]
    idx = np.flatnonzero(keep)
    if len(idx) > a.max_points:
        idx = np.random.default_rng(0).choice(idx, a.max_points, replace=False)
    P = xyz[idx]

    if a.color == "rgb":
        col = [f"rgb({r},{g},{b})" for r, g, b in rgb[idx]]
        marker = dict(size=a.size, color=col)
    elif a.color == "height":
        marker = dict(size=a.size, color=P[:, 2], colorscale="Viridis", showscale=True,
                      colorbar=dict(title="z [m]"))
    else:   # camera that observed the point first
        names = {iid: img.name.split("/")[0] for iid, img in rec.images.items()}
        pids = np.array(list(rec.points3D.keys()))[idx]
        cams = [names[rec.points3D[int(p)].track.elements[0].image_id] for p in pids]
        uc = sorted(set(cams))
        marker = dict(size=a.size, color=[uc.index(c) for c in cams], colorscale="Turbo",
                      showscale=True, colorbar=dict(title="camera", tickvals=list(range(len(uc))),
                                                    ticktext=uc))

    zs = a.zscale
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2] * zs, mode="markers",
                               marker=marker, name="map points"))
    # GT path of map keyframes (one camera is enough, base_link positions)
    path = index[index.cam == index.cam.iloc[0]].sort_values("stamp_ns")
    for lap, g in path.groupby("lap"):
        fig.add_trace(go.Scatter3d(x=g.x, y=g.y, z=g.z * zs, mode="lines",
                                   line=dict(color="black", width=3), name=f"keyframes lap {lap}",
                                   showlegend=bool(lap == path.lap.iloc[0])))
    # track borders, lifted to the height of the nearest keyframe
    bounds = load_bounds(a.data)
    if bounds:
        tree = cKDTree(path[["x", "y"]].values)
        zk = path.z.values
        for i, b in enumerate(bounds):
            if a.center:
                b = b[np.hypot(b[:, 0] - a.center[0], b[:, 1] - a.center[1]) < a.radius]
                if len(b) < 2:
                    continue
            z = zk[tree.query(b)[1]]
            fig.add_trace(go.Scatter3d(x=b[:, 0], y=b[:, 1], z=z * zs, mode="lines",
                                       line=dict(color="grey", width=2), name="track borders",
                                       showlegend=i == 0))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]",
                                 zaxis_title=f"z [m] x{zs:g}"),
                      title=f"{mp.name}: {len(P)} points shown", margin=dict(l=0, r=0, t=40, b=0))
    out = a.out or mp / ("map_3d.html" if not a.center else
                         f"map_3d_{a.center[0]:.0f}_{a.center[1]:.0f}.html")
    fig.write_html(out)
    print(f"saved {out}  (open in a browser)")


# ----------------------------------------------------------------------------- frame
def view_frame(a):
    from featmap import Cam, T_map_base_row
    data, mp = Path(a.data), Path(a.map)
    calib = json.loads((data / "calib.json").read_text())
    cam = Cam(calib, a.cam)
    index = pd.read_parquet(mp / "index.parquet")
    sub = index[index.cam == a.cam].reset_index(drop=True)
    r = sub.iloc[a.kf]

    frames = pd.read_parquet(data / "frames.parquet")
    fr = frames[(frames.cam == a.cam) & (frames.stamp_ns == r.stamp_ns)].iloc[0]
    T_mc = T_map_base_row(fr) @ cam.T_bc
    T_cm = np.linalg.inv(T_mc)
    img = cv2.imread(str(data / fr.path))

    z = np.load(mp / f"{a.cam}.npz")
    xyz, kf, kp = z["xyz"].astype(np.float64), z["kf"], z["kp"]
    allp = np.unique(np.round(xyz, 3), axis=0)
    Xc = allp @ T_cm[:3, :3].T + T_cm[:3, 3]
    m = (Xc[:, 2] > 1) & (np.linalg.norm(Xc, axis=1) < a.max_dist)
    uv, _ = cv2.projectPoints(Xc[m], np.zeros(3), np.zeros(3), cam.K, cam.D)
    uv = uv.reshape(-1, 2)
    d = np.linalg.norm(Xc[m], axis=1)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
    cmap = cv2.applyColorMap(np.clip(d / a.max_dist * 255, 0, 255).astype(np.uint8)[:, None],
                             cv2.COLORMAP_TURBO)[:, 0]
    vis = (img * 0.6).astype(np.uint8)
    for (u, v), c in zip(uv[inside].astype(int), cmap[inside]):
        cv2.circle(vis, (u, v), 2, tuple(int(x) for x in c), -1)
    own = kp[kf == a.kf]
    for u, v in own.astype(int):
        cv2.circle(vis, (u, v), 6, (255, 255, 255), 1)
    txt = (f"{a.cam} kf {a.kf} lap {r.lap}: {inside.sum()} map points in view (<{a.max_dist:.0f} m), "
           f"{len(own)} keyframe keypoints with 3D")
    cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = a.out or mp / f"frame_{a.cam}_{a.kf}.jpg"
    cv2.imwrite(str(out), vis)
    print(f"{txt}\nsaved {out}  (colour: blue = near, red = {a.max_dist:.0f} m)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("3d")
    v.add_argument("--data", required=True)
    v.add_argument("--map", required=True)
    v.add_argument("--center", nargs=2, type=float, default=None, help="x y of a region to zoom")
    v.add_argument("--radius", type=float, default=150.0)
    v.add_argument("--max-points", type=int, default=150_000)
    v.add_argument("--color", choices=["rgb", "height", "camera"], default="rgb")
    v.add_argument("--size", type=float, default=1.5)
    v.add_argument("--zscale", type=float, default=2.0, help="exaggerate height")
    v.add_argument("--max-err", type=float, default=2.0)
    v.add_argument("--min-track", type=int, default=3)
    v.add_argument("--out", default=None)
    f = sub.add_parser("frame")
    f.add_argument("--data", required=True)
    f.add_argument("--map", required=True)
    f.add_argument("--cam", required=True)
    f.add_argument("--kf", type=int, default=0, help="keyframe index of this camera")
    f.add_argument("--max-dist", type=float, default=80.0)
    f.add_argument("--out", default=None)
    a = ap.parse_args()
    view_3d(a) if a.cmd == "3d" else view_frame(a)