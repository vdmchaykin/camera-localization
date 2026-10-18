#!/usr/bin/env python3
"""
First pose of a bag without GT: either given by hand, or found on the map.

    # given (e.g. by the organisers):
    python init_pose.py --data data_bag --map maps/colmap_7cam_kf2 --x 12.3 --y -45.6 --yaw 1.57
    # found automatically (global relocalization on the first frame sets):
    python init_pose.py --data data_bag --map maps/colmap_7cam_kf2

Automatic: DINOv2 of every camera of the first frame set is compared with ALL map keyframes of
the same camera; the best keyframes give candidate poses; for each candidate the normal
PnP localization runs around it, and the candidate with most inliers wins. The result is
checked on the next frame sets (it must move consistently with the twist speed).
The pose is written into the first frame set of data_bag/frames.parquet.
"""
import argparse, json, types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import featmap as F


def opts(a):
    return types.SimpleNamespace(
        dino=True, dino_model="dinov2", dino_radius=30.0, k_cand=3, max_dyaw_deg=45.0,
        min_inliers=20, pnp_px=3.0, max_kp=1024, lg_mp=False, profile=False,
        refine_max_pts=0, pnp_iters=300, pnp_prior_gate=0.0, sp_batch=False,
        cam_yaw_offsets=None)


def shape_init(a, data):
    """DIAGNOSTIC ONLY (uses the whole bag, i.e. future data, so it is not causal):
    integrate the twist of the whole bag from an arbitrary origin and align the resulting path
    shape to the driven paths of the training run (gt_tf.parquet). A full lap has a unique
    shape, so this finds where the bag starts without any image. Use it only to test the
    camera localization on a bag that comes without a first pose."""
    from scipy.spatial import cKDTree
    st = pd.read_parquet(data / "state.parquet").sort_values("stamp_ns")
    t = st.stamp_ns.values * 1e-9
    dt = np.diff(t, prepend=t[0])
    yaw = np.cumsum(st.wz_radps.values * dt)
    vx, vy = st.vx_mps.values, st.vy_mps.values
    x = np.cumsum((vx * np.cos(yaw) - vy * np.sin(yaw)) * dt)
    y = np.cumsum((vx * np.sin(yaw) + vy * np.cos(yaw)) * dt)
    P = np.c_[x, y][::20]                                     # ~5 Hz samples
    gt = pd.read_parquet(Path(a.train_data) / "gt_tf.parquet").sort_values("stamp_ns")
    gyaw = R.from_quat(gt[["qx", "qy", "qz", "qw"]].values).as_euler("zyx")[:, 0]
    ref = gt[["x", "y"]].values
    tree = cKDTree(ref)
    # candidate starts: every ~3 m along the training path, heading of the path there
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(ref, axis=0).T))]
    keep = np.unique(np.searchsorted(cum, np.arange(0, cum[-1], 3.0)).clip(0, len(ref) - 1))
    best = (np.inf, None)
    for i in keep[::2]:
        for dyaw in np.radians([-4, -2, 0, 2, 4]):
            th = gyaw[i] + dyaw
            c, s_ = np.cos(th), np.sin(th)
            Q = np.c_[ref[i, 0] + c * P[:, 0] - s_ * P[:, 1], ref[i, 1] + s_ * P[:, 0] + c * P[:, 1]]
            d, _ = tree.query(Q[::5], distance_upper_bound=50)
            cost = np.mean(np.minimum(d, 50))
            if cost < best[0]:
                best = (cost, (ref[i, 0], ref[i, 1], th))
    x0, y0, th0 = best[1]
    # local refinement
    for step_xy, step_th in ((2.0, np.radians(1)), (0.5, np.radians(0.3)), (0.1, np.radians(0.1))):
        improved = True
        while improved:
            improved = False
            for dx, dy, dth in [(step_xy, 0, 0), (-step_xy, 0, 0), (0, step_xy, 0), (0, -step_xy, 0),
                                (0, 0, step_th), (0, 0, -step_th)]:
                c, s_ = np.cos(th0 + dth), np.sin(th0 + dth)
                Q = np.c_[x0 + dx + c * P[:, 0] - s_ * P[:, 1], y0 + dy + s_ * P[:, 0] + c * P[:, 1]]
                cost = np.mean(np.minimum(tree.query(Q, distance_upper_bound=50)[0], 50))
                if cost < best[0] - 1e-4:
                    best, x0, y0, th0, improved = (cost, None), x0 + dx, y0 + dy, th0 + dth, True
    print(f"[shape, DIAGNOSTIC, non-causal] first pose ({x0:.2f}, {y0:.2f}, {np.degrees(th0):.1f} deg), "
          f"mean distance of the lap shape to training paths {best[0]:.2f} m")
    return (float(x0), float(y0), float(th0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--x", type=float); ap.add_argument("--y", type=float)
    ap.add_argument("--yaw", type=float, help="rad")
    ap.add_argument("--top", type=int, default=5, help="DINOv2 candidates per camera")
    ap.add_argument("--check-sets", type=int, default=5)
    ap.add_argument("--shape", action="store_true",
                    help="DIAGNOSTIC: first pose from the shape of the whole lap (non-causal)")
    ap.add_argument("--train-data", default="data", help="for --shape: training gt_tf.parquet")
    a = ap.parse_args()
    data, mp = Path(a.data), Path(a.map)
    f = pd.read_parquet(data / "frames.parquet")
    f["t_bin"] = (f.t_rel * 10).round().astype(int)
    sets = [s for _, s in f.groupby("t_bin")]
    loc = None

    if a.x is not None:
        pose = (a.x, a.y, a.yaw)
        print(f"given first pose: {pose}")
    elif a.shape:
        pose = shape_init(a, data)
    else:
        loc = F.Localizer(data, mp, None, o=opts(a))
        s0 = sets[0]
        imgs = {r.cam: data / r.path for r in s0.itertuples() if r.cam in loc.cams}
        D = loc.dino([cv2.imread(str(p)) for p in imgs.values()])
        cand = []
        for (cam, _), d in zip(imgs.items(), D):
            cm = loc.maps[cam]
            sim = cm.dino @ d
            for i in np.argsort(-sim)[:a.top]:
                r = cm.kt.iloc[i]
                cand.append((float(sim[i]), r.x, r.y, r.yaw, cam))
        cand.sort(key=lambda c: -c[0])
        uniq = []                                   # merge candidates closer than 15 m
        for c in cand:
            if all(np.hypot(c[1] - u[1], c[2] - u[2]) > 15 for u in uniq):
                uniq.append(c)
        print(f"{len(uniq)} candidate places from DINOv2 (best similarity {cand[0][0]:.3f})")
        best = None
        for sim, x, y, yaw, cam in uniq[:10]:
            res = loc.localize(imgs, (x, y, yaw), 30.0)
            n = res[1] if res else 0
            print(f"  candidate ({x:8.1f}, {y:8.1f}, {np.degrees(yaw):6.1f} deg) from {cam}: "
                  f"{n} inliers" + (f" -> {tuple(round(v, 2) for v in res[0])}" if res else ""))
            if res and (best is None or n > best[1]):
                best = (res[0], n)
        if best is None:
            raise SystemExit("no candidate could be localized; give the first pose by hand")
        pose = best[0]
        print(f"chosen first pose {tuple(round(v, 3) for v in pose)} with {best[1]} inliers")

    if True:
        # check: follow the twist, try a camera fix on the next frame sets
        loc = loc or F.Localizer(data, mp, None, o=opts(a))
        s0 = sets[0]
        odom = F.make_odometry(data / "state.parquet", "vx_mps", "wz_radps", 1.0, "vy_mps")
        p, t_prev = pose, int(s0.stamp_ns.max())
        for s in sets[0:a.check_sets]:
            t = int(s.stamp_ns.max())
            p = odom(p, t_prev, t)
            res = loc.localize({r.cam: data / r.path for r in s.itertuples() if r.cam in loc.cams},
                               p, 20.0)
            if res:
                d = np.hypot(res[0][0] - p[0], res[0][1] - p[1])
                print(f"  check {(t - int(s0.stamp_ns.max())) / 1e9:4.1f} s: prediction vs camera "
                      f"{d:.2f} m, {res[1]} inliers")
                p = res[0]
            else:
                print("  check: no camera fix")
            t_prev = t

    z = float(pd.read_parquet(mp / f"{F.json.loads((mp / 'meta.json').read_text())['cams'][0]}_kf.parquet").z.median())
    q = R.from_euler("z", pose[2]).as_quat()
    m = f.t_bin == sets[0].t_bin.iloc[0]
    for c, v in zip(["x", "y", "z", "qx", "qy", "qz", "qw", "yaw"], [*pose[:2], z, *q, pose[2]]):
        f.loc[m, c] = v
    f.drop(columns="t_bin").to_parquet(data / "frames.parquet")
    (data / "first_pose.json").write_text(json.dumps({"x": pose[0], "y": pose[1], "yaw": pose[2]}))
    print(f"written into {data / 'frames.parquet'} (first frame set) and {data / 'first_pose.json'}")


if __name__ == "__main__":
    main()