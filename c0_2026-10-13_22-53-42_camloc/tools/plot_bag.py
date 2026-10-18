#!/usr/bin/env python3
"""
Bag without GT: trajectory of the localizer on the track map, vs. pure odometry (twist).

    python plot_bag.py --data data_bag --run bag_run.csv [--diag bag_run.diag.csv]

Saves <run>.png: track borders, the estimate (green where a camera fix was applied, red where
not), dead reckoning from the same first pose (grey), and the distance between the two over
time. Without GT this is a plausibility check: the estimate must stay on the track, follow
its shape, and the camera fixes must be accepted most of the time.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import featmap as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--diag", default=None)
    ap.add_argument("--odom-scale", type=float, default=1.0,
                    help="speed scale for the odometry-only line (e.g. the k the filter learned)")
    a = ap.parse_args()
    data = Path(a.data)
    r = pd.read_csv(a.run).sort_values("timestamp_ns").reset_index(drop=True)
    d = pd.read_csv(a.diag) if a.diag else (pd.read_csv(Path(a.run).with_suffix(".diag.csv"))
                                            if Path(a.run).with_suffix(".diag.csv").exists() else None)
    p0 = json.loads((data / "first_pose.json").read_text())

    odom = F.make_odometry(data / "state.parquet", "vx_mps", "wz_radps", 1.0, "vy_mps")
    dr, p, tp = [], (p0["x"], p0["y"], p0["yaw"]), int(r.timestamp_ns.iloc[0])
    for t in r.timestamp_ns.values:
        p = odom(p, tp, int(t), 0.0, a.odom_scale); tp = int(t); dr.append(p)
    dr = np.array(dr)
    gap = np.hypot(r.x - dr[:, 0], r.y - dr[:, 1])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.2, 1]})
    bounds = []
    tb = data / "track_bounds.json"
    if tb.exists():
        bounds = [np.asarray(b) for _, b in json.loads(tb.read_text())]
        for b in bounds:
            ax.plot(b[:, 0], b[:, 1], color="0.75", lw=0.7)
    ax.plot(dr[:, 0], dr[:, 1], color="0.5", lw=1, ls="--", label=f"odometry only (twist x{a.odom_scale:g})")
    ok = d.ok.values.astype(bool) if d is not None and len(d) == len(r) else np.ones(len(r), bool)
    ax.scatter(r.x[ok], r.y[ok], s=3, color="tab:green", label="estimate, camera fix")
    ax.scatter(r.x[~ok], r.y[~ok], s=6, color="tab:red", label="estimate, no fix")
    ax.plot(r.x.iloc[0], r.y.iloc[0], "k*", ms=14, label="first pose")
    ax.set_aspect("equal"); ax.legend(fontsize=9)
    ax.set_title(f"{Path(a.run).name}: {len(r)} poses, fixes {ok.mean():.0%}")

    t = (r.timestamp_ns - r.timestamp_ns.iloc[0]) * 1e-9
    ax2.plot(t, gap, label="estimate vs odometry only [m]")
    if bounds:                                   # distance to the nearest border line
        pts = np.concatenate([np.c_[np.interp(np.linspace(0, len(b) - 1, len(b) * 10),
                                              np.arange(len(b)), b[:, 0]),
                                    np.interp(np.linspace(0, len(b) - 1, len(b) * 10),
                                              np.arange(len(b)), b[:, 1])] for b in bounds])
        dist, _ = cKDTree(pts).query(np.c_[r.x, r.y])
        ax2.plot(t, dist, label="distance to nearest track border [m]")
        print(f"distance to nearest border: min {dist.min():.2f} m, "
              f"{(dist < 0.5).mean():.1%} of poses closer than 0.5 m to a border line")
    ax2.set_xlabel("time [s]"); ax2.legend(); ax2.grid(alpha=0.3)
    out = Path(a.run).with_suffix(".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"estimate vs odometry-only: median {np.median(gap):.2f} m, max {gap.max():.2f} m")
    if d is not None and "n_inl" in d:
        fx = d[d.ok.astype(bool)]
        print(f"camera fixes accepted {d.ok.mean():.0%}, median inliers {fx.n_inl.median():.0f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()