#!/usr/bin/env python3
"""
Several trajectories on one track map.

    python plot_traj.py --bounds data/track_bounds.json \
        --runs bag_run.csv:day_bag mix_v7.csv:night_K3_K6 \
        --gt data/gt_tf.parquet --gt-laps 3 6 --laps-csv data/laps.csv \
        --out traj.png
    # zoom on one place:  --center 120 -340 --radius 80

Each run is a CSV with timestamp_ns,x,y,yaw (the localizer output); "file:label" sets the
legend name. --gt adds the ground truth of the given training laps (black).
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="file.csv[:label]")
    ap.add_argument("--bounds", default="data/track_bounds.json")
    ap.add_argument("--gt", default=None, help="data/gt_tf.parquet")
    ap.add_argument("--gt-laps", nargs="*", type=int, default=[])
    ap.add_argument("--laps-csv", default="data/laps.csv")
    ap.add_argument("--center", nargs=2, type=float, default=None)
    ap.add_argument("--radius", type=float, default=80)
    ap.add_argument("--out", default="traj.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13, 11))
    if Path(a.bounds).exists():
        for _, b in json.loads(Path(a.bounds).read_text()):
            b = np.asarray(b)
            ax.plot(b[:, 0], b[:, 1], color="0.8", lw=0.8)

    if a.gt and a.gt_laps:
        gt = pd.read_parquet(a.gt).sort_values("stamp_ns")
        laps = pd.read_csv(a.laps_csv)
        t0 = gt.stamp_ns.iloc[0]
        for i, l in enumerate(a.gt_laps):
            r = laps[laps.lap == l].iloc[0]
            m = (gt.stamp_ns >= t0 + r.t_start * 1e9) & (gt.stamp_ns <= t0 + r.t_end * 1e9)
            ax.plot(gt.x[m], gt.y[m], color="k", lw=1.2, label="ground truth" if i == 0 else None)

    colors = plt.get_cmap("tab10")
    for k, spec in enumerate(a.runs):
        f, _, label = spec.partition(":")
        r = pd.read_csv(f).sort_values("timestamp_ns")
        # break the line where consecutive poses are far apart in time (different laps)
        gap = np.r_[False, np.diff(r.timestamp_ns.values) > 1e9]
        seg = np.cumsum(gap)
        for j, (_, g) in enumerate(r.groupby(seg)):
            ax.plot(g.x, g.y, lw=1.0, color=colors(k % 10), label=(label or Path(f).stem) if j == 0 else None)
        ax.plot(r.x.iloc[0], r.y.iloc[0], "o", color=colors(k % 10), ms=7)
        print(f"{label or f}: {len(r)} poses, {(r.timestamp_ns.max() - r.timestamp_ns.min()) / 1e9:.1f} s")

    if a.center:
        ax.set_xlim(a.center[0] - a.radius, a.center[0] + a.radius)
        ax.set_ylim(a.center[1] - a.radius, a.center[1] + a.radius)
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(loc="best")
    ax.set_title("trajectories (circle = start)")
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()