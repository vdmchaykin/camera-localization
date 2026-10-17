#!/usr/bin/env python3
"""
Plot GT trajectories of all laps over the track borders.

    python plot_laps.py --mcap ../c0_..._camloc_0.mcap --data data/ [--laps 1 2 3]

Track borders are read once from /constructor0/track_bounds and cached in
data/track_bounds.json. The message type is not hard-coded: every list of
points (objects with .x/.y) found in the first message becomes a polyline.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcap_ros2.reader import read_ros2_messages

BOUNDS_TOPIC = "/constructor0/track_bounds"


def fields(o):
    names = getattr(o, "__slots__", None) or list(getattr(o, "__dict__", {}))
    return [n for n in names if not n.startswith("_")]


def find_polylines(o, path="msg", out=None):
    out = [] if out is None else out
    if isinstance(o, (list, tuple)):
        if len(o) > 1 and all(hasattr(p, "x") and hasattr(p, "y") for p in o):
            out.append((path, [[p.x, p.y] for p in o]))
        else:
            for i, v in enumerate(o):
                find_polylines(v, f"{path}[{i}]", out)
    elif fields(o) and not isinstance(o, (str, bytes)):
        for n in fields(o):
            find_polylines(getattr(o, n), f"{path}.{n}", out)
    return out


def load_bounds(mcap, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    for m in read_ros2_messages(mcap, topics=[BOUNDS_TOPIC]):
        print(f"track_bounds type: {m.channel.schema_id} / {type(m.ros_msg).__name__}")
        lines = find_polylines(m.ros_msg)
        break
    else:
        raise RuntimeError(f"no messages on {BOUNDS_TOPIC}")
    for path, pts in lines:
        print(f"  {path}: {len(pts)} points")
    cache.write_text(json.dumps(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--laps", nargs="*", type=int, help="default: all")
    ap.add_argument("--out", default=None, help="default: <data>/laps.png")
    a = ap.parse_args()
    data = Path(a.data)

    bounds = load_bounds(a.mcap, data / "track_bounds.json")

    gt = pd.read_parquet(data / "gt_tf.parquet")          # 100 Hz GT
    laps = pd.read_csv(data / "laps.csv")
    t = (gt.stamp_ns - gt.stamp_ns.iloc[0]) * 1e-9
    i = np.searchsorted(laps.t_start.values, t.values, side="right") - 1
    gt["lap"] = np.where((i >= 0) & (t.values < laps.t_end.values[i.clip(0)]),
                         laps.lap.values[i.clip(0)], -1)
    sel = a.laps if a.laps else sorted(l for l in gt.lap.unique() if l >= 0)

    def draw_bounds(ax):
        for _, pts in bounds:
            p = np.asarray(pts)
            ax.plot(p[:, 0], p[:, 1], color="0.6", lw=0.6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    n = len(sel)
    cols = min(n, 3) + 1
    rows = int(np.ceil(n / (cols - 1)))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    gs = fig.add_gridspec(rows, cols)
    cmap = plt.get_cmap("tab10")

    # left column: all selected laps together
    ax0 = fig.add_subplot(gs[:, 0])
    draw_bounds(ax0)
    for k, lap in enumerate(sel):
        g = gt[gt.lap == lap]
        ax0.plot(g.x, g.y, lw=1, color=cmap(k % 10), label=f"lap {lap}")
    g0 = gt.iloc[0]
    ax0.plot(g0.x, g0.y, "k*", ms=12, label="first pose")
    ax0.legend(fontsize=8, loc="best"); ax0.set_title("all selected laps")

    # small multiples: one lap per panel, colored by speed
    v = np.hypot(np.gradient(gt.x, t), np.gradient(gt.y, t))
    for k, lap in enumerate(sel):
        ax = fig.add_subplot(gs[k // (cols - 1), 1 + k % (cols - 1)])
        draw_bounds(ax)
        m = (gt.lap == lap).values
        sc = ax.scatter(gt.x[m], gt.y[m], c=v[m], s=1, cmap="viridis", vmin=0, vmax=42)
        r = laps[laps.lap == lap].iloc[0]
        ax.set_title(f"lap {lap}: {r.duration_s:.0f}s, vmax {r.v_max:.0f} m/s"
                     + ("" if r.complete else " (partial)"), fontsize=9)
    fig.colorbar(sc, ax=fig.axes, shrink=0.5, label="speed [m/s]")

    out = a.out or data / "laps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()