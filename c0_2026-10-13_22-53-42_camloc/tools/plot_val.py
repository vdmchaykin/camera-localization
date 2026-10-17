#!/usr/bin/env python3
"""
Estimated trajectory vs ground truth for validation laps.

    python plot_val.py --diag val_fr_3_6.diag.csv --data data/ [--out val_fr_3_6.png]

Left: track map with GT (black), map fixes (green) and frames without a fix (red),
the first lost frame is marked. The view is limited to the track, so estimates that
fly off are clipped at the border. Right: position / lateral error over time.
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
    ap.add_argument("--diag", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lost-m", type=float, default=5.0, help="error that counts as 'lost'")
    ap.add_argument("--exaggerate", type=float, default=1.0,
                    help="draw GT + k*(estimate - GT), e.g. 20 to see 10 cm errors")
    ap.add_argument("--err-max", type=float, default=1.0, help="colour scale limit, m")
    ap.add_argument("--bounds", action="store_true", help="also draw track borders")
    a = ap.parse_args()
    data = Path(a.data)

    d = pd.read_csv(a.diag)
    gt = pd.read_parquet(data / "gt_tf.parquet")
    d["gx"] = np.interp(d.timestamp_ns, gt.stamp_ns, gt.x)
    d["gy"] = np.interp(d.timestamp_ns, gt.stamp_ns, gt.y)
    bounds = []
    tb = data / "track_bounds.json"
    if tb.exists():
        bounds = [np.asarray(p) for _, p in json.loads(tb.read_text())]
    else:
        print("no track_bounds.json (run plot_laps.py once) - plotting without borders")

    laps = sorted(d.lap.unique())
    fig, axes = plt.subplots(len(laps), 2, figsize=(18, 8 * len(laps)),
                             gridspec_kw={"width_ratios": [1.1, 1]}, squeeze=False)
    k = a.exaggerate
    for (axm, axe), lap in zip(axes, laps):
        x = d[d.lap == lap].reset_index(drop=True)
        t = (x.timestamp_ns - x.timestamp_ns.iloc[0]) * 1e-9
        # deviation from GT, optionally exaggerated so 10 cm errors become visible
        ex = x.gx + k * (x.x - x.gx)
        ey = x.gy + k * (x.y - x.gy)

        if a.bounds:
            for b in bounds:
                axm.plot(b[:, 0], b[:, 1], color="0.85", lw=0.6)
        axm.plot(x.gx, x.gy, color="k", lw=1.2, label="ground truth (this lap)")
        sc = axm.scatter(ex[x.ok], ey[x.ok], c=x.err[x.ok], s=6, cmap="RdYlGn_r",
                         vmin=0, vmax=a.err_max, zorder=3)
        axm.scatter(ex[~x.ok], ey[~x.ok], c=x.err[~x.ok], s=14, marker="x", cmap="RdYlGn_r",
                    vmin=0, vmax=a.err_max, zorder=3)
        axm.plot([], [], "o", ms=5, color="tab:green", label="estimate, map fix")
        axm.plot([], [], "x", ms=6, color="tab:red", label="estimate, no fix")
        fig.colorbar(sc, ax=axm, shrink=0.6, label=f"position error [m] (clipped at {a.err_max})")
        axm.plot(x.gx.iloc[0], x.gy.iloc[0], "k*", ms=14, label="start")
        lost = np.flatnonzero(x.err.values > a.lost_m)
        if len(lost):
            i = lost[0]
            axm.plot(x.gx[i], x.gy[i], "o", ms=14, mfc="none", mec="tab:red", mew=2.5,
                     label=f"lost at frame {i} (t={t[i]:.0f}s)")
            axe.axvline(t[i], color="tab:red", ls="--", lw=1)
        pad = 20
        axm.set_xlim(x.gx.min() - pad, x.gx.max() + pad)
        axm.set_ylim(x.gy.min() - pad, x.gy.max() + pad)
        axm.set_aspect("equal")
        rms = np.sqrt(np.mean(x.err ** 2))
        axm.set_title(f"lap {lap}: fix rate {x.ok.mean():.0%}, <1 m {(x.err < 1).mean():.0%}, "
                      f"RMSE {rms:.2f} m" + (f"   (deviation x{k:g})" if k != 1 else ""))
        axm.legend(loc="best", fontsize=9)

        axe.semilogy(t, np.maximum(x.err, 1e-3), color="tab:blue", lw=1, label="position error")
        axe.semilogy(t, np.maximum(np.abs(x.err_lat), 1e-3), color="tab:orange", lw=0.8,
                     alpha=0.8, label="|lateral error|")
        axe.axhline(1.0, color="k", ls=":", lw=1, label="1 m")
        y0, y1 = axe.get_ylim()
        axe.fill_between(t, y0, y1, where=~x.ok.values, color="tab:red", alpha=0.08,
                         step="mid", label="no fix")
        axe.set_ylim(y0, y1)
        axe.set_xlabel("time in lap [s]"); axe.set_ylabel("error [m]")
        axe.set_title(f"lap {lap}: error over time")
        axe.legend(loc="upper left", fontsize=9)
        axe.grid(True, which="both", alpha=0.3)

    out = a.out or Path(a.diag).with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()