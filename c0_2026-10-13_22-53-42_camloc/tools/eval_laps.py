#!/usr/bin/env python3
"""
Run the official evaluate.py on single laps.

evaluate.py measures coverage over the whole GT window, so scoring a CSV that only contains
laps 3 and 6 against the full-run gt.csv reports ~20 % coverage and "NOT RANKED". This script
cuts gt.csv (and the submission) to each lap's time window and runs evaluate.py per lap.

    python eval_laps.py --gt gt.csv --laps-csv data/laps.csv --sub mask.csv --laps 3 6
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="gt.csv from extract_gt.py")
    ap.add_argument("--laps-csv", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--laps", nargs="+", type=int, required=True)
    ap.add_argument("--evaluate", default=str(Path(__file__).with_name("evaluate.py")))
    a = ap.parse_args()

    gt = pd.read_csv(a.gt).sort_values("timestamp_ns")
    sub = pd.read_csv(a.sub).sort_values("timestamp_ns")
    laps = pd.read_csv(a.laps_csv)
    t0 = int(gt.timestamp_ns.iloc[0])            # laps.csv times are relative to the first GT stamp
    tmp = Path(tempfile.mkdtemp())
    for lap in a.laps:
        r = laps[laps.lap == lap].iloc[0]
        lo, hi = t0 + int(r.t_start * 1e9), t0 + int(r.t_end * 1e9)
        g = gt[(gt.timestamp_ns >= lo) & (gt.timestamp_ns <= hi)]
        s = sub[(sub.timestamp_ns >= lo) & (sub.timestamp_ns <= hi)]
        gp, sp = tmp / f"gt_{lap}.csv", tmp / f"sub_{lap}.csv"
        g.to_csv(gp, index=False)
        s[["timestamp_ns", "x", "y", "yaw"]].to_csv(sp, index=False)
        print(f"\n===== lap {lap}: {len(s)} poses, {r.duration_s:.1f} s =====", flush=True)
        subprocess.run([sys.executable, a.evaluate, str(gp), str(sp)], check=True)


if __name__ == "__main__":
    main()