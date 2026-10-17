#!/usr/bin/env python3
"""
Two submissions side by side, using only what the official evaluate.py prints, per lap.

    python compare_two.py --gt gt.csv --laps-csv data/laps.csv --laps 3 6 \
        --a cam7_kf_ql0.03.csv --b predK3.csv,predK6.csv --name-a ours --name-b friends

--a / --b: one CSV with all laps, or comma-separated CSVs, one per lap in the order of --laps.
Each lap: gt.csv and the submission are cut to the lap's time window, then evaluate.py is run.
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd

# exactly the lines of evaluate.py: (label, regex, lower is better)
METRICS = [
    ("poses submitted",       r"poses submitted\s+(\d+)",                 None),
    ("poses evaluated",       r"evaluated\s+(\d+)",                       None),
    ("coverage (1 s bins) %", r"coverage \(1 s bins\)\s+([\d.]+)",        False),
    ("position RMSE [m]",     r"position RMSE\s+\[m\]\s+([\d.]+)",        True),
    ("position median [m]",   r"position median/p95\s+([\d.]+)",          True),
    ("position p95 [m]",      r"position median/p95\s+[\d.]+\s*/\s*([\d.]+)", True),
    ("lateral RMSE [m]",      r"lateral RMSE\s+\[m\]\s+([\d.]+)",         True),
    ("longitudinal RMSE [m]", r"longitudinal RMSE \[m\]\s+([\d.]+)",      True),
    ("yaw RMSE [deg]",        r"yaw RMSE\s+\[deg\]\s+([\d.]+)",           True),
    ("within 0.25 m %",       r"within 0.25 m\s+([\d.]+)",                False),
    ("within 0.50 m %",       r"within 0.50 m\s+([\d.]+)",                False),
    ("within 1.00 m %",       r"within 1.00 m\s+([\d.]+)",                False),
]


def run_evaluate(evaluate_py, gt, sub, lo, hi, tmp, tag):
    gp, sp = tmp / f"gt_{tag}.csv", tmp / f"sub_{tag}.csv"
    gt[(gt.timestamp_ns >= lo) & (gt.timestamp_ns <= hi)].to_csv(gp, index=False)
    s = sub[(sub.timestamp_ns >= lo) & (sub.timestamp_ns <= hi)]
    s[["timestamp_ns", "x", "y", "yaw"]].to_csv(sp, index=False)
    out = subprocess.run([sys.executable, evaluate_py, str(gp), str(sp)],
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    for label, rx, _ in METRICS:
        m = re.search(rx, out)
        vals[label] = float(m.group(1)) if m else float("nan")
    vals["ranked"] = "NOT RANKED" not in out
    return vals


def files_per_lap(spec, laps):
    f = spec.split(",")
    if len(f) == 1:
        return {l: f[0] for l in laps}
    assert len(f) == len(laps), f"{spec}: need one file per lap {laps}"
    return dict(zip(laps, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--laps-csv", required=True)
    ap.add_argument("--laps", nargs="+", type=int, required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--evaluate", default=str(Path(__file__).with_name("evaluate.py")))
    a = ap.parse_args()

    gt = pd.read_csv(a.gt).sort_values("timestamp_ns")
    laps = pd.read_csv(a.laps_csv)
    t0 = int(gt.timestamp_ns.iloc[0])          # laps.csv times are relative to the first GT stamp
    tmp = Path(tempfile.mkdtemp())
    fa, fb = files_per_lap(a.a, a.laps), files_per_lap(a.b, a.laps)

    for l in a.laps:
        r = laps[laps.lap == l].iloc[0]
        lo, hi = t0 + int(r.t_start * 1e9), t0 + int(r.t_end * 1e9)
        va = run_evaluate(a.evaluate, gt, pd.read_csv(fa[l]), lo, hi, tmp, f"a{l}")
        vb = run_evaluate(a.evaluate, gt, pd.read_csv(fb[l]), lo, hi, tmp, f"b{l}")
        rows = []
        for label, _, lower in METRICS:
            x, y = va[label], vb[label]
            if lower is None or x == y:
                win = "" if lower is None else "tie"
            else:
                win = a.name_a if ((x < y) if lower else (x > y)) else a.name_b
            rows.append({"metric": label, a.name_a: x, a.name_b: y, "better": win})
        t = pd.DataFrame(rows)
        print(f"\n===== lap {l} ({r.t_end - r.t_start:.1f} s) =====")
        print(t.to_string(index=False,
                          formatters={c: (lambda v: f"{v:.0f}" if v >= 100 and float(v).is_integer()
                                          else f"{v:.3f}" if v < 10 else f"{v:.1f}")
                                      for c in (a.name_a, a.name_b)}))
        for nm, v in ((a.name_a, va), (a.name_b, vb)):
            if not v["ranked"]:
                print(f"  {nm}: NOT RANKED (coverage below 90 %)")


if __name__ == "__main__":
    main()