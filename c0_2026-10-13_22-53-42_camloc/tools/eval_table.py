#!/usr/bin/env python3
"""
Official metrics (tools/evaluate.py) for several submissions, as one table.

Every submission is cut to each lap's time window (gt.csv too) and scored by the official
evaluate.py; per-lap numbers are its output, "all" pools the laps (RMSE weighted by poses).

    python eval_table.py --gt gt.csv --laps-csv data/laps.csv --laps 3 6 \
        --runs cam7_fe3.csv cam7_kf_ql0.03.csv friends=predK3.csv,predK6.csv

A plain file name is one CSV with all laps. NAME=a.csv,b.csv gives one CSV per lap, in the
order of --laps.
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = {
    "evaluated": r"evaluated\s+(\d+)",
    "coverage %": r"coverage \(1 s bins\)\s+([\d.]+)",
    "pos RMSE": r"position RMSE\s+\[m\]\s+([\d.]+)",
    "median": r"position median/p95\s+([\d.]+)",
    "p95": r"position median/p95\s+[\d.]+\s*/\s*([\d.]+)",
    "lat RMSE": r"lateral RMSE\s+\[m\]\s+([\d.]+)",
    "lon RMSE": r"longitudinal RMSE \[m\]\s+([\d.]+)",
    "yaw RMSE deg": r"yaw RMSE\s+\[deg\]\s+([\d.]+)",
    "<0.25m %": r"within 0.25 m\s+([\d.]+)",
    "<0.5m %": r"within 0.50 m\s+([\d.]+)",
    "<1m %": r"within 1.00 m\s+([\d.]+)",
}


def evaluate(evaluate_py, gt, sub, lo, hi, tmp, tag):
    g = gt[(gt.timestamp_ns >= lo) & (gt.timestamp_ns <= hi)]
    s = sub[(sub.timestamp_ns >= lo) & (sub.timestamp_ns <= hi)]
    gp, sp = tmp / f"gt_{tag}.csv", tmp / f"sub_{tag}.csv"
    g.to_csv(gp, index=False)
    s[["timestamp_ns", "x", "y", "yaw"]].to_csv(sp, index=False)
    out = subprocess.run([sys.executable, evaluate_py, str(gp), str(sp)],
                         capture_output=True, text=True, check=True).stdout
    row = {}
    for k, rx in FIELDS.items():
        m = re.search(rx, out)
        row[k] = float(m.group(1)) if m else np.nan
    row["ranked"] = "NOT RANKED" not in out
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="submission CSVs (see docstring)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--laps-csv", required=True)
    ap.add_argument("--laps", nargs="+", type=int, required=True)
    ap.add_argument("--evaluate", default=str(Path(__file__).with_name("evaluate.py")))
    ap.add_argument("--csv", default=None, help="also save the table here")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt).sort_values("timestamp_ns")
    laps = pd.read_csv(a.laps_csv)
    t0 = int(gt.timestamp_ns.iloc[0])
    win = {l: (t0 + int(laps[laps.lap == l].t_start.iloc[0] * 1e9),
               t0 + int(laps[laps.lap == l].t_end.iloc[0] * 1e9)) for l in a.laps}
    tmp = Path(tempfile.mkdtemp())
    rows = []
    for spec in a.runs:
        if "=" in spec:
            name, files = spec.split("=", 1)
            files = files.split(",")
            assert len(files) == len(a.laps), f"{name}: need one file per lap {a.laps}"
            per_lap = dict(zip(a.laps, files))
        else:
            name, per_lap = Path(spec).stem, {l: spec for l in a.laps}
        lap_rows = []
        for l in a.laps:
            sub = pd.read_csv(per_lap[l]).sort_values("timestamp_ns")
            r = evaluate(a.evaluate, gt, sub, *win[l], tmp, f"{name}_{l}")
            r.update(run=name, lap=str(l))
            lap_rows.append(r)
            rows.append(r)
        # pooled over laps
        n = np.array([r["evaluated"] for r in lap_rows])
        pooled = {"run": name, "lap": "all", "evaluated": n.sum(),
                  "ranked": all(r["ranked"] for r in lap_rows)}
        for k in ["pos RMSE", "lat RMSE", "lon RMSE", "yaw RMSE deg"]:
            pooled[k] = float(np.sqrt(np.sum(n * np.array([r[k] for r in lap_rows]) ** 2) / n.sum()))
        for k in ["coverage %", "<0.25m %", "<0.5m %", "<1m %"]:
            pooled[k] = float(np.sum(n * np.array([r[k] for r in lap_rows])) / n.sum())
        rows.append(pooled)

    t = pd.DataFrame(rows)
    cols = ["run", "lap", "evaluated", "coverage %", "pos RMSE", "median", "p95", "lat RMSE",
            "lon RMSE", "yaw RMSE deg", "<0.25m %", "<0.5m %", "<1m %", "ranked"]
    t = t[cols]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:.1f}"))
    # best run per metric on the pooled rows
    allr = t[t.lap == "all"].set_index("run")
    print("\nbest (pooled): " + ", ".join(
        f"{k}: {allr[k].idxmin()}" for k in ["pos RMSE", "lat RMSE", "lon RMSE", "yaw RMSE deg"]))
    if a.csv:
        t.to_csv(a.csv, index=False)
        print(f"saved {a.csv}")


if __name__ == "__main__":
    main()