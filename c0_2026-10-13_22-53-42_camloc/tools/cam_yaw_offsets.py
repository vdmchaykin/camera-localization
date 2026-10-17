#!/usr/bin/env python3
"""
Per-camera yaw bias from a validation diag file, and the offsets json for featmap.py.

Each camera's own PnP gives a base_link yaw; its median difference to GT over a lap is that
camera's systematic yaw error (extrinsic calibration / map inconsistency). Estimate it on a lap
that is NOT used for validation, then pass the json with --cam-yaw-offsets.

    python featmap.py validate --data data/ --map maps/colmap_4cam --laps 4 --out cal_lap4.csv \\
        --no-query-masks --state data/state.parquet --speed-col vx_mps --vy-col vy_mps --yawrate-col wz_radps
    python cam_yaw_offsets.py cal_lap4.diag.csv --out data/cam_yaw_offsets.json
"""
import argparse, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("diag")
ap.add_argument("--out", default=None)
a = ap.parse_args()
d = pd.read_csv(a.diag)
offs = {}
print("camera   median yaw error   spread (IQR)   fixes")
for c in sorted(col for col in d.columns if col.startswith("yawerr_")):
    v = d[c].dropna()
    cam = c.replace("yawerr_", "")
    offs[cam] = float(v.median())
    print(f"  {cam:4s}   {v.median():+.4f} deg      {v.quantile(0.75) - v.quantile(0.25):.4f} deg   {len(v)}")
if "err_yaw" in d:
    print(f"fused pose: median yaw error {d.err_yaw.median():+.4f} deg")
if a.out:
    json.dump(offs, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")