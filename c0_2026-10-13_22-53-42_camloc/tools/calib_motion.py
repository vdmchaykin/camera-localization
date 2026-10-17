#!/usr/bin/env python3
"""
Calibrate the car-state motion model against GT on TRAINING laps only (default: not 3 and 6).

The car state measures velocity at its own reference point, not at base_link. In a corner the
lateral velocity at base_link differs by wz * lever arm, so plain integration of vy drifts
outward. Fitted model (same form as a lever-arm correction plus scale factors):

    vx_eff = vx_k * vx
    vy_eff = vy_a * vy + vy_b * wz + vy_c
    wz_eff = wz_s * wz + wz_d

    python calib_motion.py --data data/ --exclude-laps 3 6 --out data/motion_calib.json

GT is used offline here (calibration, like building the map); at inference the json holds
six constants and only the allowed vx_mps / vy_mps / wz_radps are used.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--exclude-laps", nargs="+", type=int, default=[3, 6])
    ap.add_argument("--vx-col", default="vx_mps")
    ap.add_argument("--vy-col", default="vy_mps")
    ap.add_argument("--wz-col", default="wz_radps")
    ap.add_argument("--min-speed", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data)

    gt = pd.read_parquet(data / "gt_tf.parquet").sort_values("stamp_ns").reset_index(drop=True)
    st = pd.read_parquet(data / "state.parquet").sort_values("stamp_ns")
    laps = pd.read_csv(data / "laps.csv")

    t = gt.stamp_ns.values
    ts = (t - t[0]) * 1e-9
    yaw = np.unwrap(R.from_quat(gt[["qx", "qy", "qz", "qw"]].values).as_euler("zyx")[:, 0])
    k = 5                                         # +-5 samples (~0.1 s) central differences
    sl, sr = slice(0, -2 * k), slice(2 * k, None)
    dt = ts[sr] - ts[sl]
    tm = t[k:-k]
    vxm = (gt.x.values[sr] - gt.x.values[sl]) / dt
    vym = (gt.y.values[sr] - gt.y.values[sl]) / dt
    ym = yaw[k:-k]
    vx_gt = np.cos(ym) * vxm + np.sin(ym) * vym           # base_link velocity in body frame
    vy_gt = -np.sin(ym) * vxm + np.cos(ym) * vym
    wz_gt = (yaw[sr] - yaw[sl]) / dt

    vx = np.interp(tm, st.stamp_ns, st[a.vx_col])
    vy = np.interp(tm, st.stamp_ns, st[a.vy_col])
    wz = np.interp(tm, st.stamp_ns, st[a.wz_col])

    rel = (tm - t[0]) * 1e-9
    i = np.searchsorted(laps.t_start.values, rel, side="right") - 1
    lap = np.where((i >= 0) & (rel < laps.t_end.values[i.clip(0)]), laps.lap.values[i.clip(0)], -1)
    m = (lap >= 0) & ~np.isin(lap, a.exclude_laps) & (vx > a.min_speed)
    print(f"fitting on {m.sum()} samples, laps {sorted(set(lap[m]))}")

    def lsq(A, y):
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return c, np.sqrt(np.mean((A @ c - y) ** 2))

    (vx_k,), r_vx = lsq(vx[m, None], vx_gt[m])
    (vy_a, vy_b, vy_c), r_vy = lsq(np.c_[vy[m], wz[m], np.ones(m.sum())], vy_gt[m])
    (wz_s, wz_d), r_wz = lsq(np.c_[wz[m], np.ones(m.sum())], wz_gt[m])
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    print(f"vx_eff = {vx_k:.4f} * vx                          "
          f"residual {rms(vx[m] - vx_gt[m]):.3f} -> {r_vx:.3f} m/s")
    print(f"vy_eff = {vy_a:.3f} * vy + {vy_b:+.3f} * wz {vy_c:+.3f}   "
          f"residual {rms(vy[m] - vy_gt[m]):.3f} -> {r_vy:.3f} m/s")
    print(f"wz_eff = {wz_s:.4f} * wz {wz_d:+.5f}               "
          f"residual {rms(wz[m] - wz_gt[m]):.4f} -> {r_wz:.4f} rad/s")
    print(f"(vy_b ~ lever arm in m between the car-state point and base_link)")

    cal = dict(vx_k=vx_k, vy_a=vy_a, vy_b=vy_b, vy_c=vy_c, wz_s=wz_s, wz_d=wz_d,
               cols=[a.vx_col, a.vy_col, a.wz_col], fitted_on_laps=sorted(int(l) for l in set(lap[m])))
    if a.out:
        Path(a.out).write_text(json.dumps({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                           for k, v in cal.items()}, indent=2))
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()