#!/usr/bin/env python3
"""Dead-reckoning drift from whitelisted car state (v_mps, wz_radps) vs prediction horizon.

    python3 dr_drift.py      -> results/dr_drift.json

Starts: 400 random car-state messages on the track (outside the first 100 s and last 80 s), pose
initialised from GT, integrated forward; error vs GT at each horizon.
"""
import json, os
import numpy as np
from camera_model import ROOT, GTPoses
from car_state import load

HORIZONS = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]


def main():
    gt, cs = GTPoses(), load()
    cs = cs[(cs.stamp_ns > gt.t[0] + 100e9) & (cs.stamp_ns < gt.t[-1] - 80e9)]
    ts = cs.stamp_ns.values
    dt = np.r_[0, np.diff(ts)] / 1e9
    v, wz = cs.v_mps.values, cs.wz_radps.values
    rng = np.random.default_rng(0)
    pos = {h: [] for h in HORIZONS}; yaw = {h: [] for h in HORIZONS}
    for s in rng.integers(0, len(ts) - 3300, 400):
        x0, y0, a0 = (float(q) for q in gt.xy_yaw(ts[s]))
        e_max = s + int(max(HORIZONS) * 106) + 1
        yy = a0 + np.cumsum(wz[s:e_max] * dt[s + 1:e_max + 1])
        ym = yy - 0.5 * wz[s:e_max] * dt[s + 1:e_max + 1]
        xx = x0 + np.cumsum(v[s:e_max] * dt[s + 1:e_max + 1] * np.cos(ym))
        yy_ = y0 + np.cumsum(v[s:e_max] * dt[s + 1:e_max + 1] * np.sin(ym))
        for h in HORIZONS:
            k = np.searchsorted(ts[s + 1:e_max + 1], ts[s] + h * 1e9)
            k = min(k, len(xx) - 1)
            gx, gy, ga = gt.xy_yaw(ts[s + 1 + k])
            pos[h].append(np.hypot(xx[k] - gx, yy_[k] - gy))
            yaw[h].append(np.degrees(abs(np.angle(np.exp(1j * (yy[k] - ga))))))
    out = dict(h=HORIZONS,
               pos_med=[round(float(np.median(pos[h])), 3) for h in HORIZONS],
               pos_p90=[round(float(np.percentile(pos[h], 90)), 3) for h in HORIZONS],
               yaw_med=[round(float(np.median(yaw[h])), 3) for h in HORIZONS],
               yaw_p90=[round(float(np.percentile(yaw[h], 90)), 3) for h in HORIZONS], n=400)
    json.dump(out, open(os.path.join(ROOT, 'results', 'dr_drift.json'), 'w'))
    for i, h in enumerate(HORIZONS):
        print(f'{h:5.1f} s: pos med {out["pos_med"][i]:.2f} p90 {out["pos_p90"][i]:.2f} m | yaw med {out["yaw_med"][i]:.2f} p90 {out["yaw_p90"][i]:.2f} deg')


if __name__ == '__main__':
    main()
