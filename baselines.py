#!/usr/bin/env python3
"""Trivial baselines (step 2.8) to check the submission/evaluation pipeline.

    python3 baselines.py            # writes results/baseline_*.csv and prints evaluate.py output

Submission stamps = stamps of camera fl (10 Hz).
  gt         GT interpolated at the stamps                         -> should be ~0
  static     first GT pose forever
  centerline car moves along the track centre line at constant speed (mean GT speed — a GT-derived
             constant, this is only a sanity baseline), starting from the projection of the first pose
"""
import json, os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, 'data/c0_2026-10-13_22-53-42_camloc/tools/evaluate.py')
GT = os.path.join(ROOT, 'gt.csv')
OUT = os.path.join(ROOT, 'results')


def resample(p, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.c_[np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])]


def centerline(bounds, n=5000):
    L = np.array(bounds['track_left']['points'])[:, :2]
    R = np.array(bounds['track_right']['points'])[:, :2]
    L, R = resample(L, n), resample(R, n)
    # align right to left by nearest start point (borders may start at different places)
    k = np.argmin(np.linalg.norm(R - L[0], axis=1))
    R = np.roll(R, -k, axis=0)
    c = (L + R) / 2
    return resample(np.vstack([c, c[:1]]), n)


def write(name, t, x, y, yaw):
    p = os.path.join(OUT, f'baseline_{name}.csv')
    pd.DataFrame({'timestamp_ns': t, 'x': np.round(x, 4), 'y': np.round(y, 4),
                  'yaw': np.round(yaw, 6)}).to_csv(p, index=False)
    r = subprocess.run([sys.executable, EVAL, GT, p], capture_output=True, text=True).stdout
    with open(p.replace('.csv', '.txt'), 'w') as f:
        f.write(r)
    print(f'== {name}\n{r}')


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = pd.read_csv(GT).sort_values('timestamp_ns')
    st = pd.read_csv(os.path.join(ROOT, 'export/image_stamps.csv'))
    t = np.sort(st[st.cam == 'fl'].stamp_ns.values)
    tg = gt.timestamp_ns.values
    t = t[(t >= tg[0]) & (t <= tg[-1])]
    tf = (t - tg[0]).astype(float)
    tgf = (tg - tg[0]).astype(float)
    yu = np.unwrap(gt.yaw.values)

    write('gt', t, np.interp(tf, tgf, gt.x), np.interp(tf, tgf, gt.y), np.interp(tf, tgf, yu))

    x0, y0, yaw0 = gt.x.iloc[0], gt.y.iloc[0], gt.yaw.iloc[0]
    write('static', t, np.full(len(t), x0), np.full(len(t), y0), np.full(len(t), yaw0))

    bounds = json.load(open(os.path.join(ROOT, 'export/track_bounds.json')))
    c = centerline(bounds)
    seg = np.diff(np.vstack([c, c[:1]]), axis=0)
    s = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    length = s[-1]
    i0 = np.argmin(np.hypot(c[:, 0] - x0, c[:, 1] - y0))
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    direction = 1 if np.cos(heading[i0] - yaw0) > 0 else -1
    dist = np.sum(np.hypot(np.diff(gt.x), np.diff(gt.y)))
    v = dist / (tgf[-1] / 1e9)
    print(f'centerline length {length:.1f} m, mean GT speed {v:.2f} m/s, direction {direction}')
    sp = (s[i0] + direction * v * tf / 1e9) % length
    cc = np.vstack([c, c[:1]])
    x, y = np.interp(sp, s, cc[:, 0]), np.interp(sp, s, cc[:, 1])
    hu = np.unwrap(np.r_[heading, heading[:1]])
    yaw = np.interp(sp, s, hu) + (0 if direction > 0 else np.pi)
    write('centerline', t, x, y, yaw)

    # dead reckoning from the first pose with whitelisted motion fields (INTRO slide 10): v_mps, wz_radps.
    # Causal: integrate car-state messages with stamp <= t, output the state at each camera stamp.
    from car_state import load
    cs = load()
    cs = cs[cs.stamp_ns >= tg[0]]
    ts = cs.stamp_ns.values
    dt = np.r_[0, np.diff(ts)] / 1e9
    v, wz = cs.v_mps.values, cs.wz_radps.values
    yaw_i = yaw0 + np.cumsum(np.r_[0, wz[:-1]] * dt)
    yaw_m = yaw_i - 0.5 * np.r_[0, wz[:-1]] * dt      # midpoint heading over each step
    xi = x0 + np.cumsum(np.r_[0, v[:-1]] * dt * np.cos(yaw_m))
    yi = y0 + np.cumsum(np.r_[0, v[:-1]] * dt * np.sin(yaw_m))
    j = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)
    write('dead_reckoning', t, xi[j], yi[j], yaw_i[j])
    for sec in (10, 30, 60):
        k = np.searchsorted(t, t[0] + sec * 10**9)
        gx, gy = np.interp(tf[k], tgf, gt.x), np.interp(tf[k], tgf, gt.y)
        print(f'dead reckoning drift after {sec:3d} s: {np.hypot(xi[j[k]] - gx, yi[j[k]] - gy):.1f} m')


if __name__ == '__main__':
    main()
