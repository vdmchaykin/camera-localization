#!/usr/bin/env python3
"""Leave-lap-out evaluation of single-shot localization (no filter).

    python3 eval_single.py retrieval                       # retrieval accuracy, development laps
    python3 eval_single.py pose --laps 1 5 --every 10 [--cams fl fr ...]
    python3 eval_single.py pose --val                      # held-out laps: only for final checks

Laps start at the official start line (analysis/laps.csv). VAL_LAPS are held out (INTRO slide 20): their
keyframes are never in the map and they are never used for tuning. For a query in lap L the map may only
use keyframes outside lap L and outside VAL_LAPS (± MARGIN_S).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from camera_model import CAMS, ROOT, GTPoses, frame_stamps, yaw_of

MARGIN_S = 5.0
VAL_LAPS = [3, 6]          # held out: fastest laps (110.6 s, 111.5 s; v max 40.3 m/s, one stop)
DEV_LAPS = [1, 2, 4, 5, 7]


def lap_table():
    laps = pd.read_csv(os.path.join(ROOT, 'analysis/laps.csv'))
    t0 = pd.read_csv(os.path.join(ROOT, 'export/gt_tf.csv'), usecols=['timestamp_ns']).timestamp_ns.min()
    laps['a'] = t0 + (laps.t_start * 1e9).astype(np.int64)
    laps['b'] = t0 + (laps.t_end * 1e9).astype(np.int64)
    return laps


def lap_of(stamps, laps):
    return np.clip(np.searchsorted(laps.a.values, stamps, side='right') - 1, 0, len(laps) - 1)


def allowed_mask(kf, laps, lap):
    ok = np.ones(len(kf), bool)
    for l in set([lap] + VAL_LAPS):
        a, b = laps.a[l] - MARGIN_S * 1e9, laps.b[l] + MARGIN_S * 1e9
        ok &= (kf < a) | (kf > b)
    return ok


def errors(gt, stamps, x, y, yaw):
    gx, gy, gyaw = gt.xy_yaw(stamps)
    dx, dy = x - gx, y - gy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    dyaw = np.degrees(np.angle(np.exp(1j * (yaw - gyaw))))
    return np.hypot(dx, dy), lat, lon, dyaw


def run_retrieval(args):
    from localize import MapDB
    db = MapDB(args.cams)
    gt, laps = GTPoses(), lap_table()
    st = frame_stamps()
    kx, ky, _ = gt.xy_yaw(db.kf)
    qx, qy, _ = gt.xy_yaw(st)
    lq = lap_of(st, laps)
    d1, dk = [], []
    for lap in args.laps:
        allowed = allowed_mask(db.kf, laps, lap)
        for r in np.where(lq == lap)[0][::args.every]:
            ids, _ = db.retrieve(r, allowed, 3)
            dd = np.hypot(kx[ids] - qx[r], ky[ids] - qy[r])
            d1.append(dd[0]); dk.append(dd.min())
    d1, dk = np.array(d1), np.array(dk)
    for name, d in (('top-1', d1), ('best of top-3', dk)):
        print(f'{name}: n={len(d)} median {np.median(d):.1f} m, <10 m {100 * np.mean(d < 10):.1f} %, '
              f'<25 m {100 * np.mean(d < 25):.1f} %, >100 m {100 * np.mean(d > 100):.1f} %')


def run_pose(args):
    from localize import Localizer
    loc = Localizer(args.cams, k=args.k, max_kp=args.max_kp)
    gt, laps = GTPoses(), lap_table()
    st = frame_stamps()
    lq = lap_of(st, laps)
    rows = []
    t0 = time.time()
    for lap in args.laps:
        allowed = allowed_mask(loc.db.kf, laps, lap)
        for r in np.where(lq == lap)[0][::args.every]:
            out, cand = loc.localize(st[r], allowed)
            row = dict(stamp=st[r], lap=lap, ok=out is not None)
            if out is not None:
                T, n_inl, per = out
                row.update(x=T[0, 3], y=T[1, 3], yaw=yaw_of(T), inliers=n_inl,
                           **{f'inl_{c}': per.get(c, 0) for c in args.cams})
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f'{len(rows)} queries, {(time.time() - t0) / len(rows):.2f} s/query', flush=True)
    df = pd.DataFrame(rows)
    ok = df.ok.values
    pos, lat, lon, dyaw = errors(gt, df.stamp.values[ok], df.x.values[ok], df.y.values[ok], df.yaw.values[ok])
    df.loc[ok, 'err'] = pos; df.loc[ok, 'lat'] = lat; df.loc[ok, 'lon'] = lon; df.loc[ok, 'dyaw'] = dyaw
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out = os.path.join(ROOT, 'results', f'single_{args.tag}.csv')
    df.to_csv(out, index=False)
    print(f'saved {out}; {(time.time() - t0) / len(df):.2f} s/query')
    summarize(df)


def summarize(df):
    ok = df.ok.values
    print(f'solved {100 * ok.mean():.1f} % of {len(df)}')
    for thr in (0, 30, 60, 100):
        s = df[ok & (df.inliers.fillna(0) >= thr)]
        if not len(s):
            continue
        e = s.err.values
        print(f' inliers>={thr:3d}: {100 * len(s) / len(df):5.1f} % | median {np.median(e):.2f} m  '
              f'RMSE {np.sqrt(np.mean(e ** 2)):.2f}  lat RMSE(trim p95) '
              f'{np.sqrt(np.mean(np.sort(s.lat.values ** 2)[:max(1, int(.95 * len(s)))])):.2f}  '
              f'<0.5 m {100 * np.mean(e < .5):.0f} %  >5 m {100 * np.mean(e > 5):.1f} %  '
              f'yaw med {np.median(np.abs(s.dyaw)):.2f}°')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['retrieval', 'pose'])
    ap.add_argument('--laps', type=int, nargs='+', default=DEV_LAPS)
    ap.add_argument('--val', action='store_true', help='evaluate on the held-out VAL_LAPS')
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--cams', nargs='+', default=CAMS)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--max-kp', type=int, default=1024)
    ap.add_argument('--tag', default='dev')
    a = ap.parse_args()
    if a.val:
        a.laps, a.tag = VAL_LAPS, 'val_' + a.tag
    run_retrieval(a) if a.mode == 'retrieval' else run_pose(a)
