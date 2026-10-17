#!/usr/bin/env python3
"""Stage timing of one v2 guided correction (GPU idle).  python3 v2/profile_v2.py [--n 30] [--start-s 200]
Prior = GT pose + 0.2 m / 0.1 deg noise (only to drive the guided path in this benchmark)."""
import argparse, os, sys, time
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'sim')); sys.path.insert(0, ROOT)
os.environ['LOC_SYNC'] = '1'
import my_loc_v2 as L
from profile_reloc import sets, MCAP
from simulate import read_static, GT
from mcap.reader import make_reader

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=30); ap.add_argument('--start-s', type=float, default=680)   # K4 (development lap); never K3/K6 for tuning
a = ap.parse_args()
with open(MCAP, 'rb') as f:
    bag0 = make_reader(f).get_summary().statistics.message_start_time
t0 = bag0 + int(a.start_s * 1e9)
raw = sets(t0, a.n); info, tfs, bounds = read_static(MCAP, t0)
gt = GT(os.path.join(ROOT, 'gt.csv')); rng = np.random.default_rng(0)
first = next(iter(raw[0].values()))[0]
loc = L.Localizer(dict(stamp_ns=first, x=0., y=0., yaw=0.), info, tfs, bounds)
T = {k: [] for k in ('resize', 'superpoint', 'match+solve', 'solve', 'total')}
_ls = L.least_squares
def _timed(*a, **k):
    t = time.perf_counter(); r = _ls(*a, **k); T_ls[0] += time.perf_counter() - t; return r
L.least_squares = _timed
T_ls = [0.0]
sync = torch.cuda.synchronize
for k, s in enumerate(raw):
    stamp = next(iter(s.values()))[0]
    imgs = {c: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for c, (_, b) in s.items()}
    gx, gy, gyaw, _ = gt.at([stamp])
    prior = dict(x=gx[0] + rng.normal(0, .2), y=gy[0] + rng.normal(0, .2), yaw=gyaw[0] + np.radians(rng.normal(0, .1)), sig_xy=0.25, sig_yaw=np.radians(0.2))
    sync(); ta = time.perf_counter()
    half = {c: cv2.resize(im, (L.W, L.H), interpolation=cv2.INTER_AREA) for c, im in imgs.items()}
    tb = time.perf_counter()
    qf = loc._features(half); sync(); tc = time.perf_counter()
    T_ls[0] = 0.0
    r = loc._guided(stamp, qf, prior); td = time.perf_counter()
    if k >= 3:
        for key, v in (('resize', tb - ta), ('superpoint', tc - tb), ('match+solve', td - tc), ('solve', T_ls[0]), ('total', td - ta)):
            T[key].append(v * 1e3)
    if r and k % 5 == 0:
        print(f'err {np.hypot(r[0]-gx[0], r[1]-gy[0]):.3f} m, inliers {r[3]}')
for key, v in T.items():
    print(f'  {key:12s} mean {np.mean(v):7.1f} ms  p95 {np.percentile(v, 95):7.1f} ms')
