#!/usr/bin/env python3
"""Stage timing of SuperPoint as v2 runs it (7 cameras, one batch), lap K4 (development).
python3 v3/sp_profile.py [--n 30] [--start-s 680]"""
import argparse, os, sys, time
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'sim'))
from profile_reloc import sets
from camera_model import CAMS, H, W
from vislib import Features, _CLAHE
from lightglue.superpoint import simple_nms, top_k_keypoints, sample_descriptors

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=30); ap.add_argument('--start-s', type=float, default=680)
a = ap.parse_args()
from mcap.reader import make_reader
from profile_reloc import MCAP
with open(MCAP, 'rb') as f:
    bag0 = make_reader(f).get_summary().statistics.message_start_time
raw = sets(bag0 + int(a.start_s * 1e9), a.n)
half = [{c: cv2.resize(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_AREA)
         for c, (_, b) in s.items()} for s in raw]
sp = Features(max_kp=512).sp
sync = torch.cuda.synchronize
T = {}
def tick(k, t):
    sync(); now = time.perf_counter(); T.setdefault(k, []).append((now - t) * 1e3); return now

@torch.no_grad()
def run(h, rec):
    t = time.perf_counter()
    g = np.stack([_CLAHE.apply(cv2.cvtColor(h[c], cv2.COLOR_BGR2GRAY)) for c in CAMS])
    t = tick('clahe_cpu', t) if rec else t
    x = torch.from_numpy(g).float().div(255)[:, None].cuda()
    t = tick('upload', t) if rec else t
    s = sp
    y = s.relu(s.conv1a(x)); y = s.relu(s.conv1b(y)); y = s.pool(y)
    y = s.relu(s.conv2a(y)); y = s.relu(s.conv2b(y)); y = s.pool(y)
    y = s.relu(s.conv3a(y)); y = s.relu(s.conv3b(y)); y = s.pool(y)
    y = s.relu(s.conv4a(y)); y = s.relu(s.conv4b(y))
    sc = torch.nn.functional.softmax(s.convPb(s.relu(s.convPa(y))), 1)[:, :-1]
    b, _, hh, ww = sc.shape
    sc = sc.permute(0, 2, 3, 1).reshape(b, hh, ww, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, hh * 8, ww * 8)
    dsc = torch.nn.functional.normalize(s.convDb(s.relu(s.convDa(y))), p=2, dim=1)
    t = tick('backbone+heads', t) if rec else t
    sc = simple_nms(sc, s.conf.nms_radius)
    t = tick('nms', t) if rec else t
    p = s.conf.remove_borders
    sc[:, :p] = -1; sc[:, :, :p] = -1; sc[:, -p:] = -1; sc[:, :, -p:] = -1
    best = torch.where(sc > s.conf.detection_threshold)
    v = sc[best]
    kps = [torch.stack(best[1:3], -1)[best[0] == i] for i in range(b)]
    scs = [v[best[0] == i] for i in range(b)]
    kps, scs = zip(*[top_k_keypoints(k, q, 512) for k, q in zip(kps, scs)])
    t = tick('select+topk', t) if rec else t
    kps = [torch.flip(k, [1]).float() for k in kps]
    d = [sample_descriptors(k[None], dd[None], 8)[0] for k, dd in zip(kps, dsc)]
    t = tick('sample_desc', t) if rec else t
    _ = [k.cpu().numpy() for k in kps] + [q.cpu().numpy() for q in d]
    tick('download', t) if rec else t

for i, h in enumerate(half):
    run(h, i >= 3)
tot = sum(np.mean(v) for v in T.values())
for k, v in T.items():
    print(f'  {k:16s} mean {np.mean(v):6.1f} ms  p95 {np.percentile(v, 95):6.1f}')
print(f'  {"sum":16s} mean {tot:6.1f} ms')
