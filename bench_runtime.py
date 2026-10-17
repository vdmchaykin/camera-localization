#!/usr/bin/env python3
"""Runtime vs accuracy of single-shot localization variants (GPU must be otherwise idle).

    python3 bench_runtime.py [--n 60]     -> results/bench_runtime.csv + table

Per query, timed with cuda sync: jpeg load + CLAHE, DINOv2 global descriptors computed on the fly
(as in a real run), retrieval, SuperPoint, LightGlue, PnP + refinement. Same queries for every variant
(dev laps 1 and 5, map without the query lap and the held-out laps).
"""
import argparse, os, time
import cv2
import numpy as np
import pandas as pd
import torch
from camera_model import CAMS, ROOT, GTPoses, frame_path, frame_stamps, yaw_of
from eval_single import allowed_mask, errors, lap_of, lap_table
from global_desc import describe, prep

VARIANTS = [
    # name, cams, k candidates, max keypoints
    ('7 камер, 3 кандидата, 1024 точки', CAMS, 3, 1024),
    ('7 камер, 1 кандидат, 1024', CAMS, 1, 1024),
    ('7 камер, 1 кандидат, 512', CAMS, 1, 512),
    ('5 камер без fl, r; 1 кандидат, 512', ['fr', 'cl', 'cr', 'rl', 'rr'], 1, 512),
    ('4 боковые, 1 кандидат, 512', ['cl', 'cr', 'rl', 'rr'], 1, 512),
]


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    a = ap.parse_args()
    from localize import Localizer
    from vislib import Features
    gt, laps, st = GTPoses(), lap_table(), frame_stamps()
    lq = lap_of(st, laps)
    rows_all = np.concatenate([np.where(lq == l)[0] for l in (1, 5)])
    rows = rows_all[np.linspace(0, len(rows_all) - 1, a.n).astype(int)]
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()
    loc = Localizer(CAMS, k=3)
    out = []
    for name, cams, k, max_kp in VARIANTS:
        fe = Features(max_kp=max_kp)
        loc.fe = fe
        loc.cams_used = list(cams)
        loc.db.cams_used = list(cams)
        T = {s: [] for s in ('load', 'dino', 'retrieve', 'superpoint', 'lightglue', 'pnp', 'total')}
        res = []
        for n, r in enumerate(rows):
            stamp = st[r]
            sync(); t0 = time.perf_counter()
            imgs = {c: cv2.imread(frame_path(c, stamp)) for c in cams if frame_path(c, stamp)}
            gray = {c: Features.load_gray(frame_path(c, stamp)) for c in imgs}
            t1 = time.perf_counter()
            x = torch.from_numpy(np.stack([prep(frame_path(c, stamp), loc.cams[c].mask) for c in imgs])).cuda()
            with torch.no_grad():
                g = describe(dino, x).cpu().numpy()
            sync(); t2 = time.perf_counter()
            allowed = allowed_mask(loc.db.kf, laps, lq[r])
            sim = np.zeros(len(loc.db.kf), np.float32)
            for gi, c in zip(g, imgs):
                sim += loc.db.G[c][loc.db.kf_row] @ gi
            sim[~allowed] = -np.inf
            order = np.argsort(-sim)
            cand = []
            for i in order:
                if len(cand) == k:
                    break
                if all(abs(i - j) >= 8 for j in cand):
                    cand.append(i)
            t3 = time.perf_counter()
            qf = {c: fe.extract(gray[c], loc.cams[c].mask) for c in imgs}
            sync(); t4 = time.perf_counter()
            corr = loc.correspondences(qf, cand)
            sync(); t5 = time.perf_counter()
            sol = loc.solve(corr)
            t6 = time.perf_counter()
            if n >= 3:   # warm-up excluded
                for key, dt in (('load', t1 - t0), ('dino', t2 - t1), ('retrieve', t3 - t2), ('superpoint', t4 - t3),
                                ('lightglue', t5 - t4), ('pnp', t6 - t5), ('total', t6 - t0)):
                    T[key].append(dt)
            if sol is not None:
                Tb = sol[0]
                pos, lat, lon, dy = errors(gt, np.array([stamp]), np.array([Tb[0, 3]]), np.array([Tb[1, 3]]),
                                           np.array([yaw_of(Tb)]))
                res.append((pos[0], lat[0], dy[0]))
            else:
                res.append((np.nan, np.nan, np.nan))
        e = np.array(res)
        ok = np.isfinite(e[:, 0])
        row = dict(variant=name, cams=len(cams), k=k, max_kp=max_kp,
                   **{f'{key}_ms': round(1000 * float(np.mean(v)), 1) for key, v in T.items()},
                   solved=round(100 * ok.mean(), 1), median_m=round(float(np.median(e[ok, 0])), 3),
                   rmse_m=round(float(np.sqrt(np.mean(e[ok, 0] ** 2))), 3),
                   lat_rmse_m=round(float(np.sqrt(np.mean(e[ok, 1] ** 2))), 3),
                   gt1m=int((e[ok, 0] > 1).sum()))
        out.append(row)
        print(f"{name:38s} total {row['total_ms']:6.0f} ms (dino {row['dino_ms']:.0f}, sp {row['superpoint_ms']:.0f}, "
              f"lg {row['lightglue_ms']:.0f}, pnp {row['pnp_ms']:.0f}, load {row['load_ms']:.0f}) | "
              f"median {row['median_m']:.3f} rmse {row['rmse_m']:.3f} lat {row['lat_rmse_m']:.3f} >1m {row['gt1m']}",
              flush=True)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, 'results', 'bench_runtime.csv'), index=False)


if __name__ == '__main__':
    main()
