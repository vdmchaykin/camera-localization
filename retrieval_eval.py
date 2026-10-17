#!/usr/bin/env python3
"""DINOv2 retrieval accuracy, leave-lap-out on the development laps (held-out laps never in the map).

    python3 retrieval_eval.py      -> results/retrieval_dev.json (per-query errors for each camera set)

Map = keyframes every 4 m (db_build.keyframes) outside the query lap and VAL_LAPS. Error = GT distance
from the query set to the retrieved keyframe.
"""
import json, os
import numpy as np
from camera_model import CAMS, ROOT, GTPoses, frame_stamps
from db_build import keyframes
from eval_single import DEV_LAPS, allowed_mask, lap_of, lap_table

CONFIGS = [CAMS] + [[c] for c in CAMS] + [['fl', 'fr', 'r', 'rl']]
EVERY = 5


def top3(sim):
    o = np.argsort(-sim)[:40]
    out = [o[0]]
    for i in o[1:]:
        if len(out) == 3:
            break
        if all(abs(i - j) >= 8 for j in out):
            out.append(i)
    return out


def main():
    gt, laps, st = GTPoses(), lap_table(), frame_stamps()
    kf = keyframes(gt, st)
    kr = np.searchsorted(st, kf)
    G = {c: np.load(os.path.join(ROOT, 'features', f'global_{c}.npy')).astype(np.float32) for c in CAMS}
    kx, ky, _ = gt.xy_yaw(kf)
    qx, qy, _ = gt.xy_yaw(st)
    lq = lap_of(st, laps)
    rows = np.concatenate([np.where(lq == lap)[0][::EVERY] for lap in DEV_LAPS])
    out = dict(rows=rows.tolist(), t=np.round((st[rows] - gt.t[0]) / 1e9, 1).tolist(),
               x=np.round(qx[rows], 1).tolist(), y=np.round(qy[rows], 1).tolist(), lap=lq[rows].tolist(),
               configs={})
    for cams in CONFIGS:
        name = '+'.join(cams) if len(cams) < len(CAMS) else 'all'
        e1, e3 = [], []
        for r in rows:
            sim = sum(G[c][kr] @ G[c][r] for c in cams)
            sim[~allowed_mask(kf, laps, lq[r])] = -np.inf
            t = top3(sim)
            d = np.hypot(kx[t] - qx[r], ky[t] - qy[r])
            e1.append(d[0]); e3.append(d.min())
        e1, e3 = np.array(e1), np.array(e3)
        out['configs'][name] = dict(err=np.round(e1, 1).tolist(), median=round(float(np.median(e1)), 2),
                                    lt10=round(100 * float(np.mean(e1 < 10)), 1),
                                    lt25=round(100 * float(np.mean(e1 < 25)), 1),
                                    gt100=round(100 * float(np.mean(e1 > 100)), 1),
                                    best3_lt10=round(100 * float(np.mean(e3 < 10)), 1))
        c = out['configs'][name]
        print(f"{name:14s} median {c['median']:5.2f} m  <10 m {c['lt10']:5.1f} %  <25 m {c['lt25']:5.1f} %  "
              f">100 m {c['gt100']:4.1f} %  best-of-3 <10 m {c['best3_lt10']:5.1f} %")
    json.dump(out, open(os.path.join(ROOT, 'results', 'retrieval_dev.json'), 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main()
