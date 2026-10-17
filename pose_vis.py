#!/usr/bin/env python3
"""Visual material for single-shot pose results (eval_single.py pose ...).

    python3 pose_vis.py --tag dev_flfr --cams fl fr
      -> results/pose_<tag>.json            per-query arrays for the report
         report/img/match_<tag>_<k>.jpg     match examples: query (left) vs top-1 map keyframe (right),
                                            one row per camera; green = consistent with the estimated pose
                                            (reprojection < 3 px), red = rejected
"""
import argparse, json, os
import cv2
import numpy as np
import pandas as pd
from camera_model import ROOT, GTPoses, frame_path, frame_stamps, yaw_of
from eval_single import allowed_mask, lap_table

MAX_LINES = 140


def examples(df):
    ok = df[df.ok].copy()
    picks = [('много совпадений', ok.sort_values('inliers', ascending=False).index[0]),
             ('типичный', (ok.err - ok.err.median()).abs().idxmin()),
             ('быстро', ok.assign(v=ok.v).sort_values('v', ascending=False).index[0]),
             ('худший', ok.err.idxmax())]
    seen, out = set(), []
    for name, i in picks:
        if i not in seen:
            seen.add(i); out.append((name, i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='dev_flfr')
    ap.add_argument('--cams', nargs='+', default=['fl', 'fr'])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, 'results', f'single_{a.tag}.csv'))
    gt, laps, st = GTPoses(), lap_table(), frame_stamps()
    gx, gy, gyaw = gt.xy_yaw(df.stamp.values)
    t0 = gt.t[0]
    df['gx'], df['gy'] = gx, gy
    xa, ya, _ = gt.xy_yaw(df.stamp.values - 50_000_000)
    xb, yb, _ = gt.xy_yaw(df.stamp.values + 50_000_000)
    df['v'] = np.hypot(xb - xa, yb - ya) / 0.1

    from localize import Localizer
    loc = Localizer(a.cams)
    ex_meta = []
    for k, (name, i) in enumerate(examples(df)):
        r = df.loc[i]
        stamp, lap = int(r.stamp), int(r.lap)
        allowed = allowed_mask(loc.db.kf, laps, lap)
        cand, _ = loc.db.retrieve(int(np.searchsorted(st, stamp)), allowed, loc.k)
        qf = loc.query_features(stamp)
        out = loc.solve(loc.correspondences(qf, cand))
        if out is None:
            continue
        T = out[0]
        kf = cand[0]
        rows = []
        n_in = n_all = 0
        for c in a.cams:
            dbf, xyz = loc.db.kf_points(c, kf)
            m = loc.fe.match(qf[c], dbf)
            cam = loc.cams[c]
            Tc = T @ cam.T_base_cam
            pc = (xyz[m[:, 1]] - Tc[:3, 3]) @ Tc[:3, :3]
            uv = cam.project_cam(pc)
            res = np.linalg.norm(uv - qf[c]['kps'][m[:, 0]], axis=1)
            inl = np.nan_to_num(res, nan=1e9) < 3.0
            n_in += int(inl.sum()); n_all += len(m)
            q = cv2.imread(frame_path(c, stamp)); d = cv2.imread(frame_path(c, loc.db.kf[kf]))
            canvas = np.hstack([q, d])
            W = q.shape[1]
            sel = np.random.default_rng(0).permutation(len(m))[:MAX_LINES]
            for j in sorted(sel, key=lambda j: inl[j]):
                p0 = tuple(np.round(qf[c]['kps'][m[j, 0]]).astype(int))
                p1 = tuple(np.round(dbf['kps'][m[j, 1]]).astype(int) + np.array([W, 0]))
                col = (60, 200, 60) if inl[j] else (60, 60, 230)
                cv2.line(canvas, p0, p1, col, 1, cv2.LINE_AA)
                cv2.circle(canvas, p0, 2, col, -1); cv2.circle(canvas, p1, 2, col, -1)
            for x, txt in ((10, f'{c}  query  t={(stamp - t0) / 1e9:.1f}s'),
                           (W + 10, f'{c}  map keyframe  t={(loc.db.kf[kf] - t0) / 1e9:.1f}s')):
                cv2.putText(canvas, txt, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, txt, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(canvas)
        img = np.vstack(rows)
        img = cv2.resize(img, (1200, int(img.shape[0] * 1200 / img.shape[1])), interpolation=cv2.INTER_AREA)
        path = os.path.join(ROOT, 'report', 'img', f'match_{a.tag}_{k}.jpg')
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        kx, ky, _ = gt.xy_yaw(loc.db.kf[kf])
        kf_lap = int(np.searchsorted(laps.a.values, loc.db.kf[kf], side='right') - 1)
        ex_meta.append(dict(name=name, img=f'img/match_{a.tag}_{k}.jpg', t=round((stamp - t0) / 1e9, 1), lap=lap,
                            kf_t=round((loc.db.kf[kf] - t0) / 1e9, 1), kf_lap=kf_lap,
                            kf_dist=round(float(np.hypot(kx - r.gx, ky - r.gy)), 2),
                            err=round(float(r.err), 3), lat=round(float(r.lat), 3), lon=round(float(r.lon), 3),
                            dyaw=round(float(r.dyaw), 3), inliers=int(r.inliers), v=round(float(r.v), 1),
                            shown_matches=n_all, shown_inliers=n_in))
        print(name, ex_meta[-1])
    ok = df.ok.values
    res = dict(tag=a.tag, cams=a.cams, n=int(len(df)), solved=round(100 * float(ok.mean()), 1),
               t=np.round((df.stamp.values - t0) / 1e9, 1).tolist(), lap=df.lap.astype(int).tolist(),
               gx=np.round(df.gx, 2).tolist(), gy=np.round(df.gy, 2).tolist(),
               x=np.round(df.x.fillna(0), 2).tolist(), y=np.round(df.y.fillna(0), 2).tolist(),
               err=np.round(df.err.fillna(-1), 3).tolist(), lat=np.round(df.lat.fillna(0), 3).tolist(),
               lon=np.round(df.lon.fillna(0), 3).tolist(), dyaw=np.round(df.dyaw.fillna(0), 3).tolist(),
               inl=df.inliers.fillna(0).astype(int).tolist(), v=np.round(df.v, 1).tolist(),
               **{f'inl_{c}': df[f'inl_{c}'].fillna(0).astype(int).tolist() for c in a.cams},
               examples=ex_meta)
    json.dump(res, open(os.path.join(ROOT, 'results', f'pose_{a.tag}.json'), 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main()
