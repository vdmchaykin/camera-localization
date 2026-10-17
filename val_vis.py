#!/usr/bin/env python3
"""Material for the validation block (held-out laps, frozen config).

    python3 val_vis.py --tag val_final --meas results/single_val_7cam_k1_512_2hz.csv
      -> results/val.json, report/img/val_lap<L>_<k>.jpg

Frames: track borders projected with the GT pose (green) and with the predicted pose (magenta). The predicted
overlay uses the predicted x, y, yaw with GT z / roll / pitch (the method outputs only x, y, yaw), so the
picture isolates the submitted error. Visualisation only: nothing here feeds the method.
"""
import argparse, json, os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from camera_model import ROOT, GTPoses, frame_path, load_cameras
from eval_single import errors, lap_table

CAMS_SHOW = ['fl', 'cl', 'cr', 'r']


def eval_txt(path):
    out = {}
    for ln in open(path):
        v = ln.split()
        if ln.startswith('position RMSE'): out['rmse'] = float(v[3])
        elif ln.startswith('position median/p95'): out['med'], out['p95'] = float(v[2]), float(v[4])
        elif ln.startswith('lateral RMSE'): out['lat'] = float(v[3])
        elif ln.startswith('longitudinal RMSE'): out['lon'] = float(v[3])
        elif ln.startswith('yaw RMSE'): out['yaw'] = float(v[3])
        elif ln.startswith('within 0.25'): out['lt025'] = float(v[3])
        elif ln.startswith('within 0.50'): out['lt05'] = float(v[3])
        elif ln.startswith('within 1.00'): out['lt1'] = float(v[3])
        elif ln.startswith('coverage'): out['cov'] = float(v[4])
        elif ln.startswith('poses submitted'): out['n'] = int(v[2])
    return out


def resample(p, step):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    u = np.arange(0, s[-1], step)
    return np.stack([np.interp(u, s, p[:, k]) for k in range(p.shape[1])], 1)


def overlay(cams, gt, bounds, stamp, pred_xyyaw, out_path, label):
    Tg = gt.T(stamp)
    Tp = Tg.copy()
    rpy = Rotation.from_matrix(Tg[:3, :3]).as_euler('ZYX')
    Tp[:3, :3] = Rotation.from_euler('ZYX', [pred_xyyaw[2], rpy[1], rpy[2]]).as_matrix()
    Tp[0, 3], Tp[1, 3] = pred_xyyaw[0], pred_xyyaw[1]
    tiles = []
    for c in CAMS_SHOW:
        im = cv2.imread(frame_path(c, stamp))
        cam = cams[c]
        for T, col, r in ((Tg, (60, 220, 60), 3), (Tp, (255, 60, 255), 2)):
            Tc = T @ cam.T_base_cam
            for b in bounds:
                pc = (b - Tc[:3, 3]) @ Tc[:3, :3]
                uv = cam.project_cam(pc)
                ok = np.isfinite(uv[:, 0]) & (np.linalg.norm(pc, axis=1) < 80)
                for p in uv[ok]:
                    cv2.circle(im, (int(p[0]), int(p[1])), r, col, -1 if r == 2 else 1, cv2.LINE_AA)
        cv2.putText(im, c, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, c, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    img = cv2.resize(img, (1200, int(img.shape[0] * 1200 / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='val_final')
    ap.add_argument('--meas', default='results/single_val_7cam_k1_512_2hz.csv')
    ap.add_argument('--laps', type=int, nargs='+', default=[3, 6])
    a = ap.parse_args()
    gt, laps, cams = GTPoses(), lap_table(), load_cameras()
    t0 = gt.t[0]
    bj = json.load(open(os.path.join(ROOT, 'export', 'track_bounds.json')))
    bounds = [resample(np.array(v['points']), 0.5) for v in bj.values()]
    single = pd.read_csv(os.path.join(ROOT, a.meas)).set_index('stamp')
    out = dict(laps={}, frozen=open(os.path.join(ROOT, 'results', 'FROZEN_CONFIG.txt')).read())
    allpos, alllat, alllon, allyaw = [], [], [], []
    for lap in a.laps:
        f = pd.read_csv(os.path.join(ROOT, 'results', f'filter_{a.tag}_lap{lap}.csv'))
        st = f.timestamp_ns.values
        pos, lat, lon, dyaw = errors(gt, st, f.x.values, f.y.values, f.yaw.values)
        allpos.append(pos); alllat.append(lat); alllon.append(lon); allyaw.append(dyaw)
        gx, gy, _ = gt.xy_yaw(st)
        xa, ya, _ = gt.xy_yaw(st - 50_000_000); xb, yb, _ = gt.xy_yaw(st + 50_000_000)
        v = np.hypot(xb - xa, yb - ya) / 0.1
        m = pd.read_csv(os.path.join(ROOT, 'results', f'filter_{a.tag}_lap{lap}_meas.csv'))
        sub = slice(None, None, 2)
        rec = dict(t=np.round((st[sub] - t0) / 1e9, 1).tolist(),
                   x=np.round(f.x.values[sub], 3).tolist(), y=np.round(f.y.values[sub], 3).tolist(),
                   gx=np.round(gx[sub], 3).tolist(), gy=np.round(gy[sub], 3).tolist(),
                   err=np.round(pos[sub], 3).tolist(), lat=np.round(lat[sub], 3).tolist(),
                   lon=np.round(lon[sub], 3).tolist(), dyaw=np.round(dyaw[sub], 3).tolist(),
                   v=np.round(v[sub], 1).tolist(),
                   metrics=eval_txt(os.path.join(ROOT, 'results', f'filter_{a.tag}_lap{lap}.txt')),
                   meas=dict(t=np.round((m.stamp.values - t0) / 1e9, 1).tolist(), ok=m.accepted.astype(int).tolist(),
                             inl=m.inliers.astype(int).tolist(),
                             err=np.round(single.loc[m.stamp.values, 'err'].values, 3).tolist()),
                   cdf=np.round(np.percentile(pos, np.arange(0, 101)), 4).tolist(), frames=[])
        moments = [('худший момент', int(np.argmax(pos))), ('типичный', int(np.argmin(np.abs(pos - np.median(pos))))),
                   ('самый быстрый', int(np.argmax(v)))]
        for k, (name, i) in enumerate(moments):
            path = os.path.join(ROOT, 'report', 'img', f'val_lap{lap}_{k}.jpg')
            overlay(cams, gt, bounds, int(st[i]), (f.x.values[i], f.y.values[i], f.yaw.values[i]), path, name)
            rec['frames'].append(dict(name=name, img=f'img/val_lap{lap}_{k}.jpg', t=round((st[i] - t0) / 1e9, 1),
                                      err=round(float(pos[i]), 3), lat=round(float(lat[i]), 3),
                                      lon=round(float(lon[i]), 3), dyaw=round(float(dyaw[i]), 3), v=round(float(v[i]), 1)))
        out['laps'][lap] = rec
        print(lap, rec['metrics'])
    pos, lat, lon, dyaw = (np.concatenate(z) for z in (allpos, alllat, alllon, allyaw))
    rms = lambda z: round(float(np.sqrt(np.mean(z ** 2))), 3)
    out['combined'] = dict(n=int(len(pos)), rmse=rms(pos), med=round(float(np.median(pos)), 3),
                           p95=round(float(np.percentile(pos, 95)), 3), lat=rms(lat), lon=rms(lon), yaw=rms(dyaw),
                           lt025=round(100 * float(np.mean(pos < .25)), 1), lt05=round(100 * float(np.mean(pos < .5)), 1),
                           lt1=round(100 * float(np.mean(pos < 1)), 1))
    print('combined', out['combined'])
    json.dump(out, open(os.path.join(ROOT, 'results', 'val.json'), 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main()
