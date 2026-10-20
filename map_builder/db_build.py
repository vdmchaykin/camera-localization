#!/usr/bin/env python3
"""Build the 3D keyframe map: keyframes every KF_SPACING m of GT travel; per camera SuperPoint features of
each keyframe are matched with the next keyframe and triangulated with GT poses (training-time use of GT).

    python3 db_build.py      -> features/db_<cam>.npz:
        kf_stamp (K,) set stamps, off (K+1,) offsets, kps (N,2), desc (N,256) f16, xyz (N,3) map frame,
        par (N,) parallax deg
"""
import os, time
import cv2
import numpy as np
from camera_model import CAMS, ROOT, GTPoses, frame_path, frame_stamps, load_cameras
from vislib import Features

OUT = os.path.join(ROOT, 'features')
KF_SPACING = 4.0
MAX_PTS = 500
MIN_PARALLAX_DEG = 0.75
MAX_DEPTH = 150.0
MAX_REPROJ_PX = 1.5


def keyframes(gt, st):
    x, y, _ = gt.xy_yaw(st)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    k = np.searchsorted(d, np.arange(0, d[-1], KF_SPACING))
    return st[np.unique(k)]


def triangulate(cam, T0, T1, uv0, uv1):
    """T*: T_map_cam. uv*: half-res pixels (M,2). -> xyz (M,3) map, ok (M,), parallax deg (M,)."""
    n0, n1 = cam.undistort_points(uv0), cam.undistort_points(uv1)
    C0, C1 = np.linalg.inv(T0), np.linalg.inv(T1)    # T_cam_map
    X = cv2.triangulatePoints(C0[:3], C1[:3], n0.T, n1.T)
    X = (X[:3] / X[3]).T
    ok = np.isfinite(X).all(1)
    p0 = X @ C0[:3, :3].T + C0[:3, 3]
    p1 = X @ C1[:3, :3].T + C1[:3, 3]
    ok &= (p0[:, 2] > 1) & (p1[:, 2] > 1) & (np.linalg.norm(p0, axis=1) < MAX_DEPTH)
    f = cam.K[0, 0]
    with np.errstate(all='ignore'):
        e0 = np.linalg.norm(p0[:, :2] / p0[:, 2:] - n0, axis=1) * f
        e1 = np.linalg.norm(p1[:, :2] / p1[:, 2:] - n1, axis=1) * f
        r0 = X - T0[:3, 3]; r1 = X - T1[:3, 3]
        cosang = np.sum(r0 * r1, 1) / (np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1))
    par = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ok &= (e0 < MAX_REPROJ_PX) & (e1 < MAX_REPROJ_PX) & (par > MIN_PARALLAX_DEG)
    return X, ok, par


def main():
    os.makedirs(OUT, exist_ok=True)
    gt, cams, fe = GTPoses(), load_cameras(), Features()
    kf = keyframes(gt, frame_stamps())
    T_base = gt.T(kf)
    print(f'{len(kf)} keyframes', flush=True)
    for cn in CAMS:
        p = os.path.join(OUT, f'db_{cn}.npz')
        if os.path.exists(p):
            continue
        cam = cams[cn]
        T_cam = T_base @ cam.T_base_cam
        t0 = time.time()
        feats = [None] * len(kf)
        best = [None] * len(kf)          # per kf: (xyz (N,3), par (N,)) aligned with feats kps, nan if none

        def get(i):
            if feats[i] is None:
                path = frame_path(cn, kf[i])
                feats[i] = Features.load_gray(path) if path else None
                if feats[i] is not None:
                    feats[i] = fe.extract(feats[i], cam.mask)
                    n = len(feats[i]['kps'])
                    best[i] = (np.full((n, 3), np.nan, np.float32), np.zeros(n, np.float32))
            return feats[i]

        kps_l, desc_l, xyz_l, par_l, off = [], [], [], [], [0]

        def flush(i):
            f = feats[i]
            if f is not None:
                xyz, par = best[i]
                sel = np.where(np.isfinite(xyz[:, 0]))[0]
                sel = sel[np.argsort(-f['scores'][sel])][:MAX_PTS]
                kps_l.append(f['kps'][sel]); desc_l.append(f['desc'][sel])
                xyz_l.append(xyz[sel]); par_l.append(par[sel])
                off.append(off[-1] + len(sel))
            else:
                off.append(off[-1])
            feats[i] = best[i] = None

        for i in range(len(kf)):
            f0 = get(i)
            if i + 1 < len(kf):
                f1 = get(i + 1)
                if f0 is not None and f1 is not None:
                    m = fe.match(f0, f1)
                    if len(m):
                        X, ok, par = triangulate(cam, T_cam[i], T_cam[i + 1], f0['kps'][m[:, 0]], f1['kps'][m[:, 1]])
                        for side, (j, idx) in enumerate(((i, m[:, 0]), (i + 1, m[:, 1]))):
                            bx, bp = best[j]
                            upd = ok & (par > bp[idx])
                            bx[idx[upd]] = X[upd]; bp[idx[upd]] = par[upd]
            flush(i)
            if i % 500 == 0:
                print(f'{cn} {i}/{len(kf)} pts {off[-1]} {time.time() - t0:.0f} s', flush=True)
        np.savez(p, kf_stamp=kf, off=np.array(off), kps=np.concatenate(kps_l), desc=np.concatenate(desc_l),
                 xyz=np.concatenate(xyz_l), par=np.concatenate(par_l))
        print(f'{cn}: {off[-1]} points, {off[-1] / len(kf):.0f}/kf, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
