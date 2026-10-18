"""Keyframe map and single-shot (global) localization of a camera set against it.

MapDB: per camera the keyframes' SuperPoint keypoints, descriptors and 3D points (features/db_<cam>.npz) and the DINOv2
global descriptor of every keyframe (features/global_<cam>.npy).
Relocalizer: LightGlue 2D-3D matches per camera -> per-camera PnP-RANSAC hypotheses scored on all cameras -> robust
multi-camera refinement of T_map_base. The keyframe(s) to match against are chosen by the caller (DINOv2 retrieval).
"""
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .camera_model import CAMS
from .paths import FEAT


class MapDB:
    def __init__(self):
        self.db = {}
        for c in CAMS:
            z = np.load(os.path.join(FEAT, f'db_{c}.npz'))
            self.db[c] = {k: z[k] for k in ('kf_stamp', 'off', 'kps', 'desc', 'xyz')}   # not 'par' (parallax, unused)
        self.kf = self.db[CAMS[0]]['kf_stamp']
        kf_row = np.searchsorted(np.load(os.path.join(FEAT, 'set_stamps.npy')), self.kf)  # training-run set rows
        self.G = np.stack([np.load(os.path.join(FEAT, f'global_{c}.npy'), mmap_mode='r')[kf_row] for c in CAMS])

    def kf_points(self, cam, i):
        d = self.db[cam]
        a, b = d['off'][i], d['off'][i + 1]
        return dict(kps=d['kps'][a:b], desc=d['desc'][a:b]), d['xyz'][a:b]


def T_from_x(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


def x_from_T(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


class Relocalizer:
    def __init__(self, cams, db, fe):
        self.cams, self.db, self.fe = cams, db, fe       # camera_model.Camera per cam, MapDB, vislib.Features

    def correspondences(self, qf, kf_ids):
        corr = {c: ([], []) for c in CAMS}
        for i in kf_ids:
            for c, f in qf.items():
                dbf, xyz = self.db.kf_points(c, i)
                m = self.fe.match(f, dbf)
                if len(m):
                    corr[c][0].append(f['kps'][m[:, 0]]); corr[c][1].append(xyz[m[:, 1]])
        res = {}
        for c, (uv, X) in corr.items():
            if uv:
                uv, X = np.concatenate(uv), np.concatenate(X)
                res[c] = (self.cams[c].undistort_points(uv), X)
        return res

    def residuals(self, T_map_base, corr):
        """pixel reprojection residual norms per camera (dict)."""
        out = {}
        for c, (n, X) in corr.items():
            T = T_map_base @ self.cams[c].T_base_cam
            pc = (X - T[:3, 3]) @ T[:3, :3]
            with np.errstate(all='ignore'):
                e = np.linalg.norm(pc[:, :2] / pc[:, 2:] - n, axis=1) * self.cams[c].K[0, 0]
            e[pc[:, 2] <= 0.5] = np.inf
            out[c] = e
        return out

    def solve(self, corr, thr_px=3.0):
        hyps = []
        for c, (n, X) in corr.items():
            if len(n) < 8:
                continue
            f = self.cams[c].K[0, 0]
            ok, rv, tv, inl = cv2.solvePnPRansac(X.astype(np.float64), n.astype(np.float64), np.eye(3), None,
                                                 reprojectionError=thr_px / f, iterationsCount=300,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or inl is None or len(inl) < 6:
                continue
            C = np.eye(4); C[:3, :3] = cv2.Rodrigues(rv)[0]; C[:3, 3] = tv.ravel()   # T_cam_map
            hyps.append(np.linalg.inv(C) @ np.linalg.inv(self.cams[c].T_base_cam))
        if not hyps:
            return None
        scores = []
        for Tb in hyps:
            r = self.residuals(Tb, corr)
            scores.append(sum(int((e < thr_px).sum()) for e in r.values()))
        Tb = hyps[int(np.argmax(scores))]
        # robust refinement on all cameras
        for it in range(2):
            r = self.residuals(Tb, corr)
            sel = {c: e < (3 * thr_px if it == 0 else thr_px) for c, e in r.items()}
            if sum(s.sum() for s in sel.values()) < 10:
                return None

            def fun(x):
                T = T_from_x(x)
                res = []
                for c, (n, X) in corr.items():
                    s = sel[c]
                    if not s.any():
                        continue
                    Tc = T @ self.cams[c].T_base_cam
                    pc = (X[s] - Tc[:3, 3]) @ Tc[:3, :3]
                    z = np.maximum(pc[:, 2:], 0.1)
                    res.append(((pc[:, :2] / z - n[s]) * self.cams[c].K[0, 0]).ravel())
                return np.concatenate(res)
            sol = least_squares(fun, x_from_T(Tb), loss='huber', f_scale=thr_px / 2, max_nfev=50)
            Tb = T_from_x(sol.x)
        r = self.residuals(Tb, corr)
        inl = {c: int((e < thr_px).sum()) for c, e in r.items()}
        return Tb, sum(inl.values()), inl
