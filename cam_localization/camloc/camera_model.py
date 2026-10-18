"""Shared geometry: cameras (half-res intrinsics, extrinsics), GT poses, track centre line.

All 4x4 transforms are T_a_b (maps points in frame b to frame a).
"""
import json, os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from .paths import MASKS, PROJECT as ROOT
# training-run export (only for the offline map tools; the localizer does not read it)
EX = os.environ.get('CAMLOC_EXPORT', os.path.join(ROOT, 'data', 'export'))
FRAMES = os.environ.get('CAMLOC_FRAMES', os.path.join(ROOT, 'data', 'frames'))
CAMS = ['fl', 'fr', 'cl', 'cr', 'rl', 'rr', 'r']
SCALE = 0.5
W, H = 753, 364


def tf_mat(q, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


class Camera:
    def __init__(self, name, info, T_base_cam):
        self.name = name
        K = np.array(info['k'], float).reshape(3, 3)
        self.K_full = K
        self.K = K.copy()
        self.K[0, 0] *= SCALE; self.K[1, 1] *= SCALE
        self.K[0, 2] = (K[0, 2] + 0.5) * SCALE - 0.5
        self.K[1, 2] = (K[1, 2] + 0.5) * SCALE - 0.5
        self.D = np.array(info['d'], float)
        self.T_base_cam = T_base_cam
        self._mask = None

    @property
    def mask(self):
        if self._mask is None:
            p = os.path.join(MASKS, f'{self.name}.png')
            self._mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else np.full((H, W), 255, np.uint8)
        return self._mask

    def undistort_points(self, uv):
        """pixel (half-res) -> normalized image coords (N,2)."""
        return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    def project_cam(self, pc):
        """points in camera frame (N,3) -> pixels (N,2), nan where behind / far outside FOV."""
        pc = np.asarray(pc, float)
        uv = np.full((len(pc), 2), np.nan)
        z = pc[:, 2]
        ok = z > 0.5
        x, y = pc[:, 0] / np.where(ok, z, 1), pc[:, 1] / np.where(ok, z, 1)
        ok &= (np.abs(x) < 1.15 * W / 2 / self.K[0, 0]) & (np.abs(y) < 1.15 * H / 2 / self.K[1, 1])
        if ok.any():
            uv[ok] = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        return uv


def load_cameras():
    tfs = json.load(open(os.path.join(EX, 'tf_static.json')))
    info = json.load(open(os.path.join(EX, 'camera_info.json')))
    calib_p = os.path.join(ROOT, 'calib', 'extrinsics_refined.json')
    refined = json.load(open(calib_p)) if os.path.exists(calib_p) else {}
    cams = {}
    for c in CAMS:
        if c in refined:
            T = np.array(refined[c])
        else:
            t = tfs[f'constructor0/base_link -> constructor0/camera_{c}']
            r, p = t['rotation'], t['translation']
            T = tf_mat([r['x'], r['y'], r['z'], r['w']], [p['x'], p['y'], p['z']])
        cams[c] = Camera(c, info[c], T)
    return cams


class GTPoses:
    """map -> base_link GT (full 6-DoF), interpolated. Only for training / calibration / evaluation."""

    def __init__(self):
        g = pd.read_csv(os.path.join(EX, 'gt_tf.csv')).sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
        self.t = g.timestamp_ns.values.astype(np.int64)
        self.xyz = g[['x', 'y', 'z']].values
        self.rot = Rotation.from_quat(g[['qx', 'qy', 'qz', 'qw']].values)
        self.yaw = np.unwrap(g.yaw.values)
        self._slerp = Slerp((self.t - self.t[0]) / 1e9, self.rot)

    def T(self, stamp):
        """4x4 T_map_base at stamp (scalar or array -> (N,4,4))."""
        s = np.atleast_1d(np.asarray(stamp, np.int64))
        tt = np.clip((s - self.t[0]) / 1e9, 0, (self.t[-1] - self.t[0]) / 1e9)
        T = np.tile(np.eye(4), (len(s), 1, 1))
        T[:, :3, :3] = self._slerp(tt).as_matrix()
        for k in range(3):
            T[:, k, 3] = np.interp(tt, (self.t - self.t[0]) / 1e9, self.xyz[:, k])
        return T[0] if np.ndim(stamp) == 0 else T

    def xy_yaw(self, stamp):
        tt = (np.asarray(stamp, np.int64) - self.t[0]) / 1e9
        tg = (self.t - self.t[0]) / 1e9
        return np.interp(tt, tg, self.xyz[:, 0]), np.interp(tt, tg, self.xyz[:, 1]), np.interp(tt, tg, self.yaw)


_INDEX = None


def frame_index():
    """DataFrame indexed by set stamp (= fl stamp), column per cam with that cam's stamp (-1 if missing).
    Cameras are hw-synced; their stamps differ from fl by < 0.1 ms."""
    global _INDEX
    if _INDEX is None:
        st = pd.read_csv(os.path.join(EX, 'image_stamps.csv'))
        ref = np.sort(st[st.cam == 'fl'].stamp_ns.values)
        idx = pd.DataFrame(index=pd.Index(ref, name='stamp'))
        for c in CAMS:
            s = np.sort(st[st.cam == c].stamp_ns.values)
            j = np.searchsorted(s, ref).clip(1, len(s) - 1)
            jj = np.where(np.abs(s[j] - ref) < np.abs(s[j - 1] - ref), j, j - 1)
            idx[c] = np.where(np.abs(s[jj] - ref) < 5_000_000, s[jj], -1)
        _INDEX = idx
    return _INDEX


def frame_stamps():
    """sorted set stamps (fl stamps)."""
    return frame_index().index.values


def frame_path(cam, stamp):
    """path of camera `cam` image of the set with set stamp `stamp` (None if the frame is missing)."""
    s = frame_index().at[int(stamp), cam]
    return os.path.join(FRAMES, cam, f'{int(s)}.jpg') if s >= 0 else None


def yaw_of(T):
    return np.arctan2(T[..., 1, 0], T[..., 0, 0])
