"""Shared geometry: cameras (half-res intrinsics, extrinsics, body masks).

All 4x4 transforms are T_a_b (maps points in frame b to frame a).
"""
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .paths import MASKS
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


def yaw_of(T):
    return np.arctan2(T[..., 1, 0], T[..., 0, 0])
