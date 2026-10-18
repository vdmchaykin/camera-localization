"""Causal EKF: car-state dead reckoning + absolute pose corrections from the keyframe map.

TrackEKF  state [x, y, yaw, k, bw]: pose in `map`, speed scale factor k (v_true = k * v, v = car-state body velocity)
          and yaw-rate bias bw (w_true = wz_radps - bw). Predicted with every car-state message with stamp <= t,
          updated with the pose measured from the camera set at t.
CalibEKF  TrackEKF with the calibrated car state -> base_link motion model (model/motion_calib.json).
CVEKF     no car state: constant velocity / turn rate.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TrackEKF:
    # process noise (per sqrt(s) and per metre travelled)
    Q_POS_T = 0.10        # m / sqrt(s)   (plateau 0.1-0.2 on dev laps 1, 5)
    Q_POS_D = 0.010       # fraction of distance travelled (1 sigma)
    Q_YAW_T = 0.004       # rad / sqrt(s)
    Q_K = 2e-4            # 1 / sqrt(s)
    Q_BW = 2e-5           # rad/s / sqrt(s)
    # measurement noise of a PnP pose with 100 inliers (body frame), scaled by sqrt(100 / inliers)
    R_LAT, R_LON, R_YAW = 0.12, 0.30, np.radians(0.30)
    MIN_SCALE = 0.7       # sqrt(100 / inliers) floored: matches within one frame are correlated, so more
                          # than ~200 inliers do not make the pose more certain
    GATE_CHI2 = 16.0      # 3 dof, ~99.9 %
    MIN_INLIERS = 30
    # recovery: two consecutive rejected measurements asking for the same correction mean the filter is wrong
    RECOVER_POS, RECOVER_YAW = 0.3, np.radians(0.3)

    def __init__(self, x, y, yaw):
        self.X = np.array([x, y, yaw, 1.0, 0.0])
        self.P = np.diag([0.1 ** 2, 0.1 ** 2, np.radians(0.3) ** 2, 0.01 ** 2, 0.002 ** 2])
        self.n_upd = self.n_rej = self.n_recover = 0
        self.since_upd = 0.0
        self.last_rej = None      # innovation of the previous measurement if it was rejected

    def predict(self, vx, vy, wz, dt):
        """vx, vy: body-frame velocity (vy carries the side slip in corners), wz: yaw rate."""
        if dt <= 0:
            return
        x, y, yaw, k, bw = self.X
        w = wz - bw
        ym = yaw + 0.5 * w * dt
        c, s = np.cos(ym), np.sin(ym)
        ux, uy = (vx * c - vy * s) * dt, (vx * s + vy * c) * dt      # map-frame displacement / k
        dx, dy = k * ux, k * uy
        d = np.hypot(dx, dy)
        self.X = np.array([x + dx, y + dy, wrap(yaw + w * dt), k, bw])
        F = np.eye(5)
        F[0, 2] = -dy; F[0, 3] = ux; F[0, 4] = 0.5 * dt * dy
        F[1, 2] = dx;  F[1, 3] = uy; F[1, 4] = -0.5 * dt * dx
        F[2, 4] = -dt
        s2 = self.Q_POS_T ** 2 * dt + (self.Q_POS_D * abs(d)) ** 2
        Q = np.diag([s2, s2, self.Q_YAW_T ** 2 * dt, self.Q_K ** 2 * dt, self.Q_BW ** 2 * dt])
        self.P = F @ self.P @ F.T + Q
        self.since_upd += dt

    def update(self, zx, zy, zyaw, inliers):
        if inliers < self.MIN_INLIERS:
            self.n_rej += 1
            return False
        yaw = zyaw
        c, s = np.cos(yaw), np.sin(yaw)
        sc = max(self.MIN_SCALE, np.sqrt(100.0 / inliers))
        Rb = np.diag([(self.R_LON * sc) ** 2, (self.R_LAT * sc) ** 2])
        Rot = np.array([[c, -s], [s, c]])
        R = np.zeros((3, 3)); R[:2, :2] = Rot @ Rb @ Rot.T; R[2, 2] = (self.R_YAW * sc) ** 2
        H = np.zeros((3, 5)); H[0, 0] = H[1, 1] = H[2, 2] = 1
        r = np.array([zx - self.X[0], zy - self.X[1], wrap(zyaw - self.X[2])])
        S = H @ self.P @ H.T + R
        if r @ np.linalg.solve(S, r) > self.GATE_CHI2:
            prev, self.last_rej = self.last_rej, r
            consistent = prev is not None and np.hypot(*(r[:2] - prev[:2])) < self.RECOVER_POS \
                and abs(wrap(r[2] - prev[2])) < self.RECOVER_YAW
            if consistent:
                # the filter, not the measurements, is off: widen the pose covariance to cover the offset
                self.P[:3, :3] += np.diag([r[0] ** 2, r[1] ** 2, r[2] ** 2]) * 4
                S = H @ self.P @ H.T + R
                self.n_recover += 1
            else:
                self.n_rej += 1
                # long without corrections: let the covariance grow so a consistent fix is accepted again
                if self.since_upd > 5.0:
                    self.P[:3, :3] *= 2.0
                return False
        self.last_rej = None
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ r
        self.X[2] = wrap(self.X[2])
        self.P = (np.eye(5) - K @ H) @ self.P
        self.n_upd += 1
        self.since_upd = 0.0
        return True


class CalibEKF(TrackEKF):
    """TrackEKF with the car state -> base_link motion model calibrated on the training run: vx_eff = vx_k * vx,
    vy_eff = vy_a * vy + vy_b * wz + vy_c (lateral velocity at base_link), wz_eff = wz_s * wz + wz_d."""

    def __init__(self, x, y, yaw, cal):
        super().__init__(x, y, yaw)
        self.cal = cal

    def predict(self, vx, vy, wz, dt):
        c = self.cal
        super().predict(c['vx_k'] * vx, c['vy_a'] * vy + c['vy_b'] * wz + c['vy_c'], c['wz_s'] * wz + c['wz_d'], dt)


class CVEKF:
    """No car state: X = [x, y, yaw, vx, vy, w] (body-frame velocity, yaw rate), white-noise accelerations.
    Same measurement model / gate / recovery rule as TrackEKF."""
    A_LON, A_LAT, A_YAW = 8.0, 3.0, 0.8           # m/s^2, m/s^2, rad/s^2 (1 sigma, white noise)
    Q_POS_T, Q_YAW_T = 0.05, 0.002
    R_LAT, R_LON, R_YAW = TrackEKF.R_LAT, TrackEKF.R_LON, TrackEKF.R_YAW
    MIN_SCALE, GATE_CHI2, MIN_INLIERS = TrackEKF.MIN_SCALE, TrackEKF.GATE_CHI2, TrackEKF.MIN_INLIERS
    RECOVER_POS, RECOVER_YAW = 1.0, np.radians(1.0)

    def __init__(self, x, y, yaw):
        self.X = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        self.P = np.diag([0.1 ** 2, 0.1 ** 2, np.radians(0.3) ** 2, 30.0 ** 2, 3.0 ** 2, 0.5 ** 2])
        self.n_upd = self.n_rej = self.n_recover = 0
        self.since_upd = 0.0
        self.last_rej = None

    @staticmethod
    def f(X, dt):
        x, y, yaw, vx, vy, w = X
        ym = yaw + 0.5 * w * dt
        c, s = np.cos(ym), np.sin(ym)
        return np.array([x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt, yaw + w * dt, vx, vy, w])

    def predict(self, dt):
        if dt <= 0:
            return
        X0 = self.X
        F = np.zeros((6, 6))
        for i in range(6):
            d = np.zeros(6); d[i] = 1e-5
            F[:, i] = (self.f(X0 + d, dt) - self.f(X0 - d, dt)) / 2e-5
        self.X = self.f(X0, dt)
        self.X[2] = wrap(self.X[2])
        Q = np.diag([self.Q_POS_T ** 2 * dt, self.Q_POS_T ** 2 * dt, self.Q_YAW_T ** 2 * dt,
                     self.A_LON ** 2 * dt, self.A_LAT ** 2 * dt, self.A_YAW ** 2 * dt])
        self.P = F @ self.P @ F.T + Q
        self.since_upd += dt

    def update(self, zx, zy, zyaw, inliers):
        n = len(self.X)
        if inliers < self.MIN_INLIERS:
            self.n_rej += 1
            return False
        c, s = np.cos(zyaw), np.sin(zyaw)
        sc = max(self.MIN_SCALE, np.sqrt(100.0 / inliers))
        Rot = np.array([[c, -s], [s, c]])
        R = np.zeros((3, 3))
        R[:2, :2] = Rot @ np.diag([(self.R_LON * sc) ** 2, (self.R_LAT * sc) ** 2]) @ Rot.T
        R[2, 2] = (self.R_YAW * sc) ** 2
        Hm = np.zeros((3, n)); Hm[0, 0] = Hm[1, 1] = Hm[2, 2] = 1
        r = np.array([zx - self.X[0], zy - self.X[1], wrap(zyaw - self.X[2])])
        S = Hm @ self.P @ Hm.T + R
        if r @ np.linalg.solve(S, r) > self.GATE_CHI2:
            prev, self.last_rej = self.last_rej, r
            if prev is not None and np.hypot(*(r[:2] - prev[:2])) < self.RECOVER_POS \
                    and abs(wrap(r[2] - prev[2])) < self.RECOVER_YAW:
                self.P[:3, :3] += np.diag(r ** 2) * 4
                S = Hm @ self.P @ Hm.T + R
                self.n_recover += 1
            else:
                self.n_rej += 1
                if self.since_upd > 2.0:
                    self.P[:3, :3] *= 2.0
                return False
        self.last_rej = None
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.X = self.X + K @ r
        self.X[2] = wrap(self.X[2])
        self.P = (np.eye(n) - K @ Hm) @ self.P
        self.n_upd += 1
        self.since_upd = 0.0
        return True
