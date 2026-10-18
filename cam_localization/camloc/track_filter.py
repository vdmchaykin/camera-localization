#!/usr/bin/env python3
"""Causal EKF: car-state dead reckoning + absolute pose corrections from the keyframe map (PnP).

State [x, y, yaw, k, bw]: pose in `map`, speed scale factor k (v_true = k * v_mps) and yaw-rate bias bw
(w_true = wz_radps - bw). Predict with every whitelisted car-state message with stamp <= t; at each camera
set stamp t optionally update with a PnP pose computed from images at t; output one pose per camera set.
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


def run(cs, stamps, meas, x0, y0, yaw0, ekf_cls=TrackEKF):
    """cs: car state (stamp_ns, v_mps, wz_radps); stamps: camera set stamps (sorted);
    meas: dict stamp -> (x, y, yaw, inliers). Returns (N,3) poses at the stamps."""
    f = ekf_cls(x0, y0, yaw0)
    ts, vx, vy, wz = cs.stamp_ns.values, cs.vx_mps.values, cs.vy_mps.values, cs.wz_radps.values
    j = max(0, np.searchsorted(ts, stamps[0], side='right') - 1)
    t_cur = stamps[0]
    out = np.zeros((len(stamps), 3))
    f.meas_log = []          # (stamp, accepted, inliers) for every measurement offered to the filter
    for n, t in enumerate(stamps):
        # integrate car-state messages up to t (each message holds until the next one)
        while j + 1 < len(ts) and ts[j + 1] <= t:
            f.predict(vx[j], vy[j], wz[j], (ts[j + 1] - t_cur) / 1e9)
            t_cur = ts[j + 1]; j += 1
        f.predict(vx[j], vy[j], wz[j], (t - t_cur) / 1e9)
        t_cur = t
        if t in meas:
            f.meas_log.append((t, f.update(*meas[t]), meas[t][3]))
        out[n] = f.X[:3]
    return out, f

