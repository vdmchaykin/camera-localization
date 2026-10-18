"""The localizer: keyframe map + day descriptor layer (day_layer.py and its bases) + EKF with car state.

    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)
    loc.summary(); loc.close()

first_pose   dict(stamp_ns, x, y, yaw) - the first pose of the run, if given. None = not known: the localizer finds
             itself (global relocalization on its first camera sets, see below), causally.
camera_info  {cam: dict(k, d, width, height, distortion_model)}, tf_static {'<parent> -> <child>': transform},
track_bounds {name: (N, 3) array} - from the run's mcap (camloc/bag.py).
images       {cam: BGR image or None} of one camera set; car_state: list of dict(stamp_ns, vx_mps, vy_mps, wz_radps,
             ...) with stamps <= stamp_ns not delivered before.

Without a first pose (boot), per camera set while not initialised:
  - global relocalization (DINOv2 retrieval + LightGlue + PnP-RANSAC); a solution counts if it has
    >= BOOT_MIN_INL inliers and lies on the drivable area;
  - every counted solution is carried forward to the current stamp with the car state (vx, vy, wz);
  - after >= BOOT_N sets: if >= BOOT_MIN_AGREE (and at least half) of them lie within BOOT_AGREE_M of their median,
    that median (circular-mean yaw) is the pose at this stamp and tracking starts; otherwise boot goes on (after
    BOOT_MAX sets the cluster around the solution with the most inliers is taken);
  - while booting, step() returns the current best guess.
Env: CAMLOC_BOOT_N (10), CAMLOC_BOOT_MIN_INL (30), CAMLOC_BOOT_AGREE_M (2.0), CAMLOC_BOOT_MIN_AGREE (3),
     CAMLOC_BOOT_MAX (50).
"""
import os
import numpy as np

from . import base as v1
from . import day_layer

BOOT_N = int(os.environ.get('CAMLOC_BOOT_N', 10))
BOOT_MIN_INL = int(os.environ.get('CAMLOC_BOOT_MIN_INL', 30))
BOOT_AGREE_M = float(os.environ.get('CAMLOC_BOOT_AGREE_M', 2.0))
BOOT_MIN_AGREE = int(os.environ.get('CAMLOC_BOOT_MIN_AGREE', 3))
BOOT_MAX = int(os.environ.get('CAMLOC_BOOT_MAX', 50))


class Localizer(day_layer.Localizer):
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        self.booting = first_pose is None
        fp = first_pose if first_pose is not None else dict(stamp_ns=0, x=0.0, y=0.0, yaw=0.0)
        super().__init__(fp, camera_info, tf_static, track_bounds)
        self.boot, self.boot_cs = [], []        # boot solutions dict(t, x, y, yaw, inl, ok); car state (t, vx, vy, wz)
        self.guess = (0.0, 0.0, 0.0)
        self.boot_info = dict(given_first_pose=first_pose is not None)

    # ------------------------------------------------------------ boot (no first pose)
    def _carry(self, x, y, yaw, t_from, t_to):
        """dead-reckon a pose from t_from to t_to with the buffered car state (zero-order hold)."""
        cs = self.boot_cs
        if not cs:
            return x, y, yaw
        ts = np.array([c[0] for c in cs])
        i = max(int(np.searchsorted(ts, t_from, side='right')) - 1, 0)
        tc = t_from
        while tc < t_to:
            tn = min(cs[i + 1][0], t_to) if i + 1 < len(cs) else t_to
            dt = (tn - tc) / 1e9
            if dt > 0:
                _, vx, vy, wz = cs[i]
                ym = yaw + 0.5 * wz * dt
                x += (vx * np.cos(ym) - vy * np.sin(ym)) * dt
                y += (vx * np.sin(ym) + vy * np.cos(ym)) * dt
                yaw += wz * dt
            tc = tn
            if i + 1 < len(cs) and cs[i + 1][0] <= tc:
                i += 1
        return x, y, yaw

    def _vote(self, t):
        ok = [b for b in self.boot if b['ok']]
        if not ok:
            return None
        P = np.array([self._carry(b['x'], b['y'], b['yaw'], b['t'], t) for b in ok])
        med = np.median(P[:, :2], 0)
        agree = np.linalg.norm(P[:, :2] - med, axis=1) < BOOT_AGREE_M
        need = max(BOOT_MIN_AGREE, (len(ok) + 1) // 2)
        if agree.sum() < need:
            if len(self.boot) < BOOT_MAX:
                return med, None
            best = P[int(np.argmax([b['inl'] for b in ok]))]          # BOOT_MAX reached: trust the best solution
            agree = np.linalg.norm(P[:, :2] - best[:2], axis=1) < BOOT_AGREE_M
        A = P[agree]
        return med, (float(np.median(A[:, 0])), float(np.median(A[:, 1])),
                     float(np.angle(np.mean(np.exp(1j * A[:, 2])))), int(agree.sum()), len(ok))

    def _boot_step(self, stamp_ns, images, car_state):
        t = int(stamp_ns)
        for m in car_state or []:
            self.boot_cs.append((int(m['stamp_ns']), m['vx_mps'], m['vy_mps'], m['wz_radps']))
        r = v1.Localizer._relocalize(self, t, images)
        ok = r is not None and r[3] >= BOOT_MIN_INL and self.drivable.inside(r[0], r[1])
        self.boot.append(dict(t=t, x=None if r is None else r[0], y=None if r is None else r[1],
                              yaw=None if r is None else r[2], inl=0 if r is None else int(r[3]), ok=bool(ok)))
        if ok:
            self.guess = (r[0], r[1], r[2])
        v = self._vote(t) if len(self.boot) >= BOOT_N else None
        if v is not None and v[1] is not None:
            x, y, yaw, n_agree, n_ok = v[1]
            self.first_pose = dict(stamp_ns=t, x=x, y=y, yaw=yaw)
            self.boot_info.update(first_pose=self.first_pose, n_sets=len(self.boot), n_ok=n_ok, n_agree=n_agree,
                                  inliers=[b['inl'] for b in self.boot])
            self.booting = False
            return super().step(stamp_ns, images, car_state)    # the filter starts here
        if v is not None:
            self.guess = (float(v[0][0]), float(v[0][1]), self.guess[2])
        return self.guess

    def step(self, stamp_ns, images, car_state):
        if self.booting:
            return self._boot_step(stamp_ns, images, car_state)
        return super().step(stamp_ns, images, car_state)

    def summary(self):
        s = super().summary()
        s['boot'] = dict(self.boot_info, not_initialised=self.booting) if self.booting else self.boot_info
        return s
