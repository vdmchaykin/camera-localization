"""Our method behind the sim/simulate.py interface: keyframe-map relocalization (localize.py) + causal EKF.

    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

Frozen config (results/FROZEN_CONFIG.txt): 7 cameras, k=1 map candidate, SuperPoint 512 keypoints, LightGlue,
PnP-RANSAC + multi-camera refinement; a correction every 0.5 s; track_filter.TrackEKF with car state.
Without car state (car_state is None) a constant-velocity / constant-turn-rate EKF (CVEKF) predicts instead.

Inputs at run time are only the constructor/step arguments. The trained model is read from the project:
features/db_*.npz + features/global_*.npy (3D keyframe map from the training run), masks/*.png, DINOv2 /
SuperPoint / LightGlue weights (torch hub / lightglue package cache). Calibration comes from the arguments.

Runtime: relocalization (~0.5-0.7 s on a GTX 1080) runs in a background thread on the newest camera set. Its pose
for stamp t_m is fused when it is ready: the filter is rewound to its snapshot at t_m, updated, and propagated
again to the current stamp (delayed measurement; only data with stamp <= current step is used). step() itself
only integrates the car state, so a pose is returned for every set within a few ms.
Env LOC_SYNC=1: relocalize inside step() (deterministic; step blocks for the whole relocalization every 0.5 s).
Env LOC_MAP_CONFIG=<json>: map subset (default sim/map_config.json, see there; hidden test: map_config_test.json).
"""
import copy, json, os, sys, threading, time
from collections import deque
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from camera_model import CAMS, H, W, Camera, tf_mat, yaw_of  # noqa: E402
from global_desc import IN_H, IN_W, MEAN, STD, describe  # noqa: E402
from localize import Localizer as _Reloc, MapDB  # noqa: E402
from track_filter import TrackEKF, wrap  # noqa: E402
from vislib import Features  # noqa: E402

PERIOD_S = 0.5
K_CAND = 1
MAX_KP = 512
HIST_S = 10.0
_CLAHE = cv2.createCLAHE(2.0, (8, 8))


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


class Localizer:
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        t_start = time.perf_counter()
        self.first_pose = first_pose
        self.track_bounds = track_bounds            # not used by this method (kept for the interface)
        # cameras from the given calibration (camera_model.Camera: half-res intrinsics + body masks)
        self.cams = {}
        for c in CAMS:
            t = tf_static[f'constructor0/base_link -> constructor0/camera_{c}']
            r, p = t['rotation'], t['translation']
            info = camera_info[c]
            assert abs(info['width'] * 0.5 - W) <= 1, f'map is for 1506 px wide images, got {info["width"]}'
            self.cams[c] = Camera(c, info, tf_mat([r['x'], r['y'], r['z'], r['w']], [p['x'], p['y'], p['z']]))
            _ = self.cams[c].mask
        # map (trained model) + subset for evaluation hygiene
        cfg_p = os.environ.get('LOC_MAP_CONFIG', os.path.join(HERE, 'map_config.json'))
        cfg = json.load(open(cfg_p))
        self.reloc = _Reloc.__new__(_Reloc)          # reuse localize.Localizer matching/PnP code, our cameras
        self.reloc.cams, self.reloc.k, self.reloc.cams_used = self.cams, K_CAND, list(CAMS)
        self.reloc.db = MapDB(CAMS)
        self.reloc.fe = Features(max_kp=MAX_KP)
        db = self.reloc.db
        self.kf = db.kf.astype(np.int64)
        self.allowed_base = np.ones(len(self.kf), bool)
        for a, b in cfg['exclude_windows_ns']:
            self.allowed_base &= (self.kf < a) | (self.kf > b)
        self.self_excl_ns = int(cfg['self_exclude_s'] * 1e9)
        self.Gk = torch.from_numpy(np.stack([db.G[c][db.kf_row] for c in CAMS])).cuda()   # (7, K, 768)
        db.G = None                                   # per-frame descriptors of the training run: not needed
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).eval().cuda()
        # filter state
        self.f = None
        self.use_state = None
        self.cs_t, self.cs_v = [], []                 # car-state buffer: stamps, (vx, vy, wz)
        self.hist = deque()                           # (stamp, filter snapshot) after propagation to each step
        self.t_last = None
        # relocalization worker
        self.sync = os.environ.get('LOC_SYNC', '0') == '1'
        self._lock = threading.Lock()
        self._job = None
        self._busy = False
        self._results = deque()
        self._last_req = None
        self.reloc_ms, self.fuse_delay_ms, self.n_meas, self.n_fail = [], [], 0, 0
        self.n_allowed = int(self.allowed_base.sum())
        self._relocalize(int(first_pose['stamp_ns']), {c: np.zeros((2 * H, 2 * W, 3), np.uint8) for c in CAMS})  # warm-up
        self.reloc_ms.clear()
        self._stop = threading.Event()
        self._thread = None
        if not self.sync:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        self.init_s = time.perf_counter() - t_start

    # ------------------------------------------------------------ relocalization (images -> map pose)
    @torch.no_grad()
    def _relocalize(self, stamp, images):
        t0 = time.perf_counter()
        half = {c: cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for c, im in images.items() if im is not None}
        if not half:
            return None
        names = [c for c in CAMS if c in half]
        x = []
        for c in names:
            im = half[c].copy()
            im[self.cams[c].mask == 0] = 127
            im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
            x.append(((im - MEAN) / STD).transpose(2, 0, 1))
        g = describe(self.dino, torch.from_numpy(np.stack(x)).cuda())                         # (n, 768)
        idx = torch.tensor([CAMS.index(c) for c in names], device='cuda')
        sim = torch.einsum('ckd,cd->k', self.Gk[idx].float(), g).cpu().numpy()
        allowed = self.allowed_base.copy()
        if self.self_excl_ns > 0:
            allowed &= np.abs(self.kf - stamp) > self.self_excl_ns
        sim[~allowed] = -np.inf
        cand = [int(np.argmax(sim))]
        qf = {c: self.reloc.fe.extract(_CLAHE.apply(cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY)), self.cams[c].mask)
              for c in names}
        out = self.reloc.solve(self.reloc.correspondences(qf, cand))
        self.reloc_ms.append((time.perf_counter() - t0) * 1e3)
        if out is None:
            return None
        T, n_inl, _ = out
        return float(T[0, 3]), float(T[1, 3]), float(yaw_of(T)), int(n_inl)

    def _worker(self):
        while not self._stop.is_set():
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                time.sleep(0.002)
                continue
            try:
                res = self._relocalize(*job)
            except Exception as e:                    # never take the filter down
                print(f'[my_loc] relocalization failed: {e!r}', flush=True)
                res = None
            with self._lock:
                self._results.append((job[0], res))
                self._busy = False

    # ------------------------------------------------------------ filter
    def _propagate(self, f, t_from, t_to):
        if t_to <= t_from:
            return
        if not self.use_state:
            f.predict((t_to - t_from) / 1e9)
            return
        ts = self.cs_t
        i = int(np.searchsorted(ts, t_from, side='right')) - 1
        hold = self.cs_v[max(i, 0)] if ts else None     # before the first message: hold the first one
        tc = t_from
        j = i + 1
        while j < len(ts) and ts[j] <= t_to:
            if hold is not None:
                f.predict(*hold, (ts[j] - tc) / 1e9)
            tc, hold = ts[j], self.cs_v[j]
            j += 1
        if hold is not None:
            f.predict(*hold, (t_to - tc) / 1e9)

    def _fuse(self, t_m, meas, t_now):
        """apply a measurement taken at step stamp t_m and re-propagate to t_now."""
        snaps = list(self.hist)
        k = next((i for i, (t, _) in enumerate(snaps) if t == t_m), None)
        if k is None:
            return
        f = copy.deepcopy(snaps[k][1])
        f.update(*meas)
        self.hist[k] = (t_m, copy.deepcopy(f))
        for i in range(k + 1, len(snaps)):
            self._propagate(f, snaps[i - 1][0], snaps[i][0])
            self.hist[i] = (snaps[i][0], copy.deepcopy(f))
        self.f = f
        self.fuse_delay_ms.append((t_now - t_m) / 1e6)

    def step(self, stamp_ns, images, car_state):
        t = int(stamp_ns)
        if self.f is None:
            self.use_state = car_state is not None
            fp = self.first_pose
            self.f = (TrackEKF if self.use_state else CVEKF)(fp['x'], fp['y'], fp['yaw'])
            self.t_last = t
        if self.use_state and car_state:
            for m in car_state:
                self.cs_t.append(int(m['stamp_ns'])); self.cs_v.append((m['vx_mps'], m['vy_mps'], m['wz_radps']))
        self._propagate(self.f, self.t_last, t)
        self.t_last = t
        self.hist.append((t, copy.deepcopy(self.f)))
        while self.hist and self.hist[0][0] < t - HIST_S * 1e9:
            self.hist.popleft()
        if self.use_state and len(self.cs_t) > 4000:      # keep ~10 s of car state before the oldest snapshot
            cut = int(np.searchsorted(self.cs_t, self.hist[0][0], side='right')) - 2
            if cut > 0:
                del self.cs_t[:cut], self.cs_v[:cut]
        # relocalization: request on the newest set every PERIOD_S, fuse whatever has finished
        due = self._last_req is None or t - self._last_req >= PERIOD_S * 1e9 - 2e7
        if self.sync:
            if due:
                self._last_req = t
                res = self._relocalize(t, images)
                self._results.append((t, res))
        else:
            with self._lock:
                if due and not self._busy:
                    self._job, self._busy, self._last_req = (t, images), True, t
        while True:
            with self._lock:
                if not self._results:
                    break
                t_m, res = self._results.popleft()
            if res is None:
                self.n_fail += 1
                continue
            self.n_meas += 1
            self._fuse(t_m, res, t)
        x, y, yaw = self.f.X[:3]
        return float(x), float(y), float(wrap(yaw))

    def close(self):
        """stop the relocalization thread (a live thread at interpreter exit aborts the process)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        torch.cuda.synchronize()

    def summary(self):
        r = np.array(self.reloc_ms) if self.reloc_ms else np.array([np.nan])
        d = np.array(self.fuse_delay_ms) if self.fuse_delay_ms else np.array([np.nan])
        return dict(mode='sync' if self.sync else 'async', car_state=self.use_state, init_s=round(self.init_s, 1),
                    map_keyframes_allowed=self.n_allowed, relocalizations=len(self.reloc_ms),
                    reloc_ms_mean=round(float(np.mean(r)), 1), reloc_ms_p95=round(float(np.percentile(r, 95)), 1),
                    fuse_delay_ms_mean=round(float(np.mean(d)), 1), measurements=self.n_meas, failed=self.n_fail,
                    accepted=self.f.n_upd, rejected=self.f.n_rej, recovered=self.f.n_recover)
