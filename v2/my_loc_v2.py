"""v2 localizer: prior-guided relocalization + calibrated motion model + corner-aware scheduling.

Same interface and async delayed-measurement machinery as sim/my_loc.py (inherited, not modified):
    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

Changes vs v1 (sim/my_loc.py), each switchable by env var for ablations (default = on):
  V2_GUIDED=1   correction from the filter prior: keyframe nearest to the predicted pose (no DINOv2), map points of the
                two nearest keyframes projected with the prior, matched to SuperPoint keypoints only inside a window
                around the projection (descriptor mutual best; no LightGlue), pose refined from the prior with a robust
                least squares on all cameras (no per-camera PnP-RANSAC). Falls back to the v1 path (DINOv2 + LightGlue
                + PnP-RANSAC) at start-up failures, when the filter is uncertain or after repeated guided failures.
  V2_BATCH=1    SuperPoint on the 7 cameras in one batch (same preprocessing as lightglue's extract()).
  V2_SP_RESIZE  SuperPoint input long side: 'native' (default, 753 px) or 1024 (as the map was built; same
                accuracy on K4, ~50 ms slower).
  V2_FAST_SOLVE=1  robust Gauss-Newton with analytic Jacobian instead of scipy least_squares (~40 ms faster).
  V2_CALIB=1    car-state motion model calibrated on development laps (v2/model/motion_calib.json):
                vy_eff = a*vy + b*wz + c (lateral velocity at base_link), wz_eff = s*wz + d, vx_eff = k*vx.
  V2_CORNER=1   scheduling by segment: in corners (|yaw rate| > CORNER_WZ) a new correction starts as soon as the
                previous one is done, on straights every STRAIGHT_PERIOD_S.
Model files: features/ (map, as v1), masks/, v2/model/kf_poses.npz, v2/model/motion_calib.json (v2/build_model.py).
"""
import json, os, sys, threading, time
import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'sim'))
import my_loc as v1  # noqa: E402  sim/my_loc.py
from camera_model import CAMS, H, W, yaw_of  # noqa: E402
from localize import T_from_x, x_from_T  # noqa: E402
from track_filter import TrackEKF, wrap  # noqa: E402
from lightglue.utils import ImagePreprocessor  # noqa: E402

ENV = lambda k, d='1': os.environ.get(k, d) == '1'
GUIDED, BATCH, CALIB, CORNER = ENV('V2_GUIDED'), ENV('V2_BATCH'), ENV('V2_CALIB'), ENV('V2_CORNER')
FAST_SOLVE = ENV('V2_FAST_SOLVE')    # analytic Gauss-Newton instead of scipy least_squares
CORNER_WZ = 0.10            # rad/s
STRAIGHT_PERIOD_S = float(os.environ.get('V2_STRAIGHT_PERIOD', '0.3'))
CORNER_PERIOD_S = 0.0
N_KF = 2                    # keyframes whose points are projected
KF_MAX_DIST = 8.0           # m, prior -> keyframe
KF_MAX_DYAW = np.radians(45)
SIM_MIN = 0.70              # SuperPoint descriptor cosine similarity
WIN_MIN, WIN_MAX = 5.0, 40.0
THR_PX = 3.0
MIN_INLIERS_GUIDED = 40
LOST_SIGMA = 1.5            # m: filter position std above this -> global (v1) relocalization
LOST_FAILS = 3              # consecutive guided failures -> global relocalization


class CalibEKF(TrackEKF):
    """TrackEKF with the calibrated car-state -> base_link motion model."""
    CAL = None

    def predict(self, vx, vy, wz, dt):
        c = self.CAL
        super().predict(c['vx_k'] * vx, c['vy_a'] * vy + c['vy_b'] * wz + c['vy_c'], c['wz_s'] * wz + c['wz_d'], dt)


class Localizer(v1.Localizer):
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        super().__init__(first_pose, camera_info, tf_static, track_bounds)     # map, models, worker, v1 warm-up
        t0 = time.perf_counter()
        z = np.load(os.path.join(HERE, 'model', 'kf_poses.npz'))
        assert (z['kf_stamp'] == self.kf).all(), 'kf_poses.npz does not match the map'
        self.kf_T, self.kf_xy, self.kf_yaw = z['T'], np.c_[z['x'], z['y']], z['yaw']
        CalibEKF.CAL = json.load(open(os.path.join(HERE, 'model', 'motion_calib.json')))
        sp_res = os.environ.get('V2_SP_RESIZE', 'native')        # 'native' = no upsampling; '1024' = as the map was built
        self.pre = ImagePreprocessor(**{**self.reloc.fe.sp.preprocess_conf,
                                        'resize': None if sp_res == 'native' else int(sp_res)})
        self.n_guided = self.n_global = self.n_guided_fail = 0
        self.guided_ms, self.global_ms, self.guided_inl = [], [], []
        self.fails = 0
        self.last_wz = 0.0
        self._v2_ready = True
        self._relocalize(int(first_pose['stamp_ns']), {c: np.zeros((2 * H, 2 * W, 3), np.uint8) for c in CAMS},
                         self._prior_from(first_pose['x'], first_pose['y'], first_pose['yaw'], np.eye(5) * 0.01))
        self.guided_ms.clear(); self.n_guided = self.n_guided_fail = 0; self.fails = 0
        self.init_s += time.perf_counter() - t0

    # ------------------------------------------------------------ features
    @torch.no_grad()
    def _features(self, half):
        """SuperPoint on all cameras; batched when possible. -> {cam: dict(kps, scores, desc)}"""
        names = [c for c in CAMS if c in half]
        grays = {c: v1._CLAHE.apply(cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY)) for c in names}
        fe = self.reloc.fe
        if not BATCH:
            return {c: fe.extract(grays[c], self.cams[c].mask) for c in names}
        x = torch.from_numpy(np.stack([grays[c] for c in names])).float().div(255)[:, None].cuda()
        xr, scale = self.pre(x)
        try:
            f = fe.sp({'image': xr})
        except RuntimeError:                       # unequal keypoint counts -> per image
            return {c: fe.extract(grays[c], self.cams[c].mask) for c in names}
        k = ((f['keypoints'] + 0.5) / scale[None] - 0.5).cpu().numpy()
        s = f['keypoint_scores'].cpu().numpy()
        d = f['descriptors'].cpu().numpy()
        out = {}
        for b, c in enumerate(names):
            ki = np.round(k[b]).astype(int)
            keep = self.cams[c].mask[ki[:, 1].clip(0, H - 1), ki[:, 0].clip(0, W - 1)] > 0
            out[c] = dict(kps=k[b][keep].astype(np.float32), scores=s[b][keep].astype(np.float32),
                          desc=d[b][keep].astype(np.float16))
        return out

    # ------------------------------------------------------------ prior
    def _prior_from(self, x, y, yaw, P):
        return dict(x=float(x), y=float(y), yaw=float(yaw), sig_xy=float(np.sqrt(max(P[0, 0], P[1, 1]))),
                    sig_yaw=float(np.sqrt(P[2, 2])))

    def _nearest_kfs(self, prior, stamp):
        allowed = self.allowed_base.copy()
        if self.self_excl_ns > 0:
            allowed &= np.abs(self.kf - stamp) > self.self_excl_ns
        d = np.hypot(*(self.kf_xy - [prior['x'], prior['y']]).T)
        ok = allowed & (d < KF_MAX_DIST) & (np.abs(wrap(self.kf_yaw - prior['yaw'])) < KF_MAX_DYAW)
        idx = np.where(ok)[0]
        return idx[np.argsort(d[idx])][:N_KF]

    def _T_prior(self, prior, kf_i):
        """prior x, y, yaw with z / roll / pitch of the nearest keyframe."""
        T = self.kf_T[kf_i].copy()
        R_rel = Rotation.from_euler('z', prior['yaw'] - self.kf_yaw[kf_i]).as_matrix()
        T[:3, :3] = R_rel @ T[:3, :3]
        T[0, 3], T[1, 3] = prior['x'], prior['y']
        return T

    # ------------------------------------------------------------ guided correction
    @torch.no_grad()
    def _guided(self, stamp, qf, prior):
        kfs = self._nearest_kfs(prior, stamp)
        if not len(kfs):
            return None
        T0 = self._T_prior(prior, kfs[0])
        corr = {}
        for c, f in qf.items():
            if len(f['kps']) < 8:
                continue
            X = np.concatenate([self.reloc.db.kf_points(c, i)[1] for i in kfs])
            D = np.concatenate([self.reloc.db.kf_points(c, i)[0]['desc'] for i in kfs])
            if len(X) < 8:
                continue
            cam = self.cams[c]
            Tc = T0 @ cam.T_base_cam
            pc = (X - Tc[:3, 3]) @ Tc[:3, :3]
            uv = cam.project_cam(pc)
            vis = np.isfinite(uv[:, 0])
            if vis.sum() < 8:
                continue
            X, D, uv, depth = X[vis], D[vis], uv[vis], pc[vis, 2]
            f_px = cam.K[0, 0]
            win = np.clip(f_px * 2.5 * prior['sig_xy'] / depth + f_px * 2.5 * prior['sig_yaw'] + 3, WIN_MIN, WIN_MAX)
            uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
            k_t = torch.from_numpy(f['kps']).cuda()
            inwin = torch.cdist(uv_t, k_t) < torch.from_numpy(win.astype(np.float32)).cuda()[:, None]   # (n, m)
            S = torch.from_numpy(D).cuda().float() @ torch.from_numpy(f['desc']).cuda().float().T
            S[~inwin] = -1
            bj = S.argmax(1)                         # best query keypoint for each map point
            bi = S.argmax(0)                         # best map point for each query keypoint
            i = torch.arange(len(X), device='cuda')
            ok = ((S[i, bj] > SIM_MIN) & (bi[bj] == i)).cpu().numpy()
            bj = bj.cpu().numpy()
            if ok.sum() < 4:
                continue
            corr[c] = (cam.undistort_points(f['kps'][bj[ok]]), X[ok])
        n_corr = sum(len(v[0]) for v in corr.values())
        if n_corr < MIN_INLIERS_GUIDED:
            return None
        Tb = T0
        for thr in (4 * THR_PX, THR_PX):
            r = self.reloc.residuals(Tb, corr)
            sel = {c: e < thr for c, e in r.items()}
            if sum(s.sum() for s in sel.values()) < MIN_INLIERS_GUIDED:
                return None
            Tb = self._refine(Tb, corr, sel) if FAST_SOLVE else self._refine_scipy(Tb, corr, sel)
        r = self.reloc.residuals(Tb, corr)
        n_inl = int(sum((e < THR_PX).sum() for e in r.values()))
        if n_inl < MIN_INLIERS_GUIDED:
            return None
        return float(Tb[0, 3]), float(Tb[1, 3]), float(yaw_of(Tb)), n_inl

    def _refine(self, Tb, corr, sel, iters=8):
        """robust Gauss-Newton on T_map_base, analytic Jacobian, Huber weights (pixel residuals, all cameras).
        Perturbation: R <- exp([dth]x) R, t <- t + dt (map frame)."""
        n_l, X_l, A_l, f_l, tbc_l, Rbc_l = [], [], [], [], [], []
        for c, (n, X) in corr.items():
            m = sel[c]
            if m.any():
                n_l.append(n[m]); X_l.append(X[m])
                k = int(m.sum())
                cam = self.cams[c]
                f_l.append(np.full(k, cam.K[0, 0])); tbc_l.append(np.repeat(cam.T_base_cam[None, :3, 3], k, 0))
                Rbc_l.append(np.repeat(cam.T_base_cam[None, :3, :3], k, 0))
        n, X, f = np.concatenate(n_l), np.concatenate(X_l), np.concatenate(f_l)
        tbc, Rbc = np.concatenate(tbc_l), np.concatenate(Rbc_l)
        R, t = Tb[:3, :3].copy(), Tb[:3, 3].copy()
        delta = THR_PX / 2
        for _ in range(iters):
            d = X - t
            q = d @ R                                              # R^T d (base frame)
            pc = np.einsum('kji,kj->ki', Rbc, q - tbc)             # Rbc^T (q - tbc)
            z = np.maximum(pc[:, 2], 0.1)
            r = (pc[:, :2] / z[:, None] - n) * f[:, None]
            A = np.einsum('kji,jl->kil', Rbc, R.T)                 # Rbc^T R^T
            sk = np.zeros((len(d), 3, 3))
            sk[:, 0, 1], sk[:, 0, 2], sk[:, 1, 2] = -d[:, 2], d[:, 1], -d[:, 0]
            sk[:, 1, 0], sk[:, 2, 0], sk[:, 2, 1] = d[:, 2], -d[:, 1], d[:, 0]
            Jp = np.zeros((len(d), 2, 3))
            Jp[:, 0, 0] = Jp[:, 1, 1] = f / z
            Jp[:, 0, 2] = -f * pc[:, 0] / z ** 2
            Jp[:, 1, 2] = -f * pc[:, 1] / z ** 2
            J = np.concatenate([Jp @ A @ sk, -Jp @ A], axis=2)     # (k, 2, 6)
            e = np.linalg.norm(r, axis=1)
            w = np.where(e < delta, 1.0, delta / np.maximum(e, 1e-9))
            Hm = np.einsum('k,kai,kaj->ij', w, J, J)
            g = np.einsum('k,kai,ka->i', w, J, r)
            dx = -np.linalg.solve(Hm + 1e-9 * np.eye(6), g)
            R = Rotation.from_rotvec(dx[:3]).as_matrix() @ R
            t = t + dx[3:]
            if np.abs(dx).max() < 1e-6:
                break
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
        return T

    def _refine_scipy(self, Tb, corr, sel):
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
        return T_from_x(least_squares(fun, x_from_T(Tb), loss='huber', f_scale=THR_PX / 2, max_nfev=30).x)

    # ------------------------------------------------------------ relocalization entry (worker / sync)
    @torch.no_grad()
    def _relocalize(self, stamp, images, prior=None):
        if not getattr(self, '_v2_ready', False) or not GUIDED or prior is None \
                or prior['sig_xy'] > LOST_SIGMA or self.fails >= LOST_FAILS:
            t0 = time.perf_counter()
            res = super()._relocalize(stamp, images)             # v1: DINOv2 + LightGlue + PnP-RANSAC
            if getattr(self, '_v2_ready', False):
                self.n_global += 1
                self.global_ms.append((time.perf_counter() - t0) * 1e3)
                self.fails = 0 if res is not None else self.fails
            return res
        t0 = time.perf_counter()
        half = {c: cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for c, im in images.items() if im is not None}
        qf = self._features(half)
        res = self._guided(stamp, qf, prior)
        dt = (time.perf_counter() - t0) * 1e3
        self.guided_ms.append(dt); self.reloc_ms.append(dt)
        self.n_guided += 1
        if res is None:
            self.n_guided_fail += 1
            self.fails += 1
        else:
            self.fails = 0
            self.guided_inl.append(res[3])
        return res

    # ------------------------------------------------------------ step (v1 step + prior in the job + scheduling)
    def step(self, stamp_ns, images, car_state):
        t = int(stamp_ns)
        if self.f is None:
            self.use_state = car_state is not None
            fp = self.first_pose
            cls = (CalibEKF if CALIB else TrackEKF) if self.use_state else v1.CVEKF
            self.f = cls(fp['x'], fp['y'], fp['yaw'])
            self.t_last = t
        if self.use_state and car_state:
            for m in car_state:
                self.cs_t.append(int(m['stamp_ns'])); self.cs_v.append((m['vx_mps'], m['vy_mps'], m['wz_radps']))
            self.last_wz = car_state[-1]['wz_radps']
        elif not self.use_state:
            self.last_wz = float(self.f.X[5])
        self._propagate(self.f, self.t_last, t)
        self.t_last = t
        self.hist.append((t, v1.copy.deepcopy(self.f)))
        while self.hist and self.hist[0][0] < t - v1.HIST_S * 1e9:
            self.hist.popleft()
        if self.use_state and len(self.cs_t) > 4000:
            cut = int(np.searchsorted(self.cs_t, self.hist[0][0], side='right')) - 2
            if cut > 0:
                del self.cs_t[:cut], self.cs_v[:cut]
        if CORNER:
            period = CORNER_PERIOD_S if abs(self.last_wz) > CORNER_WZ else STRAIGHT_PERIOD_S
        else:
            period = v1.PERIOD_S
        due = self._last_req is None or t - self._last_req >= period * 1e9 - 2e7
        prior = self._prior_from(*self.f.X[:3], self.f.P)
        if self.sync:
            if due:
                self._last_req = t
                self._results.append((t, self._relocalize(t, images, prior)))
        else:
            with self._lock:
                if due and not self._busy:
                    self._job, self._busy, self._last_req = (t, images, prior), True, t
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

    def summary(self):
        s = super().summary()
        g = np.array(self.guided_ms) if self.guided_ms else np.array([np.nan])
        G = np.array(self.global_ms) if self.global_ms else np.array([np.nan])
        s.update(flags=dict(guided=GUIDED, batch=BATCH, calib=CALIB, corner=CORNER),
                 guided=self.n_guided, guided_fail=self.n_guided_fail,
                 guided_ms_mean=round(float(np.mean(g)), 1), guided_ms_p95=round(float(np.percentile(g, 95)), 1),
                 guided_inliers_median=int(np.median(self.guided_inl)) if self.guided_inl else 0,
                 global_=self.n_global, global_ms_mean=round(float(np.mean(G)), 1))
        return s
