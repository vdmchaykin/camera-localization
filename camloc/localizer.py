"""The localizer: keyframe map + day descriptor layer + EKF with car state.

    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)
    loc.summary()

first_pose   dict(stamp_ns, x, y, yaw) - the first pose of the run, if given. None = not known: the localizer finds
             itself (global relocalization on its first camera sets, see below), causally.
camera_info  {cam: dict(k, d, width, distortion_model)}, tf_static {'<parent> -> <child>': transform},
track_bounds {name: (N, 3) array} - from the run's mcap (camloc/bag.py).
images       {cam: BGR image or None} of one camera set; car_state: list of dict(stamp_ns, vx_mps, vy_mps, wz_radps)
             with stamps <= stamp_ns not delivered before, or None for a run without car state (a constant-velocity EKF
             predicts instead).

Every camera set:
  1. EKF prediction with the car state up to the set stamp (track_filter.CalibEKF).
  2. Correction guided by the EKF prior:
     - half-res images; per camera a gamma maps the median grey to the night median of the map (model/night_stats.json),
       then CLAHE; saturated pixels (sun, glare: grey >= SAT_GREY, dilated) are masked, a camera with more than
       SAT_CAM_FRAC of them is skipped;
     - SuperPoint (FP16, compiled backbone, MAX_KP keypoints per camera) on all cameras in one batch;
     - the map points of the N_KF keyframes nearest to the prior (at most one per pass) are projected with the prior
       and matched inside a window around the projection (size from the prior sigma): mutual best descriptor among
       the keypoints, then a dense search in the SuperPoint descriptor map for the map points left unmatched. Every map
       point has a night and a day descriptor (day layer, model/day/); the similarity is the max of the two;
     - robust Gauss-Newton on all cameras from the prior; a failed correction is retried once with lower similarity
       thresholds; a pose off the drivable area (track + pit lane borders + BOUNDS_MARGIN) is rejected.
  3. Lost (prior sigma > LOST_SIGMA or LOST_FAILS failed corrections in a row): global relocalization (DINOv2 retrieval
     among the keyframes near the prior + LightGlue + PnP-RANSAC, localize.py), at most every GLOBAL_MIN_S; in between
     the guided correction is retried.
  4. EKF update with the measured pose (gated, see track_filter.TrackEKF).

Without a first pose (boot), per camera set while not initialised:
  - global relocalization over the whole map; a solution counts if it has >= BOOT_MIN_INL inliers and lies on the
    drivable area;
  - every counted solution is carried forward to the current stamp with the car state (vx, vy, wz);
  - after >= BOOT_N sets: if >= BOOT_MIN_AGREE (and at least half) of them lie within BOOT_AGREE_M of their median,
    that median (circular-mean yaw) is the pose at this stamp and tracking starts; otherwise boot goes on (after
    BOOT_MAX sets the cluster around the solution with the most inliers is taken);
  - while booting, step() returns the current best guess.

Env: CAMLOC_DAY (1; 0 = night descriptors only, ablation), CAMLOC_DAY_DIR (model/day), CAMLOC_COMPILE (1; 0 = no
torch.compile), LOC_MAP_CONFIG (configs/map_full.json; tools/run.py --map-config), CAMLOC_BOOT_N (10),
CAMLOC_BOOT_MIN_INL (30), CAMLOC_BOOT_AGREE_M (2.0), CAMLOC_BOOT_MIN_AGREE (3), CAMLOC_BOOT_MAX (50).
"""
import json, os, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints
from scipy.spatial.transform import Rotation

from .camera_model import CAMS, H, W, Camera, tf_mat, yaw_of
from .global_desc import IN_H, IN_W, MEAN, STD, describe
from .localize import MapDB, Relocalizer
from .paths import CONFIGS, DAY as DAY_DIR, MODEL
from .track_filter import CVEKF, CalibEKF, wrap
from .vislib import Features

DAY = os.environ.get('CAMLOC_DAY', '1') == '1'
COMPILE = os.environ.get('CAMLOC_COMPILE', '1') == '1'
# global relocalization (start, lost)
GLOBAL_MAX_KP = 512         # SuperPoint keypoints per camera for LightGlue
LOST_SIGMA = 1.5            # m: filter position std above this -> global relocalization
LOST_FAILS = 10             # failed corrections in a row (1 s at 10 Hz) -> global relocalization
GLOBAL_MIN_S = 1.0          # s between two global attempts
GLOBAL_RADIUS_MIN = 30.0    # m: retrieval only among keyframes within max(this, 4 sigma) of the prior
# guided correction
MAX_KP = 1024               # SuperPoint keypoints per camera
N_KF = 3                    # keyframes whose points are projected
PASS_GAP_S = 20.0           # keyframes closer in time than this are the same pass
KF_MAX_DIST = 8.0           # m, prior -> keyframe
KF_MAX_DYAW = np.radians(45)
SIM_MIN = 0.70              # SuperPoint descriptor cosine similarity of a keypoint match
WIN_MIN, WIN_MAX = 5.0, 40.0
DENSE_R = 8                 # px, max search radius of the dense search
DENSE_SIM = 0.75            # dense search: best similarity ...
DENSE_MARGIN = 0.03         # ... and its margin over the best position more than 2.5 px away
RELAX_SIM_MIN, RELAX_DENSE_SIM = 0.60, 0.65     # retry of a failed correction
THR_PX = 3.0
MIN_INLIERS_GUIDED = 40
# daylight
SAT_GREY, SAT_DILATE, SAT_CAM_FRAC = 250, 11, 0.60
GAMMA_MIN, GAMMA_MAX = 0.35, 2.5
BOUNDS_RES, BOUNDS_MARGIN = 0.25, 1.5     # m
# start without a first pose
BOOT_N = int(os.environ.get('CAMLOC_BOOT_N', 10))
BOOT_MIN_INL = int(os.environ.get('CAMLOC_BOOT_MIN_INL', 30))
BOOT_AGREE_M = float(os.environ.get('CAMLOC_BOOT_AGREE_M', 2.0))
BOOT_MIN_AGREE = int(os.environ.get('CAMLOC_BOOT_MIN_AGREE', 3))
BOOT_MAX = int(os.environ.get('CAMLOC_BOOT_MAX', 50))

_CLAHE = cv2.createCLAHE(2.0, (8, 8))
_G = np.stack(np.meshgrid(np.arange(-DENSE_R, DENSE_R + 1), np.arange(-DENSE_R, DENSE_R + 1)), -1).reshape(-1, 2)


def _backbone(s, x):
    """SuperPoint encoder + heads: -> (score logits (b, 65, h/8, w/8), dense descriptors (b, 256, h/8, w/8))."""
    y = s.relu(s.conv1a(x)); y = s.relu(s.conv1b(y)); y = s.pool(y)
    y = s.relu(s.conv2a(y)); y = s.relu(s.conv2b(y)); y = s.pool(y)
    y = s.relu(s.conv3a(y)); y = s.relu(s.conv3b(y)); y = s.pool(y)
    y = s.relu(s.conv4a(y)); y = s.relu(s.conv4b(y))
    return s.convPb(s.relu(s.convPa(y))), s.convDb(s.relu(s.convDa(y)))


class Drivable:
    """raster of the drivable area: between track_left / track_right (closed loops) + pit lane polygon."""

    def __init__(self, bounds, res=BOUNDS_RES, margin=BOUNDS_MARGIN):
        pts = np.concatenate([v[:, :2] for v in bounds.values()])
        self.lo = pts.min(0) - 20
        hi = pts.max(0) + 20
        self.res = res
        self.shape = tuple((np.ceil((hi - self.lo) / res)).astype(int)[::-1])         # rows (y), cols (x)
        px = lambda v: np.round((v[:, :2] - self.lo) / res).astype(np.int32)
        a = np.zeros(self.shape, np.uint8); b = np.zeros(self.shape, np.uint8)
        cv2.fillPoly(a, [px(bounds['track_left'])], 1)
        cv2.fillPoly(b, [px(bounds['track_right'])], 1)
        m = a ^ b
        if 'pitlane_left' in bounds and 'pitlane_right' in bounds:
            cv2.fillPoly(m, [np.concatenate([px(bounds['pitlane_left']), px(bounds['pitlane_right'])[::-1]])], 1)
        k = int(np.ceil(margin / res)) * 2 + 1
        self.m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    def inside(self, x, y):
        c, r = np.round((np.array([x, y]) - self.lo) / self.res).astype(int)
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1] and bool(self.m[r, c])


class Localizer:
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        t_start = time.perf_counter()
        self.booting = first_pose is None
        self.first_pose = first_pose if first_pose is not None else dict(stamp_ns=0, x=0.0, y=0.0, yaw=0.0)
        self.boot_info = dict(given_first_pose=first_pose is not None)
        # cameras from the given calibration (camera_model.Camera: half-res intrinsics + body masks)
        self.cams = {}
        for c in CAMS:
            t = tf_static[f'constructor0/base_link -> constructor0/camera_{c}']
            r, p = t['rotation'], t['translation']
            info = camera_info[c]
            assert abs(info['width'] * 0.5 - W) <= 1, f'map is for 1506 px wide images, got {info["width"]}'
            assert info['distortion_model'] == 'plumb_bob', f'{c}: {info["distortion_model"]} (K, D read as plumb_bob)'
            self.cams[c] = Camera(c, info, tf_mat([r['x'], r['y'], r['z'], r['w']], [p['x'], p['y'], p['z']]))
            _ = self.cams[c].mask
        # map (trained model) + subset for evaluation hygiene
        cfg = json.load(open(os.environ.get('LOC_MAP_CONFIG', os.path.join(CONFIGS, 'map_full.json'))))
        self.db = MapDB()
        self.fe = Features(max_kp=GLOBAL_MAX_KP)
        self.reloc = Relocalizer(self.cams, self.db, self.fe)
        self.kf = self.db.kf.astype(np.int64)
        self.allowed_base = np.ones(len(self.kf), bool)
        for a, b in cfg['exclude_windows_ns']:
            self.allowed_base &= (self.kf < a) | (self.kf > b)
        self.self_excl_ns = int(cfg['self_exclude_s'] * 1e9)
        self.n_allowed = int(self.allowed_base.sum())
        self.Gk = torch.from_numpy(self.db.G).cuda()                # (7, K, 768) DINOv2 descriptors of the keyframes
        self.db.G = None
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).eval().cuda()
        z = np.load(os.path.join(MODEL, 'kf_poses.npz'))
        assert (z['kf_stamp'] == self.kf).all(), 'kf_poses.npz does not match the map'
        self.kf_T, self.kf_xy, self.kf_yaw = z['T'], np.c_[z['x'], z['y']], z['yaw']
        self.motion_calib = json.load(open(os.path.join(MODEL, 'motion_calib.json')))
        st = json.load(open(os.path.join(MODEL, 'night_stats.json')))
        self._target = {c: st[c]['median_grey'] for c in CAMS}
        self.drivable = Drivable(track_bounds)
        self._day = {c: np.load(os.path.join(DAY_DIR, f'day_{c}.npy'), mmap_mode='r') for c in CAMS} if DAY else {}
        # FP16 channels_last copy of SuperPoint for the guided path; compiled for a fixed batch of 7
        sp = self.fe.sp
        self.sp16 = type(sp)(**vars(sp.conf)).eval().cuda()
        self.sp16.load_state_dict(sp.state_dict())
        self.sp16 = self.sp16.half().to(memory_format=torch.channels_last)
        fn = lambda x: _backbone(self.sp16, x)
        self.backbone = torch.compile(fn, dynamic=False) if COMPILE else fn
        self.G_T = torch.from_numpy(_G.astype(np.float32)).cuda()  # dense search grid offsets (g, 2)
        # filter and per-set state
        self.f, self.use_state, self.t_last = None, None, None
        self.cs_t, self.cs_v = [], []             # car-state buffer: stamps, (vx, vy, wz)
        self.fails, self._last_global = 0, None
        self._ok, self._dense = {}, {}            # per camera of the current set: usable pixels, dense descriptor map
        self.boot, self.boot_cs = [], []    # boot solutions dict(t, x, y, yaw, inl, ok), car state (t, vx, vy, wz)
        self.guess = (0.0, 0.0, 0.0)
        # warm-up on blank images (DINOv2, SuperPoint, compiled backbone); statistics start after it
        self._reset_stats()
        fp, blank = self.first_pose, {c: np.zeros((2 * H, 2 * W, 3), np.uint8) for c in CAMS}
        self._global(int(fp['stamp_ns']), blank)
        self._guided_step(int(fp['stamp_ns']), blank, self._prior_from(fp['x'], fp['y'], fp['yaw'], np.eye(5) * 0.01))
        self.fails = 0
        self._reset_stats()
        self.init_s = time.perf_counter() - t_start

    def _reset_stats(self):
        self.reloc_ms, self.guided_ms, self.global_ms, self.guided_inl, self.gammas = [], [], [], [], []
        self.n_kp_corr, self.n_dense_corr, self.n_day_better = [], [], []
        self.n_meas = self.n_fail = self.n_guided = self.n_guided_fail = self.n_global = 0
        self.n_relax = self.n_relax_ok = self.n_bounds_rej = self.n_global_skip = self.n_cam_skip = 0

    # ------------------------------------------------------------ step
    def step(self, stamp_ns, images, car_state):
        if self.booting:
            return self._boot_step(stamp_ns, images, car_state)
        return self._track(stamp_ns, images, car_state)

    def _track(self, stamp_ns, images, car_state):
        t = int(stamp_ns)
        if self.f is None:
            self.use_state = car_state is not None
            fp = self.first_pose
            self.f = CalibEKF(fp['x'], fp['y'], fp['yaw'], self.motion_calib) if self.use_state \
                else CVEKF(fp['x'], fp['y'], fp['yaw'])
            self.t_last = t
        if self.use_state and car_state:
            for m in car_state:
                self.cs_t.append(int(m['stamp_ns'])); self.cs_v.append((m['vx_mps'], m['vy_mps'], m['wz_radps']))
        self._propagate(self.f, self.t_last, t)
        self.t_last = t
        if self.use_state and len(self.cs_t) > 4000:      # keep ~10 s of car state (propagation starts at t)
            cut = int(np.searchsorted(self.cs_t, t - 10 * 10**9, side='right')) - 2
            if cut > 0:
                del self.cs_t[:cut], self.cs_v[:cut]
        res = self._correct(t, images, self._prior_from(*self.f.X[:3], self.f.P))
        if res is None:
            self.n_fail += 1
        else:
            self.n_meas += 1
            self.f.update(*res)
        x, y, yaw = self.f.X[:3]
        return float(x), float(y), float(wrap(yaw))

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

    # ------------------------------------------------------------ measurement: guided correction or global path
    def _correct(self, stamp, images, prior):
        if not (prior['sig_xy'] > LOST_SIGMA or self.fails >= LOST_FAILS):
            return self._guided_step(stamp, images, prior)
        if self._last_global is not None and stamp - self._last_global < GLOBAL_MIN_S * 1e9:
            # too soon for another global attempt: keep trying the guided path from the (wider) prior instead
            self.n_global_skip += 1
            fails, self.fails = self.fails, 0
            res = self._guided_step(stamp, images, dict(prior, sig_xy=min(prior['sig_xy'], 0.99 * LOST_SIGMA)))
            self.fails = 0 if res is not None else fails + 1
            return res
        # global relocalization, retrieval only near the prior (a night-trained retrieval must not jump to a look-alike)
        self._last_global = stamp
        saved = self.allowed_base
        near = np.hypot(*(self.kf_xy - [prior['x'], prior['y']]).T) < max(GLOBAL_RADIUS_MIN, 4 * prior['sig_xy'])
        if (saved & near).any():
            self.allowed_base = saved & near
        t0 = time.perf_counter()
        try:
            res = self._global(stamp, images)
        finally:
            self.allowed_base = saved
        self.n_global += 1
        self.global_ms.append((time.perf_counter() - t0) * 1e3)
        self.fails = 0 if res is not None else self.fails
        if res is not None and not self.drivable.inside(res[0], res[1]):
            self.n_bounds_rej += 1
            return None
        return res

    @torch.no_grad()
    def _global(self, stamp, images):
        """DINOv2 retrieval of the best allowed keyframe + LightGlue + PnP-RANSAC -> (x, y, yaw, inliers) or None."""
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
        qf = {c: self.fe.extract(_CLAHE.apply(cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY)), self.cams[c].mask)
              for c in names}
        out = self.reloc.solve(self.reloc.correspondences(qf, cand))
        self.reloc_ms.append((time.perf_counter() - t0) * 1e3)
        if out is None:
            return None
        T, n_inl, _ = out
        return float(T[0, 3]), float(T[1, 3]), float(yaw_of(T)), int(n_inl)

    @torch.no_grad()
    def _guided_step(self, stamp, images, prior):
        t0 = time.perf_counter()
        half = {c: cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for c, im in images.items() if im is not None}
        res = self._guided(stamp, self._features(half), prior)
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

    # ------------------------------------------------------------ guided correction
    @torch.no_grad()
    def _features(self, half):
        """exposure normalisation + saturation mask + SuperPoint (FP16, one batch) -> {cam: dict(kps, desc)};
        keeps the dense descriptor maps (self._dense) and usable pixels (self._ok) for the dense search."""
        names = []
        g = np.zeros((len(CAMS), H, W), np.uint8)                  # fixed batch shape for the compiled graph
        for c in CAMS:
            if c not in half:
                continue
            raw = cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY)
            mask = self.cams[c].mask
            ok = mask > 0
            sat = cv2.dilate((raw >= SAT_GREY).astype(np.uint8), np.ones((SAT_DILATE, SAT_DILATE), np.uint8)) > 0
            if (sat & ok).sum() > SAT_CAM_FRAC * ok.sum():       # a saturated sky alone is normal by day
                self.n_cam_skip += 1
                continue
            ok = ok & ~sat
            med = float(np.median(raw[mask > 0]))
            if 5 < med < 250:
                gm = float(np.clip(np.log(self._target[c] / 255) / np.log(med / 255), GAMMA_MIN, GAMMA_MAX))
                if abs(gm - 1) > 0.05:
                    raw = cv2.LUT(raw, np.clip(255 * (np.arange(256) / 255) ** gm, 0, 255).astype(np.uint8))
                self.gammas.append(gm)
            self._ok[c] = ok.astype(np.uint8) * 255
            g[len(names)] = _CLAHE.apply(raw)
            names.append(c)
        if not names:
            self._dense = {}
            return {}
        sp = self.sp16
        x = torch.from_numpy(g).cuda().half().div_(255)[:, None].contiguous(memory_format=torch.channels_last)
        logits, dense = self.backbone(x)
        n = len(names)
        sc = F.softmax(logits[:n].float(), 1)[:, :-1]
        b, _, h, w = sc.shape
        sc = sc.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        sc = simple_nms(sc, sp.conf.nms_radius)
        p = sp.conf.remove_borders
        if p:
            sc[:, :p] = -1; sc[:, :, :p] = -1; sc[:, -p:] = -1; sc[:, :, -p:] = -1
        dense = F.normalize(dense[:n].float(), p=2, dim=1)
        self._dense = {c: dense[i:i + 1] for i, c in enumerate(names)}
        out = {}
        for i, c in enumerate(names):
            ij = torch.nonzero(sc[i] > sp.conf.detection_threshold)
            s = sc[i][ij[:, 0], ij[:, 1]]
            ij, s = top_k_keypoints(ij, s, MAX_KP)
            k = torch.flip(ij, [1]).float()
            d = sample_descriptors(k[None], dense[i:i + 1], 8)[0].T
            k, d = k.cpu().numpy(), d.cpu().numpy()
            ki = np.round(k).astype(int)
            keep = self._ok[c][ki[:, 1].clip(0, H - 1), ki[:, 0].clip(0, W - 1)] > 0
            out[c] = dict(kps=k[keep].astype(np.float32), desc=d[keep].astype(np.float16))
        return out

    def _prior_from(self, x, y, yaw, P):
        return dict(x=float(x), y=float(y), yaw=float(yaw), sig_xy=float(np.sqrt(max(P[0, 0], P[1, 1]))),
                    sig_yaw=float(np.sqrt(P[2, 2])))

    def _nearest_kfs(self, prior, stamp):
        """keyframes near the prior (distance, heading), nearest first but at most one per pass."""
        allowed = self.allowed_base.copy()
        if self.self_excl_ns > 0:
            allowed &= np.abs(self.kf - stamp) > self.self_excl_ns
        d = np.hypot(*(self.kf_xy - [prior['x'], prior['y']]).T)
        ok = allowed & (d < KF_MAX_DIST) & (np.abs(wrap(self.kf_yaw - prior['yaw'])) < KF_MAX_DYAW)
        idx = np.where(ok)[0]
        out = []
        for i in idx[np.argsort(d[idx])]:
            if all(abs(int(self.kf[i]) - int(self.kf[j])) > PASS_GAP_S * 1e9 for j in out):
                out.append(i)
                if len(out) == N_KF:
                    break
        return np.array(out, int)

    def _T_prior(self, prior, kf_i):
        """prior x, y, yaw with z / roll / pitch of the nearest keyframe."""
        T = self.kf_T[kf_i].copy()
        R_rel = Rotation.from_euler('z', prior['yaw'] - self.kf_yaw[kf_i]).as_matrix()
        T[:3, :3] = R_rel @ T[:3, :3]
        T[0, 3], T[1, 3] = prior['x'], prior['y']
        return T

    def _kf_desc(self, c, i):
        """day descriptors of keyframe i, camera c (aligned with the night ones of the map)."""
        d = self.db.db[c]
        a, b = d['off'][i], d['off'][i + 1]
        return np.asarray(self._day[c][a:b])

    def _guided(self, stamp, qf, prior):
        res = self._match(stamp, qf, prior, SIM_MIN, DENSE_SIM)
        if res is None and qf:                   # descriptors drift with the light: retry once, lower thresholds
            self.n_relax += 1
            res = self._match(stamp, qf, prior, RELAX_SIM_MIN, RELAX_DENSE_SIM)
            self.n_relax_ok += res is not None
        if res is not None and not self.drivable.inside(res[0], res[1]):
            self.n_bounds_rej += 1
            return None
        return res

    @torch.no_grad()
    def _match(self, stamp, qf, prior, sim_min, dense_sim):
        """map points projected with the prior, matched in a window (keypoints, then dense search for the rest),
        pose refined on all cameras -> (x, y, yaw, inliers) or None."""
        kfs = self._nearest_kfs(prior, stamp)
        if not len(kfs):
            return None
        T0 = self._T_prior(prior, kfs[0])
        per, n_kp = {}, 0
        for c, f in qf.items():
            pts = [self.db.kf_points(c, i) for i in kfs]
            X = np.concatenate([q[1] for q in pts])
            Dn = np.concatenate([q[0]['desc'] for q in pts])
            Dd = np.concatenate([self._kf_desc(c, i) for i in kfs]) if self._day else None
            if len(X) < 8:
                continue
            cam = self.cams[c]
            Tc = T0 @ cam.T_base_cam
            pc = (X - Tc[:3, 3]) @ Tc[:3, :3]
            uv = cam.project_cam(pc)
            vis = np.isfinite(uv[:, 0])
            if vis.sum() < 8:
                continue
            X, Dn, uv, depth = X[vis], Dn[vis], uv[vis], pc[vis, 2]
            Dd = None if Dd is None else Dd[vis]
            f_px = cam.K[0, 0]
            win = np.clip(f_px * 2.5 * prior['sig_xy'] / depth + f_px * 2.5 * prior['sig_yaw'] + 3, WIN_MIN, WIN_MAX)
            used = np.zeros(len(X), bool)
            kp_uv, kp_X = np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32)
            if len(f['kps']) >= 8:
                uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
                k_t = torch.from_numpy(f['kps']).cuda()
                inwin = torch.cdist(uv_t, k_t) < torch.from_numpy(win.astype(np.float32)).cuda()[:, None]
                q = torch.from_numpy(f['desc']).cuda().float().T
                Sn = torch.from_numpy(Dn).cuda().float() @ q
                Sd = None if Dd is None else torch.from_numpy(Dd).cuda().float() @ q
                S = Sn if Sd is None else torch.maximum(Sn, Sd)
                S[~inwin] = -1
                bj = S.argmax(1); bi = S.argmax(0)       # best keypoint per map point, best map point per keypoint
                i = torch.arange(len(X), device='cuda')
                ok = ((S[i, bj] > sim_min) & (bi[bj] == i))
                if Sd is not None and ok.any():
                    self.n_day_better.append(float((Sd[i, bj] > Sn[i, bj])[ok].float().mean()))
                ok = ok.cpu().numpy()
                bj = bj.cpu().numpy()
                if ok.sum() >= 4:
                    used = ok
                    kp_uv, kp_X = f['kps'][bj[ok]], X[ok]
            n_kp += len(kp_uv)
            per[c] = (X, Dn, Dd, uv, win, used, kp_uv, kp_X)
        corr, n_dense = {}, 0
        for c, (X, Dn, Dd, uv, win, used, kp_uv, kp_X) in per.items():
            U, XX = [kp_uv], [kp_X]
            if (~used).any():
                rest = np.where(~used)[0]
                fu, j = self._dense_match(c, uv[rest], Dn[rest], None if Dd is None else Dd[rest], win[rest], dense_sim)
                U.append(fu); XX.append(X[rest[j]]); n_dense += len(fu)
            U, XX = np.concatenate(U), np.concatenate(XX)
            if len(U) >= 4:
                corr[c] = (self.cams[c].undistort_points(U), XX)
        self.n_kp_corr.append(n_kp); self.n_dense_corr.append(n_dense)
        if sum(len(v[0]) for v in corr.values()) < MIN_INLIERS_GUIDED:
            return None
        Tb = T0
        for thr in (4 * THR_PX, THR_PX):
            r = self.reloc.residuals(Tb, corr)
            sel = {c: e < thr for c, e in r.items()}
            if sum(s.sum() for s in sel.values()) < MIN_INLIERS_GUIDED:
                return None
            Tb = self._refine(Tb, corr, sel)
        r = self.reloc.residuals(Tb, corr)
        n_inl = int(sum((e < THR_PX).sum() for e in r.values()))
        if n_inl < MIN_INLIERS_GUIDED:
            return None
        return float(Tb[0, 3]), float(Tb[1, 3]), float(yaw_of(Tb)), n_inl

    @torch.no_grad()
    def _dense_match(self, c, uv, Dn, Dd, win, dense_sim):
        """dense guided search: for each projected map point (uv (n,2), night / day descriptors Dn / Dd (n,256) or
        Dd None, window win (n,) px) the dense descriptor map is sampled on a 1 px grid inside the window (radius
        <= DENSE_R) and the best position is taken, sub-pixel by a parabola; accepted when similar enough and distinct
        from the rest of the window. -> (uv_found (m,2), index (m,))"""
        dm = self._dense.get(c)
        if dm is None or not len(uv):
            return np.zeros((0, 2), np.float32), np.zeros(0, int)
        G_T = self.G_T
        n, g, R = len(uv), len(_G), DENSE_R
        uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
        r = torch.from_numpy(np.clip(win, 3, R).astype(np.float32)).cuda()
        pos = uv_t[:, None, :] + G_T[None]                                          # (n, g, 2)
        inside = (pos[..., 0] >= 4) & (pos[..., 0] <= W - 5) & (pos[..., 1] >= 4) & (pos[..., 1] <= H - 5)
        inwin = inside & (G_T.norm(dim=1)[None] <= r[:, None])
        desc = sample_descriptors(pos.reshape(1, -1, 2), dm, 8)[0].reshape(-1, n, g)
        sim = torch.einsum('nc,cng->ng', torch.from_numpy(Dn).cuda().float(), desc)
        if Dd is not None:
            sim = torch.maximum(sim, torch.einsum('nc,cng->ng', torch.from_numpy(Dd).cuda().float(), desc))
        sim[~inwin] = -2
        best, bi = sim.max(1)
        far = (G_T[None] - G_T[bi][:, None]).norm(dim=2) > 2.5
        second = torch.where(far, sim, torch.full_like(sim, -2)).max(1).values
        ok = (best > dense_sim) & (best - second > DENSE_MARGIN)
        S = 2 * R + 1
        by, bx = bi // S, bi % S
        sim3 = sim.reshape(n, S, S)
        ar = torch.arange(n, device='cuda')

        def para(m, c0, p):
            den = m - 2 * c0 + p
            return torch.where((den < -1e-6), 0.5 * (m - p) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        cx = (bx > 0) & (bx < S - 1); cy = (by > 0) & (by < S - 1)
        c0 = sim3[ar, by, bx]
        dx = torch.where(cx, para(sim3[ar, by, (bx - 1).clamp(0)], c0, sim3[ar, by, (bx + 1).clamp(max=S - 1)]), 0 * c0)
        dy = torch.where(cy, para(sim3[ar, (by - 1).clamp(0), bx], c0, sim3[ar, (by + 1).clamp(max=S - 1), bx]), 0 * c0)
        found = pos[ar, bi] + torch.stack([dx, dy], 1)
        ok, found = ok.cpu().numpy(), found.cpu().numpy()
        fi = np.round(found).astype(int)
        ok &= self.cams[c].mask[fi[:, 1].clip(0, H - 1), fi[:, 0].clip(0, W - 1)] > 0
        ok &= self._ok[c][fi[:, 1].clip(0, H - 1), fi[:, 0].clip(0, W - 1)] > 0
        return found[ok].astype(np.float32), np.where(ok)[0]

    def _refine(self, Tb, corr, sel, iters=8):
        """robust Gauss-Newton on T_map_base, analytic Jacobian, Huber weights (pixel residuals, all cameras).
        Perturbation: R <- exp([dth]x) R, t <- t + dt (map frame)."""
        n_l, X_l, f_l, tbc_l, Rbc_l = [], [], [], [], []
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

    # ------------------------------------------------------------ start without a first pose
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
        r = self._global(t, images)
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
            return self._track(stamp_ns, images, car_state)    # the filter starts here
        if v is not None:
            self.guess = (float(v[0][0]), float(v[0][1]), self.guess[2])
        return self.guess

    # ------------------------------------------------------------ statistics
    def summary(self):
        mean = lambda v: round(float(np.mean(v)), 1) if v else None
        p95 = lambda v: round(float(np.percentile(v, 95)), 1) if v else None
        med = lambda v: float(np.median(v)) if v else None
        f = self.f
        return dict(
            car_state=self.use_state, day_layer=DAY, init_s=round(self.init_s, 1),
            map_keyframes_allowed=self.n_allowed,
            relocalizations=len(self.reloc_ms), reloc_ms_mean=mean(self.reloc_ms), reloc_ms_p95=p95(self.reloc_ms),
            measurements=self.n_meas, failed=self.n_fail,
            accepted=f and f.n_upd, rejected=f and f.n_rej, recovered=f and f.n_recover,
            guided=self.n_guided, guided_fail=self.n_guided_fail, guided_ms_mean=mean(self.guided_ms),
            guided_ms_p95=p95(self.guided_ms), guided_inliers_median=med(self.guided_inl),
            kp_corr_median=med(self.n_kp_corr), dense_corr_median=med(self.n_dense_corr),
            relax=self.n_relax, relax_ok=self.n_relax_ok, bounds_rejected=self.n_bounds_rej,
            cams_skipped_saturated=self.n_cam_skip,
            gamma_median=med(self.gammas),
            gamma_p5=round(float(np.percentile(self.gammas, 5)), 3) if self.gammas else None,
            day_better_share_median=med(self.n_day_better),
            global_=self.n_global, global_ms_mean=mean(self.global_ms), global_skipped_to_guided=self.n_global_skip,
            boot=dict(self.boot_info, not_initialised=self.booting) if self.booting else self.boot_info)
