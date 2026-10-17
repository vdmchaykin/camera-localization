"""Synthetic daylight / sun stress test: wraps any localizer and perturbs its camera images before step().

    STRESS_BASE=v5/my_loc_v5.py:Localizer STRESS=day,shadow,glare \
    python3 sim/simulate.py ... --localizer v6/sun_stress.py:Localizer

There is no daylight data, so this does not predict day accuracy; it checks how the pipeline reacts to what sun
changes: (day) brighter, lower-contrast scene, lamps no longer bright blobs, bright sky behind dark silhouettes
(edge polarity flips); (shadow) hard cast shadows on the lower image that move every 2 s; (glare) the sun at a
fixed map azimuth / elevation: saturated disk + veiling glare in cameras that look towards it, depending on the
car heading; (overexp) exposure x2.2 with clipping; (relight) smooth random gain field = light from another direction,
the harshest proxy for the night->day change of the descriptors. The perturbation runs at half resolution (the localizers work
there) and its time is reported separately (stress_ms) — subtract it from step_ms.
Env: STRESS (comma list, default 'day,shadow,glare'), SUN_AZ_DEG (default 120), SUN_EL_DEG (default 20).
"""
import importlib.util, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from camera_model import CAMS, H, W, Camera, tf_mat  # noqa: E402


def _load(spec):
    path, cls = spec.rsplit(':', 1)
    path = os.path.join(ROOT, path) if not os.path.isabs(path) else path
    sys.path.insert(0, os.path.dirname(path))
    m = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    return getattr(mod, cls)


Base = _load(os.environ.get('STRESS_BASE', 'v5/my_loc_v5.py:Localizer'))
MODES = set(filter(None, os.environ.get('STRESS', 'day,shadow,glare').split(',')))
AZ, EL = np.radians(float(os.environ.get('SUN_AZ_DEG', '120'))), np.radians(float(os.environ.get('SUN_EL_DEG', '20')))
SUN_MAP = np.array([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)])


def day(h):
    g = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
    lamp = cv2.dilate((g > 0.85).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if lamp.any():                                     # lamps / lit blobs: replaced by their surroundings
        fill = cv2.blur(np.where(lamp[..., None], 0, h), (31, 31)) / np.maximum(cv2.blur((~lamp).astype(np.float32), (31, 31)), 1e-3)[..., None]
        h = np.where(lamp[..., None], fill, h)
    sky = np.zeros_like(g, bool)                       # dark upper region -> bright sky (silhouette polarity flips)
    top = int(0.45 * H)
    sky[:top] = cv2.blur(g[:top], (15, 15)) < 0.08
    h = 0.05 + 0.95 * np.power(np.clip(h, 0, 1), 0.6)   # ambient light: brighter, dark areas lifted
    h = np.clip((h - 0.5) * 1.25 + 0.55, 0, 1)        # direct sun: contrast back up
    h = h * np.array([0.92, 1.0, 1.05], np.float32)   # warmer
    grad = np.linspace(0.92, 0.78, H, dtype=np.float32)[:, None, None] * np.array([1.0, 0.95, 0.85], np.float32)
    return np.where(sky[..., None], grad, h)


def shadow(h, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.float32)
    for _ in range(rng.integers(2, 4)):
        cx, cy = rng.uniform(0, W), rng.uniform(0.6 * H, H)
        a = rng.uniform(0, np.pi); L, Wd = rng.uniform(40, 200), rng.uniform(15, 70)
        c, s = np.cos(a), np.sin(a)
        poly = np.array([[cx + c * L - s * Wd, cy + s * L + c * Wd], [cx - c * L - s * Wd, cy - s * L + c * Wd],
                         [cx - c * L + s * Wd, cy - s * L - c * Wd], [cx + c * L + s * Wd, cy + s * L - c * Wd]])
        cv2.fillPoly(m, [poly.astype(np.int32)], 1.0)
    m = cv2.GaussianBlur(m, (5, 5), 0) * np.clip((np.arange(H) / H - 0.45) / 0.15, 0, 1)[:, None]   # on the ground only
    return h * (1 - 0.6 * m[..., None])


def relight(h, seed):
    """light from another direction: smooth random gain field (x0.4..x2.5, ~20 px scale) changes the relative
    brightness of neighbouring surfaces, i.e. the gradients the descriptors see."""
    rng = np.random.default_rng(seed + 1000)
    f = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 20)
    f = f / (f.std() + 1e-6) * 0.45
    return h * np.exp(np.clip(f, -0.9, 0.9))[..., None]


def glare(h, cam, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    d_base = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ SUN_MAP
    d = cam.T_base_cam[:3, :3].T @ d_base                 # camera optical frame
    ang = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
    if ang > 85:
        return h
    if d[2] <= 0.05:
        return h * 0.9 + 0.1
    u, v = cam.K[0, 0] * d[0] / d[2] + cam.K[0, 2], cam.K[1, 1] * d[1] / d[2] + cam.K[1, 2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = (xx - u) ** 2 + (yy - v) ** 2
    veil = 0.65 * np.exp(-r2 / (2 * (0.35 * W) ** 2)) * max(0.3, np.cos(np.radians(ang)))
    h = h * (1 - 0.5 * veil[..., None]) + veil[..., None]
    h[r2 < 28 ** 2] = 1.0                                 # sun disk, saturated
    if 0 <= u < W:                                        # vertical smear / flare streak
        x0 = int(np.clip(u, 0, W - 1))
        h[:, max(0, x0 - 2):x0 + 3] = np.maximum(h[:, max(0, x0 - 2):x0 + 3], 0.95)
    return h


class Localizer(Base):
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        super().__init__(first_pose, camera_info, tf_static, track_bounds)
        self._sc = {}
        for c in CAMS:
            t = tf_static[f'constructor0/base_link -> constructor0/camera_{c}']
            r, p = t['rotation'], t['translation']
            self._sc[c] = Camera(c, camera_info[c], tf_mat([r['x'], r['y'], r['z'], r['w']], [p['x'], p['y'], p['z']]))
        self._yaw0 = first_pose['yaw']
        self.stress_ms = []
        self._pool = ThreadPoolExecutor(len(CAMS))

    def _perturb(self, c, im, stamp, yaw):
        h = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if 'day' in MODES:
            h = day(h)
        if 'overexp' in MODES:
            h = h * 2.2
        if 'relight' in MODES:
            h = relight(h, int(stamp // 2_000_000_000) * 7 + CAMS.index(c))
        if 'shadow' in MODES:
            h = shadow(h, int(stamp // 2_000_000_000) * 7 + CAMS.index(c))
        if 'glare' in MODES:
            h = glare(h, self._sc[c], yaw)
        h = (np.clip(h, 0, 1) * 255).astype(np.uint8)
        return cv2.resize(h, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)

    def step(self, stamp_ns, images, car_state):
        t0 = time.perf_counter()
        yaw = float(self.f.X[2]) if getattr(self, 'f', None) is not None else self._yaw0   # sun placed by own heading
        if MODES:
            keys = list(images)
            outs = self._pool.map(lambda c: self._perturb(c, images[c], stamp_ns, yaw) if images[c] is not None else None, keys)
            images = dict(zip(keys, outs))
        self.stress_ms.append((time.perf_counter() - t0) * 1e3)
        return super().step(stamp_ns, images, car_state)

    def summary(self):
        s = super().summary() if hasattr(super(), 'summary') else {}
        s['stress'] = dict(modes=sorted(MODES), base=os.environ.get('STRESS_BASE', 'v5/my_loc_v5.py:Localizer'),
                           sun_az_deg=round(float(np.degrees(AZ)), 1), sun_el_deg=round(float(np.degrees(EL)), 1),
                           stress_ms_mean=round(float(np.mean(self.stress_ms)), 1) if self.stress_ms else 0)
        return s
