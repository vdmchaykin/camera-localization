"""v6 localizer: v5 prepared for daylight / sun (the map is from one night session; the hidden test may differ).

Same interface as camloc/base.py ... camloc/dense.py (inherited, none of them is modified):
    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

What a sunny run changes, and the answer here (each switchable by env var, default = on):
  V6_EXPO=1    much brighter images, other exposure: per camera, a gamma maps the median raw grey (inside the body
               mask) to the median of the night keyframes (model/night_stats.json) before CLAHE. At night
               gamma ~ 1, so the night behaviour does not change.
  V6_SAT=1     sun in the image, glare, specular reflections: saturated pixels (raw grey >= 250, dilated) are
               masked for keypoints and dense matches; a camera with > SAT_CAM_FRAC of saturated pixels is skipped (a saturated sky alone is normal by day).
  V6_RELAX=1   fewer / weaker matches (descriptors drift with the light): when a guided correction fails, it is
               retried once with dense search on and lower similarity thresholds (the prior window keeps the
               geometry tight).
  V6_BOUNDS=1  geometry that does not depend on light: a correction is rejected when the pose is off the drivable
               area (track + pit lane from /track_bounds, +BOUNDS_MARGIN m).
  V6_GLOBAL=1  the global relocalization (v1 path, ~220 ms) runs only after LOST_FAILS6 guided failures in a row
               (v2: 3), at most every GLOBAL_MIN_S, and retrieves only keyframes near the prior (radius from its
               sigma), so a night-trained retrieval cannot jump to a look-alike straight.
"""
import json, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints
from . import dense as v5, guided as v2
from .camera_model import CAMS, H, W
from .paths import MODEL

ENV = lambda k, d='1': os.environ.get(k, d) == '1'
EXPO, SAT, RELAX, BOUNDS, GLOBAL = ENV('V6_EXPO'), ENV('V6_SAT'), ENV('V6_RELAX'), ENV('V6_BOUNDS'), ENV('V6_GLOBAL')
SAT_GREY, SAT_DILATE, SAT_CAM_FRAC = 250, 11, 0.60
GAMMA_MIN, GAMMA_MAX = 0.35, 2.5
RELAX_SIM_MIN, RELAX_DENSE_SIM = 0.60, 0.65
BOUNDS_RES, BOUNDS_MARGIN = 0.25, 1.5     # m
LOST_FAILS6 = 10                          # guided failures in a row (1 s at 10 Hz) before the global path
GLOBAL_MIN_S = 1.0
GLOBAL_RADIUS_MIN = 30.0                  # m


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


class Localizer(v5.Localizer):
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        st = json.load(open(os.path.join(MODEL, 'night_stats.json')))
        self._target = {c: st[c]['median_grey'] for c in CAMS}
        self.drivable = Drivable(track_bounds)
        self._ok = {}
        self.n_relax = self.n_relax_ok = self.n_bounds_rej = self.n_global_skip = self.n_cam_skip = 0
        self.gammas = []
        self._last_global = None
        super().__init__(first_pose, camera_info, tf_static, track_bounds)
        self.n_relax = self.n_relax_ok = self.n_bounds_rej = self.n_global_skip = self.n_cam_skip = 0
        self.gammas.clear()
        if GLOBAL:
            v2.LOST_FAILS = LOST_FAILS6

    # ------------------------------------------------------------ features: exposure normalisation + saturation mask
    @torch.no_grad()
    def _features(self, half):
        names = []
        g = np.zeros((len(CAMS), H, W), np.uint8)
        for c in CAMS:
            if c not in half:
                continue
            raw = cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY)
            mask = self.cams[c].mask
            ok = mask > 0
            if SAT:
                sat = cv2.dilate((raw >= SAT_GREY).astype(np.uint8), np.ones((SAT_DILATE, SAT_DILATE), np.uint8)) > 0
                if (sat & ok).sum() > SAT_CAM_FRAC * ok.sum():
                    self.n_cam_skip += 1
                    continue
                ok = ok & ~sat
            if EXPO:
                med = float(np.median(raw[mask > 0]))
                if 5 < med < 250:
                    gm = float(np.clip(np.log(self._target[c] / 255) / np.log(med / 255), GAMMA_MIN, GAMMA_MAX))
                    if abs(gm - 1) > 0.05:
                        raw = cv2.LUT(raw, np.clip(255 * (np.arange(256) / 255) ** gm, 0, 255).astype(np.uint8))
                    self.gammas.append(gm)
            self._ok[c] = ok.astype(np.uint8) * 255
            g[len(names)] = v2.v1._CLAHE.apply(raw)
            names.append(c)
        if not names:
            self._dense = {}
            return {}
        sp, bb = self._fast_sp()
        x = torch.from_numpy(g).cuda().half().div_(255)[:, None].contiguous(memory_format=torch.channels_last)
        logits, dense = bb(x)
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
            ij, s = top_k_keypoints(ij, s, v5.MAX_KP)
            k = torch.flip(ij, [1]).float()
            d = sample_descriptors(k[None], dense[i:i + 1], 8)[0].T
            k, s, d = k.cpu().numpy(), s.cpu().numpy(), d.cpu().numpy()
            ki = np.round(k).astype(int)
            keep = self._ok[c][ki[:, 1].clip(0, H - 1), ki[:, 0].clip(0, W - 1)] > 0
            out[c] = dict(kps=k[keep].astype(np.float32), scores=s[keep].astype(np.float32),
                          desc=d[keep].astype(np.float16))
        return out

    def _dense_match(self, c, uv, D, win):
        found, idx = super()._dense_match(c, uv, D, win)
        if SAT and c in self._ok and len(found):
            fi = np.round(found).astype(int)
            keep = self._ok[c][fi[:, 1].clip(0, H - 1), fi[:, 0].clip(0, W - 1)] > 0
            found, idx = found[keep], idx[keep]
        return found, idx

    # ------------------------------------------------------------ guided: relaxed retry + drivable-area check
    def _guided(self, stamp, qf, prior):
        res = super()._guided(stamp, qf, prior)
        if res is None and RELAX and qf:
            self.n_relax += 1
            saved = v2.SIM_MIN, v5.DENSE_SIM, v5.DENSE
            v2.SIM_MIN, v5.DENSE_SIM, v5.DENSE = RELAX_SIM_MIN, RELAX_DENSE_SIM, 1
            try:
                res = super()._guided(stamp, qf, prior)
            finally:
                v2.SIM_MIN, v5.DENSE_SIM, v5.DENSE = saved
            self.n_relax_ok += res is not None
        if res is not None and BOUNDS and not self.drivable.inside(res[0], res[1]):
            self.n_bounds_rej += 1
            return None
        return res

    # ------------------------------------------------------------ global path: rate-limited, near the prior
    def _relocalize(self, stamp, images, prior=None):
        ready = getattr(self, '_v2_ready', False)
        goes_global = not ready or not v2.GUIDED or prior is None or prior['sig_xy'] > v2.LOST_SIGMA \
            or self.fails >= v2.LOST_FAILS
        if not (GLOBAL and ready and goes_global and prior is not None):
            return super()._relocalize(stamp, images, prior)
        if self._last_global is not None and stamp - self._last_global < GLOBAL_MIN_S * 1e9:
            # too soon for another global attempt: keep trying the guided path from the (wider) prior instead
            self.n_global_skip += 1
            fails, self.fails = self.fails, 0
            p = dict(prior, sig_xy=min(prior['sig_xy'], 0.99 * v2.LOST_SIGMA))
            res = super()._relocalize(stamp, images, p)
            self.fails = 0 if res is not None else fails + 1
            return res
        self._last_global = stamp
        saved = self.allowed_base
        near = np.hypot(*(self.kf_xy - [prior['x'], prior['y']]).T) < max(GLOBAL_RADIUS_MIN, 4 * prior['sig_xy'])
        if (saved & near).any():
            self.allowed_base = saved & near
        try:
            res = super()._relocalize(stamp, images, prior)
        finally:
            self.allowed_base = saved
        if res is not None and BOUNDS and not self.drivable.inside(res[0], res[1]):
            self.n_bounds_rej += 1
            return None
        return res

    def summary(self):
        s = super().summary()
        g = np.array(self.gammas) if self.gammas else np.array([1.0])
        s.update(v6=dict(expo=EXPO, sat=SAT, relax_on=RELAX, bounds=BOUNDS, global_=GLOBAL,
                         gamma_median=round(float(np.median(g)), 3), gamma_p5=round(float(np.percentile(g, 5)), 3),
                         relax=self.n_relax, relax_ok=self.n_relax_ok, bounds_rejected=self.n_bounds_rej,
                         global_skipped_to_guided=self.n_global_skip, cams_skipped_saturated=self.n_cam_skip))
        return s
