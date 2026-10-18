"""v3 localizer: v2 with a faster SuperPoint, so corrections can run more often.

Same interface as camloc/base.py and camloc/guided.py (inherited, neither is modified):
    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

Changes vs v2, each switchable by env var for ablations (default = on):
  V3_FAST_SP=1   own SuperPoint pass for the guided path: backbone + heads in FP16 / channels_last, softmax, NMS and
                 keypoint selection in FP32, keypoints selected per camera (no torch.stack). v2's batched call fails
                 whenever one camera has fewer than MAX_KP keypoints above the threshold (~40 % of the sets on the
                 development laps) and then runs SuperPoint again camera by camera.
  V3_COMPILE=1   torch.compile of the backbone (fixed batch of 7, missing cameras padded with zeros; ~2.5 s at init).
  V3_PERIOD      correction period on straights, s (default 0 = back to back, as in corners; v2: 0.3).
The global fallback (v1 path: DINOv2 + LightGlue + PnP-RANSAC) is unchanged.
"""
import os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints
from . import guided as v2
from .camera_model import CAMS, H, W

ENV = lambda k, d='1': os.environ.get(k, d) == '1'
FAST_SP, COMPILE = ENV('V3_FAST_SP'), ENV('V3_COMPILE')
v2.STRAIGHT_PERIOD_S = float(os.environ.get('V3_PERIOD', '0'))     # read by v2 step() at call time


def _backbone(s, x):
    """SuperPoint encoder + heads: -> (score logits (b, 65, h/8, w/8), dense descriptors (b, 256, h/8, w/8))."""
    y = s.relu(s.conv1a(x)); y = s.relu(s.conv1b(y)); y = s.pool(y)
    y = s.relu(s.conv2a(y)); y = s.relu(s.conv2b(y)); y = s.pool(y)
    y = s.relu(s.conv3a(y)); y = s.relu(s.conv3b(y)); y = s.pool(y)
    y = s.relu(s.conv4a(y)); y = s.relu(s.conv4b(y))
    return s.convPb(s.relu(s.convPa(y))), s.convDb(s.relu(s.convDa(y)))


class Localizer(v2.Localizer):
    def _fast_sp(self):
        """FP16 channels_last copy of the SuperPoint weights (+ compiled backbone), built on first use."""
        if getattr(self, '_sp16', None) is None:
            sp = self.reloc.fe.sp
            self._sp16 = type(sp)(**{k: v for k, v in vars(sp.conf).items()}).eval().cuda()
            self._sp16.load_state_dict(sp.state_dict())
            self._sp16 = self._sp16.half().to(memory_format=torch.channels_last)
            fn = lambda x: _backbone(self._sp16, x)
            self._bb = torch.compile(fn, dynamic=False) if COMPILE else fn
        return self._sp16, self._bb

    @torch.no_grad()
    def _features(self, half):
        if not FAST_SP:
            return super()._features(half)
        names = [c for c in CAMS if c in half]
        g = np.zeros((len(CAMS), H, W), np.uint8)                  # fixed batch shape for the compiled graph
        for i, c in enumerate(names):
            g[i] = v2.v1._CLAHE.apply(cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY))
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
        out = {}
        for i, c in enumerate(names):
            ij = torch.nonzero(sc[i] > sp.conf.detection_threshold)
            s = sc[i][ij[:, 0], ij[:, 1]]
            if sp.conf.max_num_keypoints is not None:
                ij, s = top_k_keypoints(ij, s, sp.conf.max_num_keypoints)
            k = torch.flip(ij, [1]).float()
            d = sample_descriptors(k[None], dense[i:i + 1], 8)[0].T
            k, s, d = k.cpu().numpy(), s.cpu().numpy(), d.cpu().numpy()
            ki = np.round(k).astype(int)
            keep = self.cams[c].mask[ki[:, 1].clip(0, H - 1), ki[:, 0].clip(0, W - 1)] > 0
            out[c] = dict(kps=k[keep].astype(np.float32), scores=s[keep].astype(np.float32),
                          desc=d[keep].astype(np.float16))
        return out

    def summary(self):
        s = super().summary()
        s.update(v3=dict(fast_sp=FAST_SP, compile=COMPILE, straight_period_s=v2.STRAIGHT_PERIOD_S))
        return s
