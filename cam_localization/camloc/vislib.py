"""Local features (SuperPoint + LightGlue) on half-res frames, for the global relocalization."""
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
from lightglue import LightGlue, SuperPoint  # noqa: E402

from .camera_model import H, W  # noqa: E402

DEV = 'cuda'


class Features:
    def __init__(self, max_kp=1024, thr=0.0002):
        self.sp = SuperPoint(max_num_keypoints=max_kp, detection_threshold=thr).eval().to(DEV)
        self.lg = LightGlue(features='superpoint').eval().to(DEV)

    @torch.no_grad()
    def extract(self, gray, mask=None):
        """-> dict(kps (N,2) float32 np, desc (N,256) float16 np)."""
        x = torch.from_numpy(gray).float().div(255)[None, None].to(DEV)
        f = self.sp.extract(x)
        k = f['keypoints'][0].cpu().numpy()
        d = f['descriptors'][0].cpu().numpy()
        if mask is not None:
            ki = np.round(k).astype(int)
            keep = mask[ki[:, 1].clip(0, H - 1), ki[:, 0].clip(0, W - 1)] > 0
            k, d = k[keep], d[keep]
        return dict(kps=k.astype(np.float32), desc=d.astype(np.float16))

    @torch.no_grad()
    def match(self, f0, f1, min_score=0.2):
        """-> (M,2) int index pairs into f0, f1."""
        if len(f0['kps']) < 8 or len(f1['kps']) < 8:
            return np.zeros((0, 2), int)
        size = torch.tensor([[W, H]], device=DEV, dtype=torch.float32)

        def t(f):
            return dict(keypoints=torch.from_numpy(f['kps'])[None].to(DEV),
                        descriptors=torch.from_numpy(f['desc'].astype(np.float32))[None].to(DEV),
                        image_size=size)
        r = self.lg({'image0': t(f0), 'image1': t(f1)})
        m = r['matches'][0].cpu().numpy()
        sc = r['scores'][0].cpu().numpy()
        return m[sc > min_score]
