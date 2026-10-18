#!/usr/bin/env python3
"""DINOv2 ViT-S/14 global descriptors for every frame of every camera.

    python3 global_desc.py      -> features/global_<cam>.npy  (N_sets, 768) float16, L2-normalised,
                                   row order = camera_model.frame_stamps(); zero row if frame missing

Descriptor = [CLS ; GeM(patch tokens)], each half L2-normalised. Body mask regions are filled with grey.
"""
import os, time, warnings
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from .camera_model import CAMS, frame_path, frame_stamps, load_cameras
from .paths import FEAT

warnings.filterwarnings('ignore')
OUT = FEAT
IN_W, IN_H = 462, 224        # multiples of 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BATCH = 32


def prep(path, mask):
    if path is None:
        return None
    im = cv2.imread(path)
    im[mask == 0] = 127
    im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
    return ((im - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def describe(model, x):
    out = model.forward_features(x)
    cls = out['x_norm_clstoken']
    pt = out['x_norm_patchtokens']
    gem = pt.abs().pow(3).mean(1).pow(1 / 3)
    d = torch.cat([torch.nn.functional.normalize(cls, dim=1), torch.nn.functional.normalize(gem, dim=1)], 1)
    return torch.nn.functional.normalize(d, dim=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()
    cams = load_cameras()
    st = frame_stamps()
    pool = ThreadPoolExecutor(8)
    for cam in CAMS:
        p = os.path.join(OUT, f'global_{cam}.npy')
        if os.path.exists(p):
            continue
        t0 = time.time()
        desc = np.zeros((len(st), 768), np.float16)
        paths = [frame_path(cam, s) for s in st]
        for b in range(0, len(st), BATCH):
            ims = list(pool.map(lambda q: prep(q, cams[cam].mask), paths[b:b + BATCH]))
            ok = [i for i, im in enumerate(ims) if im is not None]
            x = torch.from_numpy(np.stack([ims[i] for i in ok])).cuda()
            desc[b + np.array(ok)] = describe(model, x).cpu().numpy().astype(np.float16)
        np.save(p, desc)
        print(f'{cam}: {len(st)} frames in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
