#!/usr/bin/env python3
"""
DINOv2 global descriptors for the keyframes of a featmap / COLMAP map (one file per camera).

    python dino_desc.py --data data/ --map maps/colmap_7cam

Writes maps/<map>/<cam>_dino.npy (N_keyframes x D, L2-normalised, float16). featmap.py uses them
with --dino: among the map keyframes near the predicted pose it matches against the ones that
LOOK most similar to the query image, instead of simply the nearest ones.

Descriptor: DINOv2 ViT-S/14 on a 224 x 448 resize, [CLS token, GeM-pooled patch tokens].
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Dino:
    def __init__(self, name="dinov2_vits14", size=(224, 448)):
        self.model = torch.hub.load("facebookresearch/dinov2", name).eval().to(DEV)
        self.h, self.w = size
        self.mean, self.std = MEAN.to(DEV), STD.to(DEV)

    @torch.no_grad()
    def __call__(self, imgs_bgr):
        x = np.stack([cv2.resize(im, (self.w, self.h))[:, :, ::-1] for im in imgs_bgr])
        x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float().div(255).to(DEV)
        x = (x - self.mean) / self.std
        f = self.model.forward_features(x)
        cls = f["x_norm_clstoken"]
        p = f["x_norm_patchtokens"].clamp(min=1e-6)
        gem = p.pow(3).mean(1).pow(1 / 3)
        d = torch.cat([torch.nn.functional.normalize(cls, dim=1),
                       torch.nn.functional.normalize(gem, dim=1)], 1)
        return torch.nn.functional.normalize(d, dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    data, mp = Path(a.data), Path(a.map)
    import json
    cams = json.loads((mp / "meta.json").read_text())["cams"]
    dino = Dino()
    for cam in cams:
        kt = pd.read_parquet(mp / f"{cam}_kf.parquet")
        paths = [data / "images" / cam / f"{s}.jpg" for s in kt.stamp_ns]
        out = []
        for i in range(0, len(paths), a.batch):
            out.append(dino([cv2.imread(str(p)) for p in paths[i:i + a.batch]]))
        D = np.concatenate(out).astype(np.float16)
        np.save(mp / f"{cam}_dino.npy", D)
        print(f"[{cam}] {D.shape[0]} keyframes -> {mp / f'{cam}_dino.npy'}")


if __name__ == "__main__":
    main()