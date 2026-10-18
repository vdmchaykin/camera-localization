"""DINOv2 ViT-S/14 global descriptor of a camera image, for keyframe retrieval in the global relocalization.

Input: half-res BGR image, body mask filled with grey, resized to IN_W x IN_H, ImageNet-normalised RGB.
Descriptor = [CLS ; GeM(patch tokens)], each half L2-normalised, (768,). The map holds it for every frame of the
training run (features/global_<cam>.npy, float16).
"""
import numpy as np
import torch

IN_W, IN_H = 462, 224        # multiples of 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def describe(model, x):
    out = model.forward_features(x)
    cls = out['x_norm_clstoken']
    pt = out['x_norm_patchtokens']
    gem = pt.abs().pow(3).mean(1).pow(1 / 3)
    d = torch.cat([torch.nn.functional.normalize(cls, dim=1), torch.nn.functional.normalize(gem, dim=1)], 1)
    return torch.nn.functional.normalize(d, dim=1)
