#!/usr/bin/env python3
"""Packaging helper: features/set_stamps.npy = camera_model.frame_stamps(), the row order of features/global_<cam>.npy
(camloc/localize.py uses it to pick the keyframe rows). Written when the map was packed into the model archive.

    python3 write_set_stamps.py      -> features/set_stamps.npy  (N_sets,) int64
"""
import os
import numpy as np
from camera_model import ROOT, frame_stamps

if __name__ == '__main__':
    st = frame_stamps()
    np.save(os.path.join(ROOT, 'features', 'set_stamps.npy'), st)
    print(f'{len(st)} set stamps')
