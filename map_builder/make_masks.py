#!/usr/bin/env python3
"""Static car-body masks per camera (half-res 753x364), hand-drawn polygons on median images.

    python3 make_masks.py      -> masks/<cam>.png (255 = usable, 0 = car body / lens artefact), masks/preview.jpg

Polygons were read off masks/median_<cam>.jpg (median of moving frames: body sharp, scene blurred).
Wheels steer, so wheel polygons are generous. Temporal-variance masks failed (uniform asphalt is low-variance too).
"""
import os
import cv2
import numpy as np
from camera_model import CAMS, ROOT, H, W, frame_path, frame_stamps

OUT = os.path.join(ROOT, 'masks')
B = H  # bottom
POLYS = {
    'fl': [[(440, 335), (600, 322), (753, 318), (753, B), (440, B)],          # nose
           [(678, 262), (742, 262), (742, 304), (678, 304)]],                 # spot on the lens
    'fr': [[(0, 212), (250, 198), (450, 202), (585, 215), (595, 188), (680, 182), (753, 186), (753, B), (0, B)]],
    'cl': [[(0, 252), (188, 252), (192, 236), (552, 236), (560, 202), (672, 198), (690, 238), (753, 238),
            (753, B), (0, B)]],
    'cr': [[(0, 0), (22, 0), (18, 150), (22, 205), (98, 205), (102, 172), (218, 170), (226, 210), (640, 218),
            (753, 238), (753, B), (0, B)]],
    'rl': [],
    'rr': [[(0, 256), (272, 256), (292, 272), (572, 272), (580, 212), (698, 208), (702, 150), (753, 142),
            (753, B), (0, B)]],
    'r': [],
}
LEFT_STRIP = 6   # dark column on the left edge of every camera


def main():
    os.makedirs(OUT, exist_ok=True)
    st = frame_stamps()
    previews = []
    for cam in CAMS:
        mask = np.full((H, W), 255, np.uint8)
        mask[:, :LEFT_STRIP] = 0
        for p in POLYS[cam]:
            cv2.fillPoly(mask, [np.array(p, np.int32)], 0)
        cv2.imwrite(os.path.join(OUT, f'{cam}.png'), mask)
        im = cv2.imread(frame_path(cam, st[len(st) // 2]))
        im[mask == 0] = (0.5 * im[mask == 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(im, f'{cam} masked {100 * (mask == 0).mean():.0f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        previews.append(im)
        print(cam, f'masked {100 * (mask == 0).mean():.1f} %')
    previews.append(np.zeros_like(previews[0]))
    cv2.imwrite(os.path.join(OUT, 'preview.jpg'),
                np.vstack([np.hstack(previews[i:i + 2]) for i in range(0, 8, 2)]))


if __name__ == '__main__':
    main()
