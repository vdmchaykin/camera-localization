#!/usr/bin/env python3
"""
Static "own car body" mask per camera from the SAM 3 masks of the training frames.

The own car (wing, wheels, nose) is at the same pixels in every frame, other cars are not.
So a pixel that is labelled "car" in most frames is the body. Result:
    data/masks/<cam>.png    255 = usable pixel, 0 = car body
featmap.py already picks these files up for map building and for queries at inference
(no SAM needed at inference). colmap_map.py uses them via --body-masks.

    python make_body_masks.py --data data/ --masks sam3 --cams cl cr fl fr r rl rr
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--masks", default="sam3", help="folder under data/: sam3 or sam3_video")
    ap.add_argument("--cams", nargs="+", default=["cl", "cr", "fl", "fr", "r", "rl", "rr"])
    ap.add_argument("--frames", type=int, default=400, help="frames sampled per camera")
    ap.add_argument("--min-frac", type=float, default=0.7, help="'car' in this share of frames")
    ap.add_argument("--dilate", type=int, default=9, help="px safety margin around the body")
    a = ap.parse_args()

    data = Path(a.data)
    classes = json.loads((data / a.masks / "classes.json").read_text())
    car_id = next(c["id"] for c in classes if c["name"] == "car")
    out = data / "masks"
    out.mkdir(exist_ok=True)

    for cam in a.cams:
        files = sorted((data / a.masks / cam).glob("*.png"))
        if not files:
            print(f"[{cam}] no masks found, skipped")
            continue
        pick = [files[i] for i in np.linspace(0, len(files) - 1, min(a.frames, len(files))).astype(int)]
        acc = None
        for p in pick:
            lab = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            acc = (lab == car_id).astype(np.float32) if acc is None else acc + (lab == car_id)
        frac = acc / len(pick)
        body = (frac >= a.min_frac).astype(np.uint8)
        # keep only big blobs (the body), close holes, add a margin
        n, cc, stats, _ = cv2.connectedComponentsWithStats(body)
        big = np.zeros_like(body)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 0.002 * body.size:
                big[cc == i] = 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1, 2 * a.dilate + 1))
        big = cv2.morphologyEx(big, cv2.MORPH_CLOSE, k)
        big = cv2.dilate(big, k)
        usable = np.where(big > 0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{cam}.png"), usable)
        # preview: body in red over a median-ish frame
        print(f"[{cam}] body covers {big.mean():.0%} of the image -> {out / f'{cam}.png'}")
        vis = cv2.cvtColor((frac * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis[big > 0] = (0.5 * vis[big > 0] + [0, 0, 128]).astype(np.uint8)
        cv2.imwrite(str(out / f"{cam}_preview.jpg"), vis)


if __name__ == "__main__":
    main()