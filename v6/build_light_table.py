#!/usr/bin/env python3
"""Training-time artefact of v6 (reads the training mcap; run once, ~3 min, CPU only):

    python3 v6/build_light_table.py
      -> v6/model/night_stats.json  per camera: median (p10, p90) of the per-image median raw grey inside the body
                                    mask (target of v6's exposure normalisation)
      -> v6/model/light_<cam>.npz   per map point of features/db_<cam>.npz (same order):
                                    peak = max raw grey (half-res, no CLAHE) in a 5x5 patch at the keypoint,
                                    mean = mean raw grey in a 9x9 patch.
At night the map leans on light sources (lamps, lit boards, their reflections). In daylight those look different or
vanish, so v6 can down-weight / drop them, and the analysis tells how much of the map is "night-only".
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from mcap.reader import make_reader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from camera_model import CAMS, H, W  # noqa: E402

MCAP = os.path.join(ROOT, 'data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap')
TOPIC = {f'/constructor0/sensor/camera_{c}/compressed_image': c for c in CAMS}


def stamp_of(raw):
    # sensor_msgs/CompressedImage CDR: 4 byte header, then header.stamp sec (int32), nanosec (uint32)
    sec = int.from_bytes(raw[4:8], 'little', signed=True)
    ns = int.from_bytes(raw[8:12], 'little')
    return sec * 10**9 + ns


def jpeg_of(raw):
    # after stamp: frame_id string (uint32 len + bytes, padded to 4), format string, then uint8[] data
    o = 12
    for _ in range(2):
        n = int.from_bytes(raw[o:o + 4], 'little'); o += 4 + n
        o = (o + 3) & ~3
    n = int.from_bytes(raw[o:o + 4], 'little'); o += 4
    return raw[o:o + n]


def main():
    os.makedirs(os.path.join(HERE, 'model'), exist_ok=True)
    db = {c: np.load(os.path.join(ROOT, 'features', f'db_{c}.npz')) for c in CAMS}
    kf = db['fl']['kf_stamp'].astype(np.int64)
    off = {c: db[c]['off'] for c in CAMS}
    kps = {c: db[c]['kps'] for c in CAMS}
    peak = {c: np.zeros(len(kps[c]), np.uint8) for c in CAMS}
    mean = {c: np.zeros(len(kps[c]), np.uint8) for c in CAMS}
    jobs = []
    med = {c: [] for c in CAMS}
    masks = {c: cv2.imread(os.path.join(ROOT, 'masks', f'{c}.png'), cv2.IMREAD_GRAYSCALE) for c in CAMS}

    def work(c, i, jpg):
        g = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if g is None:
            return
        g = cv2.resize(g, (W, H)) if g.shape != (H, W) else g
        med[c].append(float(np.median(g[masks[c] > 0])))
        a, b = off[c][i], off[c][i + 1]
        if b <= a:
            return
        mx = cv2.dilate(g, np.ones((5, 5), np.uint8))
        bl = cv2.blur(g, (9, 9))
        k = np.round(kps[c][a:b]).astype(int)
        u, v = k[:, 0].clip(0, W - 1), k[:, 1].clip(0, H - 1)
        peak[c][a:b] = mx[v, u]
        mean[c][a:b] = bl[v, u]

    n = 0
    with open(MCAP, 'rb') as f, ThreadPoolExecutor(24) as pool:
        for _, ch, m in make_reader(f).iter_messages(topics=list(TOPIC)):
            s = stamp_of(m.data)
            i = int(np.searchsorted(kf, s))
            for j in (i - 1, i):
                if 0 <= j < len(kf) and abs(int(kf[j]) - s) < 5_000_000:
                    jobs.append(pool.submit(work, TOPIC[ch.topic], j, bytes(jpeg_of(m.data))))
                    n += 1
                    break
        for j in jobs:
            j.result()
    print(f'{n} keyframe images')
    import json
    stats = {c: dict(median_grey=float(np.median(med[c])), p10=float(np.percentile(med[c], 10)),
                     p90=float(np.percentile(med[c], 90))) for c in CAMS}
    json.dump(stats, open(os.path.join(HERE, 'model', 'night_stats.json'), 'w'), indent=1)
    print(json.dumps(stats))
    for c in CAMS:
        np.savez(os.path.join(HERE, 'model', f'light_{c}.npz'), peak=peak[c], mean=mean[c])
        p = peak[c]
        print(f'{c}: {len(p)} points  peak>=250 {np.mean(p >= 250) * 100:5.1f} %  peak>=200 {np.mean(p >= 200) * 100:5.1f} %'
              f'  mean>=120 {np.mean(mean[c] >= 120) * 100:5.1f} %  median peak {np.median(p):.0f}')


if __name__ == '__main__':
    main()
