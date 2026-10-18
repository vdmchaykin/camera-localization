#!/usr/bin/env python3
"""Training-time artefact: the day descriptor layer of the map, from a day-styled copy of the training run.

    python3 tools/build_day_layer.py --mcap DAY.mcap                                   # all laps (submission model)
    python3 tools/build_day_layer.py --mcap DAY.mcap --map-config configs/map_val_k3k6.json --out model/day_val
      -> <out>/day_<cam>.npy   (N, 256) float16, aligned with model/features/db_<cam>.npz['desc'] (same points, order)
      -> <out>/day_meta.json   source bag, keyframes used / skipped

The day bag is the training run with every image re-lit night -> day (Qwen image editing); geometry and header stamps
are unchanged, so each map point (keyframe image position + 3D point) keeps its place. For every map keyframe and
camera found in the day bag (header stamp within 5 ms) the day image goes through exactly the query path of the
localizer (grey, exposure normalisation to the night median, CLAHE, FP16 SuperPoint backbone, L2-normalised dense
descriptors) and the descriptor is sampled at the map keypoints. Keyframes excluded by --map-config (validation:
held-out laps K3 / K6) and keyframes missing in the day bag keep the night descriptor.
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mcap.reader import make_reader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from lightglue import SuperPoint  # noqa: E402
from lightglue.superpoint import sample_descriptors  # noqa: E402
from camloc.camera_model import CAMS, H, W  # noqa: E402
from camloc.localizer import GAMMA_MAX, GAMMA_MIN, _backbone  # noqa: E402  same query path as the localizer
from camloc.paths import CONFIGS, DAY, FEAT, MASKS, MODEL  # noqa: E402

SET_TOL_NS = 5_000_000
CLAHE = cv2.createCLAHE(2.0, (8, 8))        # as camloc/localizer.py


def cdr_stamp(raw):
    return int.from_bytes(raw[4:8], 'little', signed=True) * 10**9 + int.from_bytes(raw[8:12], 'little')


def cdr_jpeg(raw):
    o = 12
    for _ in range(2):                       # frame_id, format
        n = int.from_bytes(raw[o:o + 4], 'little')
        o += 4 + n
        o += (-(o - 4)) % 4
    n = int.from_bytes(raw[o:o + 4], 'little')
    return bytes(raw[o + 4:o + 4 + n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcap', required=True, help='day-styled copy of the training run')
    ap.add_argument('--map-config', default=os.path.join(CONFIGS, 'map_full.json'),
                    help='keyframes excluded here keep the night descriptor (validation: configs/map_val_k3k6.json)')
    ap.add_argument('--out', default=DAY)
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    db = {c: np.load(os.path.join(FEAT, f'db_{c}.npz')) for c in CAMS}
    kf = db[CAMS[0]]['kf_stamp'].astype(np.int64)
    off = {c: db[c]['off'] for c in CAMS}
    kps = {c: db[c]['kps'] for c in CAMS}
    day = {c: db[c]['desc'].copy() for c in CAMS}             # default: night descriptor
    excl = np.zeros(len(kf), bool)
    for lo, hi in json.load(open(a.map_config))['exclude_windows_ns']:
        excl |= (kf >= lo) & (kf <= hi)
    target = {c: v['median_grey'] for c, v in json.load(open(os.path.join(MODEL, 'night_stats.json'))).items()}
    masks = {}

    sp = SuperPoint(max_num_keypoints=1024, detection_threshold=0.0002).eval().cuda().half().to(
        memory_format=torch.channels_last)

    def prep(c, jpeg):
        half = cv2.resize(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), (W, H),
                          interpolation=cv2.INTER_AREA)
        raw = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        if c not in masks:
            masks[c] = cv2.imread(os.path.join(MASKS, f'{c}.png'), cv2.IMREAD_GRAYSCALE)
        med = float(np.median(raw[masks[c] > 0]))
        gm = 1.0
        if 5 < med < 250:
            gm = float(np.clip(np.log(target[c] / 255) / np.log(med / 255), GAMMA_MIN, GAMMA_MAX))
            if abs(gm - 1) > 0.05:
                raw = cv2.LUT(raw, np.clip(255 * (np.arange(256) / 255) ** gm, 0, 255).astype(np.uint8))
        return CLAHE.apply(raw), gm

    @torch.no_grad()
    def process(i, imgs):
        names = [c for c in CAMS if c in imgs]
        g = np.stack([imgs[c][0] for c in names])
        x = torch.from_numpy(g).cuda().half().div_(255)[:, None].contiguous(memory_format=torch.channels_last)
        _, dense = _backbone(sp, x)
        dense = F.normalize(dense.float(), p=2, dim=1)
        for j, c in enumerate(names):
            s, e = off[c][i], off[c][i + 1]
            if e > s:
                k = torch.from_numpy(kps[c][s:e].astype(np.float32)).cuda()
                d = sample_descriptors(k[None], dense[j:j + 1], 8)[0].T
                day[c][s:e] = d.cpu().numpy().astype(np.float16)

    # stream the day bag; a keyframe set is complete when all 7 cameras arrived
    topic = {f'/constructor0/sensor/camera_{c}/compressed_image': c for c in CAMS}
    want = ~excl
    pending, done, gammas, n_img = {}, np.zeros(len(kf), bool), [], 0
    t0, t1 = int(kf[want].min()) - SET_TOL_NS, int(kf[want].max()) + SET_TOL_NS
    with open(a.mcap, 'rb') as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=list(topic), start_time=t0 - 10**9, end_time=t1 + 10**9):
            st = cdr_stamp(msg.data)
            i = int(np.searchsorted(kf, st))
            i = min((j for j in (i - 1, i) if 0 <= j < len(kf)), key=lambda j: abs(kf[j] - st))
            if abs(kf[i] - st) > SET_TOL_NS or not want[i] or done[i]:
                continue
            c = topic[ch.topic]
            img, gm = prep(c, cdr_jpeg(msg.data))
            pending.setdefault(i, {})[c] = (img, gm)
            gammas.append(gm)
            n_img += 1
            if len(pending[i]) == len(CAMS):
                process(i, pending.pop(i))
                done[i] = True
            for j in [j for j in pending if kf[j] < st - 200_000_000]:   # a camera missing (camera r drops)
                process(j, pending.pop(j))
                done[j] = True
    for j in list(pending):
        process(j, pending.pop(j))
        done[j] = True

    for c in CAMS:
        np.save(os.path.join(OUT, f'day_{c}.npy'), day[c])
    meta = dict(source=os.path.abspath(a.mcap), keyframes=int(len(kf)), keyframes_day=int(done.sum()),
                keyframes_excluded=int(excl.sum()), map_config=os.path.basename(a.map_config), keyframes_missing=int((want & ~done).sum()),
                images=n_img, gamma_median=float(np.median(gammas)), build_s=round(time.time() - t_start, 1))
    json.dump(meta, open(os.path.join(OUT, 'day_meta.json'), 'w'), indent=1)
    # how different is the day layer from the night one (same points)
    for c in CAMS:
        sel = np.concatenate([np.arange(off[c][i], off[c][i + 1]) for i in np.where(done)[0][::50]])
        cos = np.sum(day[c][sel].astype(np.float32) * db[c]['desc'][sel].astype(np.float32), 1)
        print(f'{c}: cos(day, night) median {np.median(cos):.2f}, p10 {np.percentile(cos, 10):.2f}')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
