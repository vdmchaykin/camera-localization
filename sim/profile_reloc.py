#!/usr/bin/env python3
"""Stage timing of one relocalization of sim/my_loc.Localizer (GPU must be idle), plus JPEG decoding.

    python3 sim/profile_reloc.py [--n 40] [--start-s 523]

Reads N camera sets from the mcap (one every 0.5 s from --start-s), runs the same code path as the live worker,
timing each stage with cuda sync. Per-camera SuperPoint / LightGlue calls are summed.
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import my_loc  # noqa: E402
from simulate import CAMS, IMG_T, read_static, stamp_of  # noqa: E402
from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

MCAP = os.path.join(my_loc.ROOT, 'data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap')


def sets(t0, n, every_ns=500_000_000):
    out, cur, last = [], {}, None
    with open(MCAP, 'rb') as f:
        for _, ch, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=list(IMG_T), start_time=t0, log_time_order=True):
            s = stamp_of(msg)
            if last is not None and s - last < every_ns and not cur:
                continue
            if cur and abs(s - next(iter(cur.values()))[0]) > 5_000_000:
                if len(cur) == len(CAMS):
                    out.append(cur); last = next(iter(cur.values()))[0]
                cur = {}
                if len(out) == n:
                    break
                if s - last < every_ns:
                    continue
            cur[IMG_T[ch.topic]] = (s, bytes(msg.data))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--start-s', type=float, default=523.0)
    a = ap.parse_args()
    with open(MCAP, 'rb') as f:
        bag0 = make_reader(f).get_summary().statistics.message_start_time
    t0 = bag0 + int(a.start_s * 1e9)
    raw = sets(t0, a.n)
    info, tfs, bounds = read_static(MCAP, t0)
    first = next(iter(raw[0].values()))[0]
    os.environ['LOC_SYNC'] = '1'
    loc = my_loc.Localizer(dict(stamp_ns=first, x=0.0, y=0.0, yaw=0.0), info, tfs, bounds)
    R, fe = loc.reloc, loc.reloc.fe
    T = {k: [] for k in ('decode_full', 'decode_half', 'resize', 'dino', 'retrieve', 'superpoint', 'lightglue',
                         'pnp', 'total')}
    sync = torch.cuda.synchronize
    for k, s in enumerate(raw):
        stamp = next(iter(s.values()))[0]
        t = time.perf_counter()
        imgs = {c: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for c, (_, b) in s.items()}
        t_dec = time.perf_counter() - t
        t = time.perf_counter()
        for c, (_, b) in s.items():
            cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        t_dec2 = time.perf_counter() - t
        sync(); t0_ = time.perf_counter()
        half = {c: cv2.resize(im, (my_loc.W, my_loc.H), interpolation=cv2.INTER_AREA) for c, im in imgs.items()}
        x = []
        for c in CAMS:
            im = half[c].copy(); im[loc.cams[c].mask == 0] = 127
            im = cv2.resize(im, (my_loc.IN_W, my_loc.IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
            x.append(((im - my_loc.MEAN) / my_loc.STD).transpose(2, 0, 1))
        t1 = time.perf_counter()
        with torch.no_grad():
            g = my_loc.describe(loc.dino, torch.from_numpy(np.stack(x)).cuda())
        sync(); t2 = time.perf_counter()
        sim = torch.einsum('ckd,cd->k', loc.Gk.float(), g).cpu().numpy()
        al = loc.allowed_base & (np.abs(loc.kf - stamp) > loc.self_excl_ns)
        sim[~al] = -np.inf
        cand = [int(np.argmax(sim))]
        t3 = time.perf_counter()
        qf = {c: fe.extract(my_loc._CLAHE.apply(cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY)), loc.cams[c].mask)
              for c in CAMS}
        sync(); t4 = time.perf_counter()
        corr = R.correspondences(qf, cand)
        sync(); t5 = time.perf_counter()
        R.solve(corr)
        t6 = time.perf_counter()
        if k >= 3:
            for key, v in (('decode_full', t_dec), ('decode_half', t_dec2), ('resize', t1 - t0_), ('dino', t2 - t1),
                           ('retrieve', t3 - t2), ('superpoint', t4 - t3), ('lightglue', t5 - t4), ('pnp', t6 - t5),
                           ('total', t6 - t0_)):
                T[key].append(v * 1e3)
    res = {k: dict(mean=round(float(np.mean(v)), 1), p95=round(float(np.percentile(v, 95)), 1)) for k, v in T.items()}
    print(f'{len(T["total"])} sets, GPU {torch.cuda.get_device_name()}')
    for k, v in res.items():
        print(f'  {k:12s} mean {v["mean"]:7.1f} ms   p95 {v["p95"]:7.1f} ms')
    json.dump(res, open(os.path.join(HERE, 'runs', 'profile_reloc.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
