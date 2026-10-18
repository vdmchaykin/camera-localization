#!/usr/bin/env python3
"""Run the localizer on a run (mcap) and write the submission CSV.

    python3 tools/run.py --mcap RUN.mcap --out out/RUN                      # first pose unknown: found by the localizer
    python3 tools/run.py --mcap RUN.mcap --out out/RUN --first-pose X Y YAW # first pose of the run given
    # validation on the training run (map without the held-out laps K3 / K6, first pose = GT at the lap start):
    python3 tools/run.py --mcap TRAIN.mcap --out out/K3 --start-s 523.4 --max-s 110.6 --map-config configs/map_val_k3k6.json \
        --first-pose-from-gt --evaluate PATH/TO/tools/evaluate.py

Processing is causal: camera sets are fed in stamp order, the pose for set t uses only images and car state up to t.
Every set is processed (no drops); the time per set (decode + step) is measured and reported.

Output in --out:
  poses.csv      timestamp_ns,x,y,yaw - the submission file (one pose per camera set, track frame, yaw in rad)
  frames.csv     per set: timestamp_ns, decode_ms, step_ms, n_cams
  summary.json   runtime statistics, hardware, localizer statistics; with GT in the bag also the errors
  gt_window.csv, evaluate.txt   (only if the bag has GT /tf) GT of the window (format of the organisers'
                 tools/extract_gt.py) and the output of their tools/evaluate.py on it
"""
import argparse, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mcap', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--first-pose', type=float, nargs=3, metavar=('X', 'Y', 'YAW'), help='first pose of the run')
    ap.add_argument('--first-pose-from-gt', action='store_true',
                    help='validation only: first pose = GT (/tf) at the first camera set of the window')
    ap.add_argument('--start-s', type=float, help='window start, s from the start of the bag')
    ap.add_argument('--max-s', type=float, help='window length, s')
    ap.add_argument('--map-config', default=os.path.join(PROJECT, 'configs', 'map_full.json'),
                    help='map subset: configs/map_full.json (all keyframes, for test runs) or '
                         'configs/map_val_k3k6.json (held-out laps K3 / K6 excluded, for validation)')
    ap.add_argument('--evaluate', help="organisers' tools/evaluate.py (dataset starter kit), run on the GT window "
                                       "if the bag has GT")
    a = ap.parse_args()
    os.environ['LOC_MAP_CONFIG'] = os.path.abspath(a.map_config)

    import cv2
    import numpy as np
    import torch
    from camloc import bag
    from camloc.localizer import Localizer

    os.makedirs(a.out, exist_ok=True)
    have, (b0, _) = bag.topics(a.mcap)
    t0 = None if a.start_s is None else b0 + int(a.start_s * 1e9)
    t1 = None if a.max_s is None else (t0 or b0) + int(a.max_s * 1e9)
    info, tfs, bounds, bounds_src = bag.read_static(a.mcap)
    state_src = bag.state_source(a.mcap)
    print(f'[run] {a.mcap}\n[run] car state: {state_src}, track bounds: {bounds_src}', flush=True)

    sets = bag.stream(a.mcap, t0, t1)
    first = next(sets)
    gt = None
    if '/tf' in have:
        gt_t, gt = bag.read_gt(a.mcap, None if t0 is None else t0 - 10**9, None if t1 is None else t1 + 10**9)
        gt = gt if len(gt) else None
    if a.first_pose is not None:
        fp = dict(stamp_ns=int(first[0]), x=a.first_pose[0], y=a.first_pose[1], yaw=a.first_pose[2])
    elif a.first_pose_from_gt:
        if gt is None:
            raise SystemExit('--first-pose-from-gt: the bag has no GT (/tf map -> base_link)')
        i = int(np.searchsorted(gt_t, first[0]))
        i = min(max(i, 1), len(gt) - 1)
        w = (first[0] - gt_t[i - 1]) / (gt_t[i] - gt_t[i - 1])
        dyaw = np.angle(np.exp(1j * (gt[i, 3] - gt[i - 1, 3])))
        fp = dict(stamp_ns=int(first[0]), x=float(gt[i - 1, 0] + w * (gt[i, 0] - gt[i - 1, 0])),
                  y=float(gt[i - 1, 1] + w * (gt[i, 1] - gt[i - 1, 1])), yaw=float(gt[i - 1, 3] + w * dyaw))
    else:
        fp = None
    print(f'[run] first pose: {fp if fp else "not given - global relocalization"}', flush=True)

    t_init = time.perf_counter()
    loc = Localizer(fp, info, tfs, bounds)
    t_init = time.perf_counter() - t_init
    print(f'[run] init {t_init:.1f} s', flush=True)

    pool = ThreadPoolExecutor(len(bag.CAMS))
    decode = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    poses, frames = [], []
    last = time.perf_counter()

    def process(item):
        stamp, jpegs, cs = item
        ta = time.perf_counter()
        ims = dict(zip(jpegs, pool.map(decode, jpegs.values())))
        ims = {c: ims.get(c) for c in bag.CAMS}
        tb = time.perf_counter()
        x, y, yaw = loc.step(int(stamp), ims, cs if state_src else None)     # None: no car state in this run
        tc = time.perf_counter()
        poses.append((int(stamp), x, y, float(np.angle(np.exp(1j * yaw)))))
        frames.append((int(stamp), (tb - ta) * 1e3, (tc - tb) * 1e3, len(jpegs)))

    process(first)
    for item in sets:
        process(item)
        if time.perf_counter() - last > 30:
            last = time.perf_counter()
            print(f'[run] {len(poses)} sets, step mean {np.mean([f[2] for f in frames]):.1f} ms', flush=True)

    with open(os.path.join(a.out, 'poses.csv'), 'w') as f:
        f.write('timestamp_ns,x,y,yaw\n')
        for t, x, y, yaw in poses:
            f.write(f'{t},{x:.4f},{y:.4f},{yaw:.6f}\n')
    with open(os.path.join(a.out, 'frames.csv'), 'w') as f:
        f.write('timestamp_ns,decode_ms,step_ms,n_cams\n')
        for t, d, s, n in frames:
            f.write(f'{t},{d:.3f},{s:.3f},{n}\n')
    fr = np.array([(d, s) for _, d, s, _ in frames])
    tot = fr.sum(1)
    S = dict(mcap=os.path.abspath(a.mcap), window_s=[a.start_s, a.max_s], map_config=os.path.basename(a.map_config),
             first_pose=dict(given=a.first_pose is not None, from_gt=a.first_pose_from_gt),
             car_state=state_src, track_bounds=bounds_src, sets=len(poses),
             hardware=dict(gpu=torch.cuda.get_device_name(0), cpu=platform.processor() or platform.machine(),
                           torch=torch.__version__, python=platform.python_version()),
             init_s=round(t_init, 1),
             ms_per_set=dict(mean=float(tot.mean()), p95=float(np.percentile(tot, 95)), max=float(tot.max()),
                             decode_mean=float(fr[:, 0].mean()), step_mean=float(fr[:, 1].mean()),
                             over_100ms=int((tot > 100).sum())),
             localizer=loc.summary())
    if gt is not None:
        # as the organisers' tools/evaluate.py: GT interpolated at each pose stamp (yaw unwrapped); poses outside the
        # GT or in a GT gap > 50 ms are skipped; errors split into the GT car frame
        pt = np.array([p[0] for p in poses], np.int64)
        P = np.array([p[1:] for p in poses])                       # x, y, yaw
        i = np.searchsorted(gt_t, pt)
        ok = (i > 0) & (i < len(gt_t))
        i = np.clip(i, 1, len(gt_t) - 1)
        ok &= gt_t[i] - gt_t[i - 1] <= 50_000_000
        w = (pt - gt_t[i - 1]) / (gt_t[i] - gt_t[i - 1])
        yu = np.unwrap(gt[:, 3])
        gx = gt[i - 1, 0] + w * (gt[i, 0] - gt[i - 1, 0])
        gy = gt[i - 1, 1] + w * (gt[i, 1] - gt[i - 1, 1])
        gyaw = yu[i - 1] + w * (yu[i] - yu[i - 1])
        dx, dy = P[:, 0] - gx, P[:, 1] - gy
        lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
        lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
        pos = np.hypot(dx, dy)
        dyaw = np.degrees(np.angle(np.exp(1j * (P[:, 2] - gyaw))))
        rms = lambda v: float(np.sqrt(np.mean(v[ok] ** 2)))
        S['errors'] = dict(evaluated=int(ok.sum()), pos_rmse=rms(pos), lat_rmse=rms(lat), lon_rmse=rms(lon),
                           yaw_rmse_deg=rms(dyaw), pos_median=float(np.median(pos[ok])),
                           pos_p95=float(np.percentile(pos[ok], 95)), pos_max=float(pos[ok].max()),
                           within_1m=float(np.mean(pos[ok] < 1.0) * 100))
        sel = (gt_t >= pt[0] - 100_000_000) & (gt_t <= pt[-1] + 100_000_000)
        gp = os.path.join(a.out, 'gt_window.csv')
        with open(gp, 'w') as f:
            f.write('timestamp_ns,x,y,z,yaw\n')
            for t, r in zip(gt_t[sel], gt[sel]):
                f.write(f'{t},{r[0]:.4f},{r[1]:.4f},{r[2]:.4f},{r[3]:.6f}\n')
        if a.evaluate:
            ev = subprocess.run([sys.executable, a.evaluate, gp, os.path.join(a.out, 'poses.csv')],
                                capture_output=True, text=True)
            open(os.path.join(a.out, 'evaluate.txt'), 'w').write(ev.stdout + ev.stderr)
            S['evaluate'] = ev.stdout
    json.dump(S, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, default=str)
    m = S['ms_per_set']
    print(f"[run] {len(poses)} poses -> {os.path.join(a.out, 'poses.csv')}\n"
          f"[run] time per set: mean {m['mean']:.1f} ms (decode {m['decode_mean']:.1f} + step {m['step_mean']:.1f}), "
          f"p95 {m['p95']:.1f}, > 100 ms: {m['over_100ms']}, on {S['hardware']['gpu']}", flush=True)
    if 'errors' in S:
        e = S['errors']
        print(f"[run] vs GT: pos RMSE {e['pos_rmse']:.3f} m, lat {e['lat_rmse']:.3f}, lon {e['lon_rmse']:.3f}, "
              f"yaw {e['yaw_rmse_deg']:.3f} deg, p95 {e['pos_p95']:.3f}, max {e['pos_max']:.3f}, "
              f"< 1 m {e['within_1m']:.1f} %", flush=True)
    if 'boot' in S['localizer']:
        print(f"[run] boot: {S['localizer']['boot']}", flush=True)


if __name__ == '__main__':
    main()
