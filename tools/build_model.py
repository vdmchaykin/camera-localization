#!/usr/bin/env python3
"""Training-time artefacts derived from the training run (needs its GT /tf; run once after the map exists):

    python3 tools/build_model.py --mcap TRAIN.mcap                     # motion calibration from all laps
    python3 tools/build_model.py --mcap TRAIN.mcap --laps 1 2 4 5 7    # validation build (K3 / K6 held out)

Writes into model/ (or $CAMLOC_MODEL):
  kf_poses.npz       GT pose (T_map_base 4x4, x, y, yaw) of every map keyframe (features/db_fl.npz kf_stamp)
  motion_calib.json  car state -> base_link motion: vy_eff = a*vy + b*wz + c, wz_eff = s*wz + d, vx_eff = k*vx;
                     least squares against the GT body-frame velocity (finite differences over +-20 ms), car state
                     messages with vx > 5 m/s of the chosen laps (configs/laps.csv), every 3rd message
  night_stats.json   per camera: median / p10 / p90 of the per-image median raw grey inside the body mask over the
                     map keyframes (target of the exposure normalisation)
  track_bounds.npz   /constructor0/track_bounds (used for runs whose bag has no borders)
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from camloc import bag  # noqa: E402
from camloc.camera_model import CAMS, H, W  # noqa: E402
from camloc.paths import CONFIGS, FEAT, MASKS, MODEL, fresh  # noqa: E402


def read_tf(path):
    """map -> base_link from /tf: stamps (N,), xyz (N, 3), quaternions xyzw (N, 4), sorted, unique stamps."""
    t, p, q = [], [], []
    f, it = bag._reader(path, topics=['/tf'], log_time_order=True)
    with f:
        for _, _, _, msg in it:
            for tr in msg.transforms:
                if tr.header.frame_id == 'map' and tr.child_frame_id.endswith('base_link'):
                    a, r = tr.transform.translation, tr.transform.rotation
                    t.append(bag.stamp_of(tr)); p.append((a.x, a.y, a.z)); q.append((r.x, r.y, r.z, r.w))
    t = np.array(t, np.int64)
    o = np.argsort(t, kind='stable')
    t, p, q = t[o], np.array(p)[o], np.array(q)[o]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], p[keep], q[keep]


class GT:
    def __init__(self, path):
        self.t, self.xyz, q = read_tf(path)
        self.rot = Rotation.from_quat(q)
        self.yaw = np.unwrap(self.rot.as_euler('zyx')[:, 0])
        self.s = (self.t - self.t[0]) / 1e9
        self.slerp = Slerp(self.s, self.rot)

    def T(self, stamps):
        s = np.clip((np.asarray(stamps, np.int64) - self.t[0]) / 1e9, 0, self.s[-1])
        T = np.tile(np.eye(4), (len(s), 1, 1))
        T[:, :3, :3] = self.slerp(s).as_matrix()
        for k in range(3):
            T[:, k, 3] = np.interp(s, self.s, self.xyz[:, k])
        return T

    def xy_yaw(self, stamps):
        s = (np.asarray(stamps, np.int64) - self.t[0]) / 1e9
        return np.interp(s, self.s, self.xyz[:, 0]), np.interp(s, self.s, self.xyz[:, 1]), np.interp(s, self.s, self.yaw)


def car_state_fast(path):
    src = bag.state_source(path)
    out = []
    f, it = bag._reader(path, topics=[src], log_time_order=True)
    with f:
        for _, _, _, msg in it:
            if src == bag.STATE:
                out.append((bag.stamp_of(msg), msg.vx_mps, msg.vy_mps, msg.wz_radps))
            else:
                out.append((bag.stamp_of(msg), msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z))
    a = np.array(out)
    return a[np.argsort(a[:, 0], kind='stable')]


def motion_calib(gt, cs, laps, lap_ids):
    sel = np.zeros(len(cs), bool)
    t0 = gt.t[0]
    for L in lap_ids:
        r = laps[laps[:, 0] == L][0]
        sel |= (cs[:, 0] > t0 + r[1] * 1e9) & (cs[:, 0] < t0 + r[2] * 1e9)
    c = cs[sel][::3]
    t = c[:, 0].astype(np.int64)
    h = 20_000_000
    x0, y0, a0 = gt.xy_yaw(t - h)
    x1, y1, a1 = gt.xy_yaw(t + h)
    vxm, vym, a = (x1 - x0) / 0.04, (y1 - y0) / 0.04, (a0 + a1) / 2
    vxb = np.cos(a) * vxm + np.sin(a) * vym
    vyb = -np.sin(a) * vxm + np.cos(a) * vym
    wgt = (a1 - a0) / 0.04
    vx, vy, wz = c[:, 1], c[:, 2], c[:, 3]
    m = vx > 5
    (ay, by, cy), *_ = np.linalg.lstsq(np.c_[vy[m], wz[m], np.ones(m.sum())], vyb[m], rcond=None)
    (sw, dw), *_ = np.linalg.lstsq(np.c_[wz[m], np.ones(m.sum())], wgt[m], rcond=None)
    k = float(np.sum(vx[m] * vxb[m]) / np.sum(vx[m] ** 2))
    r0 = np.sqrt(np.mean((vyb[m] - vy[m]) ** 2))
    r1 = np.sqrt(np.mean((vyb[m] - (ay * vy[m] + by * wz[m] + cy)) ** 2))
    return dict(vy_a=float(ay), vy_b=float(by), vy_c=float(cy), wz_s=float(sw), wz_d=float(dw), vx_k=k,
                laps=[int(L) for L in lap_ids], n=int(m.sum()), vy_resid_rms_raw=float(r0),
                vy_resid_rms_calib=float(r1))


def night_stats(path, kf):
    masks = {c: cv2.imread(os.path.join(MASKS, f'{c}.png'), cv2.IMREAD_GRAYSCALE) for c in CAMS}
    med = {c: [] for c in CAMS}

    def work(c, jpg):
        g = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if g is None:
            return
        g = cv2.resize(g, (W, H)) if g.shape != (H, W) else g
        med[c].append(float(np.median(g[masks[c] > 0])))

    jobs = []
    f, it = bag._reader(path, topics=list(bag.IMG_T), log_time_order=True)
    with f, ThreadPoolExecutor(16) as pool:
        for _, ch, _, msg in it:
            s = bag.stamp_of(msg)
            i = int(np.searchsorted(kf, s))
            if any(0 <= j < len(kf) and abs(int(kf[j]) - s) < bag.SET_TOL_NS for j in (i - 1, i)):
                jobs.append(pool.submit(work, bag.IMG_T[ch.topic], bytes(msg.data)))
        for j in jobs:
            j.result()
    return {c: dict(median_grey=float(np.median(med[c])), p10=float(np.percentile(med[c], 10)),
                    p90=float(np.percentile(med[c], 90))) for c in CAMS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mcap', required=True, help='training run (with GT /tf)')
    ap.add_argument('--laps', type=int, nargs='+', help='laps for the motion calibration (default: all complete laps '
                                                        'of configs/laps.csv)')
    ap.add_argument('--out', default=MODEL)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    laps = np.genfromtxt(os.path.join(CONFIGS, 'laps.csv'), delimiter=',', names=True)
    L = np.c_[laps['lap'], laps['t_start'], laps['t_end'], laps['complete']]
    lap_ids = a.laps or [int(r[0]) for r in L if r[3]]

    kf = np.load(os.path.join(FEAT, 'db_fl.npz'))['kf_stamp'].astype(np.int64)
    gt = GT(a.mcap)
    x, y, yaw = gt.xy_yaw(kf)
    p = os.path.join(a.out, 'kf_poses.npz')
    np.savez(fresh(p), kf_stamp=kf, T=gt.T(kf), x=x, y=y, yaw=np.angle(np.exp(1j * yaw)))
    os.replace(fresh(p), p)
    print(f'kf_poses: {len(kf)} keyframes', flush=True)

    mc = motion_calib(gt, car_state_fast(a.mcap), L, lap_ids)
    p = os.path.join(a.out, 'motion_calib.json')
    json.dump(mc, open(fresh(p), 'w'), indent=1); os.replace(fresh(p), p)
    print('motion_calib:', json.dumps(mc), flush=True)

    ns = night_stats(a.mcap, kf)
    p = os.path.join(a.out, 'night_stats.json')
    json.dump(ns, open(fresh(p), 'w'), indent=1); os.replace(fresh(p), p)
    print('night_stats:', json.dumps(ns), flush=True)

    _, _, bounds, src = bag.read_static(a.mcap)
    if src == 'bag':
        p = os.path.join(a.out, 'track_bounds.npz')
        np.savez(fresh(p), **bounds); os.replace(fresh(p), p)
        print('track_bounds:', {k: v.shape for k, v in bounds.items()})


if __name__ == '__main__':
    main()
