#!/usr/bin/env python3
"""Training-time artefacts of v2 (use GT; run once):

    python3 v2/build_model.py
      -> v2/model/kf_poses.npz      T_map_base (K,4,4), x, y, yaw of every map keyframe (GT at the keyframe stamp).
                                    Same keyframes as features/db_*.npz; the map already depends on these poses
                                    (points were triangulated with them), v2 also uses them to pick the keyframe
                                    nearest to the filter's predicted pose instead of DINOv2 retrieval.
      -> v2/model/motion_calib.json car-state -> base_link motion: vy_eff = a*vy + b*wz + c, wz_eff = s*wz + d,
                                    vx_eff = k*vx; least squares against GT body-frame velocity, DEVELOPMENT laps only
                                    (K1, K2, K4, K5, K7; held-out K3, K6 never touched).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from camera_model import GTPoses  # noqa: E402
from car_state import load  # noqa: E402
from eval_single import DEV_LAPS, lap_table  # noqa: E402


def main():
    os.makedirs(os.path.join(HERE, 'model'), exist_ok=True)
    gt = GTPoses()
    kf = np.load(os.path.join(ROOT, 'features', 'db_fl.npz'))['kf_stamp'].astype(np.int64)
    T = gt.T(kf)
    x, y, yaw = gt.xy_yaw(kf)
    np.savez(os.path.join(HERE, 'model', 'kf_poses.npz'), kf_stamp=kf, T=T, x=x, y=y, yaw=np.angle(np.exp(1j * yaw)))
    print(f'kf_poses: {len(kf)} keyframes')

    cs, laps = load(), lap_table()
    dev = np.zeros(len(cs), bool)
    for L in DEV_LAPS:
        dev |= (cs.stamp_ns.values > laps.a[L]) & (cs.stamp_ns.values < laps.b[L])
    c = cs[dev].iloc[::3]
    t = c.stamp_ns.values
    h = 20_000_000
    x0, y0, a0 = gt.xy_yaw(t - h)
    x1, y1, a1 = gt.xy_yaw(t + h)
    vxm, vym, a = (x1 - x0) / 0.04, (y1 - y0) / 0.04, (a0 + a1) / 2
    vxb = np.cos(a) * vxm + np.sin(a) * vym
    vyb = -np.sin(a) * vxm + np.cos(a) * vym
    wgt = (a1 - a0) / 0.04
    vx, vy, wz = c.vx_mps.values, c.vy_mps.values, c.wz_radps.values
    m = vx > 5
    (ay, by, cy), *_ = np.linalg.lstsq(np.c_[vy[m], wz[m], np.ones(m.sum())], vyb[m], rcond=None)
    (sw, dw), *_ = np.linalg.lstsq(np.c_[wz[m], np.ones(m.sum())], wgt[m], rcond=None)
    k = float(np.sum(vx[m] * vxb[m]) / np.sum(vx[m] ** 2))
    r0 = np.sqrt(np.mean((vyb[m] - vy[m]) ** 2))
    r1 = np.sqrt(np.mean((vyb[m] - (ay * vy[m] + by * wz[m] + cy)) ** 2))
    out = dict(vy_a=float(ay), vy_b=float(by), vy_c=float(cy), wz_s=float(sw), wz_d=float(dw), vx_k=k,
               laps=DEV_LAPS, n=int(m.sum()), vy_resid_rms_raw=float(r0), vy_resid_rms_calib=float(r1))
    json.dump(out, open(os.path.join(HERE, 'model', 'motion_calib.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
