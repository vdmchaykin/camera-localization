#!/usr/bin/env python3
"""Step 2: first look at the data. Reads export/ and gt.csv, writes figures + stats.json into analysis/.

    python3 analysis/analyze.py
"""
import glob, json, os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, 'export')
OUT = os.path.join(ROOT, 'analysis')
CAMS = ['fl', 'fr', 'cl', 'cr', 'rl', 'rr', 'r']
STATS = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches='tight')
    plt.close(fig)


def resample(p, step):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u = np.arange(0, s[-1], step)
    return np.stack([np.interp(u, s, p[:, k]) for k in range(p.shape[1])], 1)


# ---------------------------------------------------------------- load
gt = pd.read_csv(os.path.join(EX, 'gt_tf.csv')).sort_values('timestamp_ns').reset_index(drop=True)
se = pd.read_csv(os.path.join(EX, 'state_estimation.csv'))
se['stamp_ns'] = se['header.stamp.sec'].astype(np.int64) * 10**9 + se['header.stamp.nanosec'].astype(np.int64)
se = se.sort_values('stamp_ns').reset_index(drop=True)
img = pd.read_csv(os.path.join(EX, 'image_stamps.csv'))
bounds = {k: np.array(v['points']) for k, v in json.load(open(os.path.join(EX, 'track_bounds.json'))).items()}
tfs = json.load(open(os.path.join(EX, 'tf_static.json')))
cinfo = json.load(open(os.path.join(EX, 'camera_info.json')))
T0 = gt.timestamp_ns.iloc[0]
gt['t'] = (gt.timestamp_ns - T0) / 1e9
se['t'] = (se.stamp_ns - T0) / 1e9

# ---------------------------------------------------------------- 1. track geometry
L = resample(bounds['track_left'][:, :2], 0.5)
R = resample(bounds['track_right'][:, :2], 0.5)
dR, iR = cKDTree(R).query(L)
C = (L + R[iR]) / 2
# smooth + resample centre line (closed)
k = 15
Cp = np.vstack([C[-k:], C, C[:k]])
C = np.stack([np.convolve(Cp[:, j], np.ones(2 * k + 1) / (2 * k + 1), 'valid') for j in range(2)], 1)
C = resample(np.vstack([C, C[:1]]), 0.5)
seg = np.diff(np.vstack([C, C[:1]]), axis=0)
S = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))[:-1]]
LEN = S[-1] + np.linalg.norm(seg[-1])
HEAD = np.arctan2(seg[:, 1], seg[:, 0])
widthL = cKDTree(L).query(C)[0]
widthR = cKDTree(R).query(C)[0]
width = widthL + widthR
np.savez(os.path.join(OUT, 'centerline.npz'), xy=C, s=S, heading=HEAD, width_left=widthL,
         width_right=widthR, length=LEN)
STATS['track'] = dict(centerline_length_m=round(LEN, 1),
                      left_len_m=round(np.sum(np.linalg.norm(np.diff(bounds['track_left'][:, :2], axis=0), axis=1)), 1),
                      right_len_m=round(np.sum(np.linalg.norm(np.diff(bounds['track_right'][:, :2], axis=0), axis=1)), 1),
                      width_m=dict(min=round(width.min(), 2), median=round(np.median(width), 2),
                                   max=round(width.max(), 2)),
                      z_range_m=[round(min(b[:, 2].min() for b in bounds.values()), 2),
                                 round(max(b[:, 2].max() for b in bounds.values()), 2)])

# project GT on centre line -> s, n, relative heading
tree = cKDTree(C)
_, ic = tree.query(gt[['x', 'y']].values)
d = gt[['x', 'y']].values - C[ic]
gt['s'] = S[ic] + d[:, 0] * np.cos(HEAD[ic]) + d[:, 1] * np.sin(HEAD[ic])
gt['n'] = -d[:, 0] * np.sin(HEAD[ic]) + d[:, 1] * np.cos(HEAD[ic])   # + = left
gt['dpsi'] = np.angle(np.exp(1j * (gt.yaw - HEAD[ic])))
gt['halfw_l'], gt['halfw_r'] = widthL[ic], widthR[ic]
# speed from GT (tf) differences
dt = np.diff(gt.t.values)
v = np.hypot(np.diff(gt.x), np.diff(gt.y)) / np.maximum(dt, 1e-3)
gt['v'] = np.r_[v[0], v]
gt['v'] = gt.v.rolling(11, center=True, min_periods=1).median()
# pit: GT-provided lane flag (0 = pit lane) mapped by stamp
lane = se.set_index('stamp_ns')['sn_map_state.current_lane']
gt['pit'] = np.interp(gt.timestamp_ns, se.stamp_ns, se['sn_map_state.current_lane']) < 0.5
gt['off_track'] = (gt.n > gt.halfw_l) | (-gt.n > gt.halfw_r)

fig, ax = plt.subplots(figsize=(12, 10))
for kk, b in bounds.items():
    ax.plot(b[:, 0], b[:, 1], 'k-' if 'track' in kk else 'b-', lw=0.6, label=kk)
sc = ax.scatter(gt.x[::20], gt.y[::20], c=gt.v[::20], s=1.5, cmap='turbo')
plt.colorbar(sc, ax=ax, label='speed [m/s]')
ax.plot(C[0, 0], C[0, 1], 'r*', ms=14, label='s = 0 (centre line start)')
ax.plot(gt.x[0], gt.y[0], 'go', ms=8, label='first GT pose')
ax.annotate('', xy=C[40], xytext=C[0], arrowprops=dict(color='r', width=2))
ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=8); ax.set_title('Track bounds + GT trajectory')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
save(fig, '01_track_trajectory.png')

fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].plot(S, width, lw=0.8); ax[0].set_ylabel('track width [m]'); ax[0].set_xlabel('s [m]')
ax[1].hist(width, 60); ax[1].set_xlabel('track width [m]')
fig.tight_layout(); save(fig, '02_track_width.png')

# ---------------------------------------------------------------- 2. laps
# laps start at the official start line: where the GT racing-line index (sn_map_state ... idx) wraps to 0
# (INTRO slide 6: s is measured from the start line). Lap 0 = out-lap from the pits, last = in-lap.
ref_idx = se['sn_map_state.track_sn_state.sn_state.idx'].values.astype(np.int64)
wrap_ns = se.stamp_ns.values[1:][np.diff(ref_idx) < -1000]
cross = np.searchsorted(gt.timestamp_ns.values, wrap_ns)
k_sl = np.searchsorted(se.stamp_ns.values, wrap_ns[0])
STATS['start_line_xy'] = [round(float(se.x_m[k_sl]), 1), round(float(se.y_m[k_sl]), 1)]
STATS['racing_line_length_m'] = round(float((ref_idx.max() + 1) * np.median(
    (se['sn_map_state.track_sn_state.sn_state.ds'] / se['sn_map_state.track_sn_state.sn_state.d_idx'])[
        se['sn_map_state.track_sn_state.sn_state.d_idx'] > 0.05])), 0)
edges = np.r_[0, cross, len(gt) - 1]
laps = []
for a, b in zip(edges[:-1], edges[1:]):
    g = gt.iloc[a:b]
    laps.append(dict(lap=len(laps), t_start=round(g.t.iloc[0], 1), t_end=round(g.t.iloc[-1], 1),
                     duration_s=round(g.t.iloc[-1] - g.t.iloc[0], 1),
                     complete=bool(0 < a and b < len(gt) - 1),
                     v_max=round(g.v.max(), 1), v_mean=round(g.v.mean(), 1),
                     pit_s=round(g.pit.sum() / 106, 1),
                     stopped_s=round((g.v < 0.5).sum() / 106, 1)))
lapdf = pd.DataFrame(laps)
lapdf.to_csv(os.path.join(OUT, 'laps.csv'), index=False)
STATS['laps'] = laps
# stops and pit intervals
def intervals(mask, t, min_len=0.5):
    m = np.r_[False, mask, False].astype(int)
    st, en = np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0] - 1
    return [(round(t[a], 1), round(t[b], 1)) for a, b in zip(st, en) if t[b] - t[a] >= min_len]
STATS['stops'] = intervals(gt.v.values < 0.5, gt.t.values)
STATS['pit_intervals'] = intervals(gt.pit.values, gt.t.values)
STATS['off_track_share'] = round(float(gt.off_track[~gt.pit].mean()), 4)

fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
ax[0].plot(gt.t, gt.v, lw=0.6); ax[0].set_ylabel('v [m/s]')
for a, b in STATS['pit_intervals']:
    for x in ax: x.axvspan(a, b, color='orange', alpha=0.25)
for c in cross:
    for x in ax: x.axvline(gt.t[c], color='gray', lw=0.5)
ax[1].plot(gt.t, gt.s, lw=0.6); ax[1].set_ylabel('s [m]')
ax[2].plot(gt.t, gt.n, lw=0.5, label='n (lateral, +left)')
ax[2].plot(gt.t, gt.halfw_l, 'k', lw=0.4); ax[2].plot(gt.t, -gt.halfw_r, 'k', lw=0.4)
ax[2].set_ylim(-15, 15); ax[2].set_ylabel('n [m]'); ax[2].set_xlabel('t [s]')
ax[0].set_title('Speed, progress, lateral offset (orange = pit lane, grey = lap line)')
fig.tight_layout(); save(fig, '03_laps_timeline.png')

# ---------------------------------------------------------------- 3. dynamics
fig, ax = plt.subplots(2, 4, figsize=(18, 7))
for j, (c, lab) in enumerate([('v_mps', 'v [m/s]'), ('ax_mps2', 'ax [m/s²]'), ('ay_mps2', 'ay [m/s²]'),
                              ('wz_radps', 'yaw rate [rad/s]')]):
    ax[0, j].plot(se.t, se[c], lw=0.4); ax[0, j].set_title(lab)
    ax[1, j].hist(se[c], 80); ax[1, j].set_yscale('log')
fig.tight_layout(); save(fig, '04_dynamics.png')
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
tr = ~gt.pit
ax[0].hist(gt.n[tr], 100); ax[0].set_xlabel('lateral offset n from centre line [m] (track only)')
ax[1].hist((gt.halfw_l - gt.n)[tr], 100, alpha=0.6, label='to left border')
ax[1].hist((gt.halfw_r + gt.n)[tr], 100, alpha=0.6, label='to right border')
ax[1].set_xlabel('distance to border [m]'); ax[1].legend()
save(fig, '05_lateral_offset.png')
STATS['dynamics'] = {c: dict(min=round(se[c].min(), 2), p50=round(se[c].median(), 2), max=round(se[c].max(), 2))
                     for c in ['v_mps', 'ax_mps2', 'ay_mps2', 'wz_radps', 'roll_rad', 'pitch_rad', 'z_m']}
STATS['dynamics']['dpsi_deg_track'] = dict(p5=round(np.degrees(np.percentile(gt.dpsi[tr], 5)), 2),
                                           p95=round(np.degrees(np.percentile(gt.dpsi[tr], 95)), 2))

# ---------------------------------------------------------------- 4. GT quality
dts = np.diff(gt.timestamp_ns.values) / 1e6
gaps = np.where(dts > 50)[0]
STATS['gt'] = dict(n=len(gt), dt_ms_median=round(float(np.median(dts)), 2), dt_ms_max=round(float(dts.max()), 1),
                   gaps_gt_50ms=int(len(gaps)), gap_total_s=round(float(dts[gaps].sum() / 1e3), 2),
                   gaps=[(round(gt.t[i], 1), round(dts[i], 0)) for i in gaps[:30]],
                   duplicated_stamps=int(gt.timestamp_ns.duplicated().sum()))
m = pd.merge_asof(gt[['timestamp_ns', 'x', 'y', 'yaw']], se[['stamp_ns', 'x_m', 'y_m', 'yaw_rad', 'se_status',
                                                             'pos_accuracy[0]', 'pos_accuracy[1]']],
                  left_on='timestamp_ns', right_on='stamp_ns', direction='nearest')
m['dpos'] = np.hypot(m.x - m.x_m, m.y - m.y_m)
m['dyaw'] = np.degrees(np.angle(np.exp(1j * (m.yaw - m.yaw_rad))))
m['dstamp_ms'] = (m.timestamp_ns - m.stamp_ns) / 1e6
STATS['tf_vs_se'] = dict(same_stamp_share=round(float((m.dstamp_ms == 0).mean()), 4),
                         dpos_m_p50=round(float(m.dpos.median()), 4), dpos_m_max=round(float(m.dpos.max()), 3),
                         dyaw_deg_max=round(float(m.dyaw.abs().max()), 3))
st1 = se[se.se_status == 1]
STATS['se_status_1'] = intervals((se.se_status == 1).values, se.t.values, 0.0)
fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
ax[0].plot(gt.t[1:], dts, lw=0.5); ax[0].set_ylabel('GT dt [ms]'); ax[0].axhline(50, color='r')
ax[1].semilogy(se.t, se['pos_accuracy[0]'], lw=0.5, label='pos_accuracy x')
ax[1].semilogy(se.t, se['pos_accuracy[1]'], lw=0.5, label='y'); ax[1].legend()
ax[2].plot(se.t, se.se_status, lw=0.5); ax[2].set_ylabel('se_status'); ax[2].set_xlabel('t [s]')
fig.tight_layout(); save(fig, '06_gt_quality.png')

# ---------------------------------------------------------------- 5. cameras & time
cam_stats = {}
fig, ax = plt.subplots(3, 1, figsize=(14, 10))
ref = np.sort(img[img.cam == 'fl'].stamp_ns.values)
for cam in CAMS:
    c = img[img.cam == cam].sort_values('stamp_ns')
    d = np.diff(c.stamp_ns.values) / 1e6
    j = np.searchsorted(ref, c.stamp_ns.values).clip(1, len(ref) - 1)
    off = c.stamp_ns.values - np.where(np.abs(ref[j] - c.stamp_ns.values) < np.abs(ref[j - 1] - c.stamp_ns.values),
                                       ref[j], ref[j - 1])
    lat = (c.log_ns - c.stamp_ns).values / 1e6
    cam_stats[cam] = dict(n=len(c), dt_ms_median=round(float(np.median(d)), 2), dt_ms_max=round(float(d.max()), 1),
                          drops=int((d > 150).sum()), drop_at_s=[round((c.stamp_ns.values[i] - T0) / 1e9, 1)
                                                                  for i in np.where(d > 150)[0]],
                          offset_to_fl_ms=dict(p50=round(float(np.median(off)) / 1e6, 2),
                                               p1=round(float(np.percentile(off, 1)) / 1e6, 2),
                                               p99=round(float(np.percentile(off, 99)) / 1e6, 2)),
                          log_latency_ms=dict(p50=round(float(np.median(lat)), 1), p99=round(float(np.percentile(lat, 99)), 1)),
                          jpeg_kb_median=round(float(c.bytes.median() / 1024), 1))
    tt = (c.stamp_ns.values - T0) / 1e9
    ax[0].plot(tt, pd.Series(c.bytes.values / 1024).rolling(50, min_periods=1).mean(), lw=0.7, label=cam)
    ax[1].plot(tt, off / 1e6, '.', ms=0.5, label=cam)
    ax[2].hist(lat, 100, histtype='step', label=cam)
ax[0].set_ylabel('jpeg size [kB] (5 s mean)'); ax[0].legend(ncol=7)
ax[1].set_ylabel('stamp - nearest fl stamp [ms]'); ax[1].set_ylim(-60, 60)
ax[2].set_xlabel('log_time - stamp [ms]'); ax[2].legend()
fig.tight_layout(); save(fig, '07_camera_timing.png')
STATS['cameras'] = cam_stats

# ---------------------------------------------------------------- 6. calibration
def tf_mat(t):
    T = np.eye(4)
    q = t['rotation']
    T[:3, :3] = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    T[:3, 3] = [t['translation'][k] for k in 'xyz']
    return T

T_bc = {cam: tf_mat(tfs[f'constructor0/base_link -> constructor0/camera_{cam}']) for cam in CAMS}
calib = {}
fig, ax = plt.subplots(figsize=(9, 9))
ax.add_patch(plt.Rectangle((-3.0, -0.95), 4.9, 1.9, fill=False))
for cam in CAMS:
    K = np.array(cinfo[cam]['k']).reshape(3, 3)
    w, h = cinfo[cam]['width'], cinfo[cam]['height']
    hfov = np.degrees(2 * np.arctan(w / 2 / K[0, 0])); vfov = np.degrees(2 * np.arctan(h / 2 / K[1, 1]))
    T = T_bc[cam]
    z_ax = T[:3, 2]; x_ax = T[:3, 0]
    yaw = np.degrees(np.arctan2(z_ax[1], z_ax[0])); pitch = np.degrees(np.arcsin(-z_ax[2]))
    roll = np.degrees(np.arctan2(-x_ax[2], -T[:3, 1][2]))  # 0 when image y points down
    calib[cam] = dict(fx=round(K[0, 0], 1), fy=round(K[1, 1], 1), fy_over_fx=round(K[1, 1] / K[0, 0], 3),
                      hfov_deg=round(hfov, 1), vfov_deg=round(vfov, 1), yaw_deg=round(yaw, 2),
                      pitch_down_deg=round(pitch, 2), roll_deg=round(roll, 2),
                      xyz=[round(v, 3) for v in T[:3, 3]], D=[round(v, 4) for v in cinfo[cam]['d']])
    p = T[:3, 3]
    for a in (-hfov / 2, hfov / 2):
        r = np.radians(yaw + a)
        ax.plot([p[0], p[0] + 6 * np.cos(r)], [p[1], p[1] + 6 * np.sin(r)], lw=0.6)
    r = np.radians(yaw)
    ax.annotate(cam, (p[0] + 4 * np.cos(r), p[1] + 4 * np.sin(r)), ha='center')
    ax.plot(*p[:2], 'ko', ms=3)
ax.set_aspect('equal'); ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.grid(alpha=0.3)
ax.set_title('Camera layout, top view (base_link: x fwd, y left), horizontal FOV')
save(fig, '08_camera_layout.png')
STATS['calibration'] = calib


def gt_pose_T(stamp):
    i = int(np.clip(np.searchsorted(gt.timestamp_ns.values, stamp), 1, len(gt) - 1))
    a = (stamp - gt.timestamp_ns[i - 1]) / (gt.timestamp_ns[i] - gt.timestamp_ns[i - 1])
    r = gt.iloc[i - 1] if a < 0.5 else gt.iloc[i]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
    T[:3, 3] = (1 - a) * gt.loc[i - 1, ['x', 'y', 'z']].values + a * gt.loc[i, ['x', 'y', 'z']].values
    return T


def project(cam, stamp, pts):
    T = gt_pose_T(stamp) @ T_bc[cam]
    pc = (pts - T[:3, 3]) @ T[:3, :3]
    K = np.array(cinfo[cam]['k']).reshape(3, 3)
    D = np.array(cinfo[cam]['d'])
    w, h = cinfo[cam]['width'], cinfo[cam]['height']
    ok = (pc[:, 2] > 1.0) & (np.linalg.norm(pc, axis=1) < 150)
    ok &= (np.abs(pc[:, 0] / np.maximum(pc[:, 2], 1e-3)) < 1.1 * w / 2 / K[0, 0])
    ok &= (np.abs(pc[:, 1] / np.maximum(pc[:, 2], 1e-3)) < 1.1 * h / 2 / K[1, 1])
    uv = np.full((len(pts), 2), np.nan)
    if ok.any():
        uv[ok] = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return uv


COLORS = {'track_left': (0, 0, 255), 'track_right': (0, 255, 0), 'pitlane_left': (255, 0, 255),
          'pitlane_right': (255, 255, 0)}
bounds3 = {k: resample(b, 0.25) for k, b in bounds.items()}
proj_dir = os.path.join(OUT, 'projections'); os.makedirs(proj_dir, exist_ok=True)
samples = {cam: sorted(glob.glob(os.path.join(EX, 'samples', cam, '*.jpg'))) for cam in CAMS}
pick = [1, 4, 8, 12]   # sample indices (~60 s apart)
brightness = {cam: [] for cam in CAMS}
for cam in CAMS:
    for f in samples[cam]:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        brightness[cam].append((int(os.path.basename(f)[:-4]), float(im.mean()), float((im > 250).mean())))
for k in pick:
    tiles = []
    for cam in CAMS:
        f = samples[cam][min(k, len(samples[cam]) - 1)]
        stamp = int(os.path.basename(f)[:-4])
        im = cv2.imread(f)
        for nm, b in bounds3.items():
            uv = project(cam, stamp, b)
            for p in uv[~np.isnan(uv[:, 0])]:
                cv2.circle(im, (int(p[0]), int(p[1])), 2, COLORS[nm], -1)
        cv2.putText(im, f'{cam}  t={(stamp - T0) / 1e9:.1f}s', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 3)
        cv2.imwrite(os.path.join(proj_dir, f'{k:02d}_{cam}.jpg'), im)
        tiles.append(cv2.resize(im, (753, 364)))
    tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
    cv2.imwrite(os.path.join(OUT, f'09_projection_sample{k:02d}.jpg'), sheet)

# ---------------------------------------------------------------- 7. frames: contact sheet + brightness
rows = []
for k in range(0, len(samples['fl']), 3):
    tiles = []
    for cam in CAMS:
        f = samples[cam][min(k, len(samples[cam]) - 1)]
        im = cv2.resize(cv2.imread(f), (377, 182))
        cv2.putText(im, f'{cam} {(int(os.path.basename(f)[:-4]) - T0) / 1e9:.0f}s', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        tiles.append(im)
    rows.append(np.hstack(tiles))
cv2.imwrite(os.path.join(OUT, '10_contact_sheet.jpg'), np.vstack(rows))
STATS['brightness'] = {cam: dict(mean_gray=round(np.mean([b[1] for b in v]), 1),
                                 saturated_share=round(np.mean([b[2] for b in v]), 4)) for cam, v in brightness.items()}

with open(os.path.join(OUT, 'stats.json'), 'w') as f:
    json.dump(STATS, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(STATS, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
