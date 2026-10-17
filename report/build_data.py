"""Collect data + downsized images for the report artifact -> report/data.json, report/img/."""
import json, os, glob
import cv2, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'report')
S = json.load(open(f'{R}/analysis/stats.json'))
gt = pd.read_csv(f'{R}/export/gt_tf.csv').sort_values('timestamp_ns')
se = pd.read_csv(f'{R}/export/state_estimation.csv')
se['st'] = se['header.stamp.sec'].astype(np.int64) * 10**9 + se['header.stamp.nanosec']
T0 = gt.timestamp_ns.iloc[0]
cl = np.load(f'{R}/analysis/centerline.npz')
C, Sx, HEAD, LEN = cl['xy'], cl['s'], cl['heading'], float(cl['length'])
from scipy.spatial import cKDTree
t = np.arange(0, (gt.timestamp_ns.iloc[-1] - T0) / 1e9, 0.5)
tg = (gt.timestamp_ns.values - T0) / 1e9
x, y = np.interp(t, tg, gt.x), np.interp(t, tg, gt.y)
yaw = np.interp(t, tg, np.unwrap(gt.yaw))
v = np.interp(t, (se.st.values - T0) / 1e9, se.v_mps)
pit = np.interp(t, (se.st.values - T0) / 1e9, se['sn_map_state.current_lane']) < 0.5
_, ic = cKDTree(C).query(np.c_[x, y])
d = np.c_[x, y] - C[ic]
s = (Sx[ic] + d[:, 0] * np.cos(HEAD[ic]) + d[:, 1] * np.sin(HEAD[ic])) % LEN
n = -d[:, 0] * np.sin(HEAD[ic]) + d[:, 1] * np.cos(HEAD[ic])
laps = pd.read_csv(f'{R}/analysis/laps.csv')
lap = np.clip(np.searchsorted(laps.t_start.values, t, side='right') - 1, 0, len(laps) - 1)
bounds = json.load(open(f'{R}/export/track_bounds.json'))
def dec(p, step=3.0):
    p = np.array(p)[:, :2]; keep = [0]
    for i in range(1, len(p)):
        if np.hypot(*(p[i] - p[keep[-1]])) >= step: keep.append(i)
    return np.round(p[keep], 1).tolist()
data = dict(
    traj=dict(t=np.round(t, 1).tolist(), x=np.round(x, 1).tolist(), y=np.round(y, 1).tolist(),
              v=np.round(v, 1).tolist(), s=np.round(s, 0).tolist(), n=np.round(n, 2).tolist(),
              yaw=np.round(np.degrees(np.angle(np.exp(1j * yaw))), 1).tolist(), pit=pit.astype(int).tolist(),
              lap=lap.tolist()),
    bounds={k: dec(v_['points']) for k, v_ in bounds.items()},
    laps=laps.to_dict('records'), stats=S,
    retr=json.load(open(f'{R}/results/retrieval_dev.json')),
    drift=json.load(open(f'{R}/results/dr_drift.json')),
    pose={k: json.load(open(f'{R}/results/pose_dev_{k}.json')) for k in ('flfr', 'flfrcl', '5cam', '7cam')
          if os.path.exists(f'{R}/results/pose_dev_{k}.json')},
    qa=None,
)


def md_inline(s):
    import html, re
    s = html.escape(s, quote=False)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)


def md_block(lines):
    out, ul = [], False
    for ln in lines:
        if ln.startswith('- '):
            if not ul: out.append('<ul>'); ul = True
            out.append(f'<li>{md_inline(ln[2:])}</li>')
        elif ln.startswith('  ') and ul and out:
            out[-1] = out[-1][:-5] + ' ' + md_inline(ln.strip()) + '</li>'
        elif ln.strip():
            if ul: out.append('</ul>'); ul = False
            out.append(f'<p>{md_inline(ln)}</p>')
    if ul: out.append('</ul>')
    return ''.join(out)


def parse_qa(path):
    """QA.md -> list of {n, title, prio, status, group, body, assume, answer} (HTML snippets)."""
    import re
    items, group = [], ''
    for chunk in re.split(r'\n(?=## |### )', open(path).read()):
        if chunk.startswith('## '):
            group = chunk.splitlines()[0][3:].strip()
            continue
        if not chunk.startswith('### '):
            continue
        lines = chunk.splitlines()
        m = re.match(r'### (\d+)\. (.+?) · (P\d) · (\S+)', lines[0])
        if not m:
            continue
        body, assume, answer = [], '', ''
        for ln in lines[1:]:
            if ln.startswith('**Пока считаем:**'):
                assume = md_inline(ln.split(':**', 1)[1].strip())
            elif ln.startswith('**Ответ:**'):
                answer = md_inline(ln.split(':**', 1)[1].strip())
            elif not ln.startswith('---'):
                body.append(ln)
        items.append(dict(n=int(m[1]), title=md_inline(m[2]), prio=m[3], status=m[4], group=group,
                          body=md_block(body), assume=assume, answer='' if answer in ('—', '-') else answer))
    return items


data['qa'] = parse_qa(f'{R}/QA.md')


def eval_txt(path):
    """evaluate.py output -> dict of the main numbers."""
    out = {}
    for ln in open(path):
        v = ln.split()
        if ln.startswith('position RMSE'): out['rmse'] = float(v[3])
        elif ln.startswith('position median/p95'): out['med'], out['p95'] = float(v[2]), float(v[4])
        elif ln.startswith('lateral RMSE'): out['lat'] = float(v[3])
        elif ln.startswith('longitudinal RMSE'): out['lon'] = float(v[3])
        elif ln.startswith('yaw RMSE'): out['yaw'] = float(v[3])
        elif ln.startswith('within 1.00'): out['lt1'] = float(v[3])
        elif ln.startswith('within 0.25'): out['lt025'] = float(v[3])
        elif ln.startswith('within 0.50'): out['lt05'] = float(v[3])
        elif ln.startswith('coverage'): out['cov'] = float(v[4])
        elif ln.startswith('poses submitted'): out['n'] = int(v[2])
    return out


def filter_data():
    """per measurement set (tag) and lap: errors over time (5 Hz) + every offered measurement."""
    import sys
    sys.path.insert(0, R)
    from camera_model import GTPoses
    from eval_single import errors
    gtp = GTPoses()
    res = {}
    fast = 'single_dev_7cam_k1_512_2hz.csv'
    for tag, single in (('flfr_1hz', 'single_dev_flfr.csv'), ('flfrcl_1hz', 'single_dev_flfrcl.csv'),
                        ('5cam_1hz', 'single_dev_5cam.csv'), ('7cam_1hz', 'single_dev_7cam.csv'),
                        ('7cam_fast_p0.5', fast), ('7cam_fast_p1', fast), ('7cam_fast_p2', fast), ('dr', None)):
        laps_ = {}
        for lap in (1, 5):
            f = f'{R}/results/filter_{tag}_lap{lap}.csv'
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)[::2]
            pos, lat, lon, dyaw = errors(gtp, d.timestamp_ns.values, d.x.values, d.y.values, d.yaw.values)
            e = dict(t=np.round((d.timestamp_ns.values - T0) / 1e9, 1).tolist(), err=np.round(pos, 3).tolist(),
                     lat=np.round(lat, 3).tolist(), lon=np.round(lon, 3).tolist(),
                     metrics=eval_txt(f.replace('.csv', '.txt')))
            mf = f.replace('.csv', '_meas.csv')
            if single and os.path.exists(mf):
                mm = pd.read_csv(mf)
                sg = pd.read_csv(f'{R}/results/{single}').set_index('stamp')
                e['meas'] = dict(t=np.round((mm.stamp.values - T0) / 1e9, 1).tolist(),
                                 ok=mm.accepted.astype(int).tolist(), inl=mm.inliers.astype(int).tolist(),
                                 err=np.round(sg.loc[mm.stamp.values, 'err'].values, 2).tolist())
            laps_[lap] = e
        if laps_:
            res[tag] = laps_
    return res


def db_progress():
    import re
    done, cur = [], None
    for ln in open(f'{R}/db_build.log'):
        m = re.match(r'(\w+): (\d+) points, (\d+)/kf, (\d+) s', ln)
        if m:
            done.append(dict(cam=m[1], points=int(m[2]), per_kf=int(m[3]), sec=int(m[4])))
        m = re.match(r'(\w+) (\d+)/(\d+) pts (\d+)', ln)
        if m:
            cur = dict(cam=m[1], kf=int(m[2]), total=int(m[3]), points=int(m[4]))
    if cur and any(dn['cam'] == cur['cam'] for dn in done):
        cur = None
    return dict(done=done, current=cur)


data['filter'] = filter_data()
data['bench'] = pd.read_csv(f'{R}/results/bench_runtime.csv').to_dict('records')
if os.path.exists(f'{R}/results/val.json'):
    data['val'] = json.load(open(f'{R}/results/val.json'))
    data['val_dr'] = {lap: eval_txt(f'{R}/results/filter_dr_val_lap{lap}.txt') for lap in (3, 6)
                      if os.path.exists(f'{R}/results/filter_dr_val_lap{lap}.txt')}
data['db'] = db_progress()
def shrink(src, dst, w, q=80):
    im = cv2.imread(src); h = int(im.shape[0] * w / im.shape[1])
    cv2.imwrite(dst, cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, q])
for f in glob.glob(f'{R}/analysis/projections/*.jpg'):
    shrink(f, f'{OUT}/img/proj_{os.path.basename(f)}', 900)
shrink(f'{R}/masks/preview.jpg', f'{OUT}/img/masks.jpg', 1300)
shrink(f'{R}/analysis/10_contact_sheet.jpg', f'{OUT}/img/contact.jpg', 1800, 78)
print(os.path.getsize(f'{OUT}/data.json') // 1024, 'KB json')
json.dump(data, open(f'{OUT}/data.json', 'w'), separators=(',', ':'))
print(os.path.getsize(f'{OUT}/data.json') // 1024, 'KB json (final)')
