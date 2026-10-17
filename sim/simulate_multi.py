#!/usr/bin/env python3
"""Several localizers / prediction CSVs on the same camera stream, in one window, scored side by side.

    python3 sim/simulate_multi.py --gt gt.csv --mcap $MCAP --start-s 523.4 --max-s 110.6 \
        --run v1=sim/my_loc.py:Localizer --run v2=v2/my_loc_v2.py:Localizer
    python3 sim/simulate_multi.py --gt gt.csv --mcap $MCAP --speed 0 --no-gui \
        --run v1=v2/runs/v1_K1/pred.csv --run v2=v2/runs/v2_K1/pred.csv          # replay saved results
    ... --run v2_nostate=v2/my_loc_v2.py:Localizer@nostate                         # that run gets no car state

--run [LABEL=]SPEC, repeatable. SPEC is path/file.py:Class (live, same interface as sim/simulate.py) or a CSV
(timestamp_ns,x,y,yaw; replayed). Suffix @nostate: the live run gets car_state=None.
Every camera set goes to every run (same images, same whitelisted car state). Window: --start-s/--max-s, or the
time range covered by the CSVs when there are only CSV runs.

Timing note: live runs share one process, CPU and GPU, so their timings (and, through the delayed corrections, their
accuracy) are worse than when each runs alone with sim/simulate.py. For final numbers run them separately; this
tool is for watching and comparing behaviour on the same frames. In real time (--speed > 0) a set is dropped for
all runs when the newest set has already arrived.

Outputs: sim/runs/multi_<tag>/<label>/{pred.csv,frames.csv,evaluate.txt,summary.json} and a comparison table
(compare.md). GUI keys: q quit, space pause, 1..9 show/hide run.
"""
import argparse, json, os, queue, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from simulate import (CAMS, EVAL, GT, NS, SET_TOL_NS, Replay, errors, lap_annotation, load_class,  # noqa: E402
                      read_static, reader)

GT_COL = (80, 220, 80)
COLORS = [(60, 60, 255), (255, 170, 40), (40, 200, 255), (220, 80, 220), (200, 200, 60), (120, 120, 255),
          (255, 255, 255), (60, 160, 160), (160, 100, 40)]


class Run:
    def __init__(self, spec, k):
        label, _, spec = spec.rpartition('=') if '=' in spec else ('', '', spec)
        self.nostate = spec.endswith('@nostate')
        spec = spec[:-len('@nostate')] if self.nostate else spec
        self.spec = spec
        self.live = not spec.endswith('.csv')
        self.label = label or (os.path.splitext(os.path.basename(spec.split(':')[0]))[0] + ('_nostate' if self.nostate else ''))
        self.color = COLORS[k % len(COLORS)]
        self.loc = None if self.live else Replay(spec)
        self.rows, self.visible = [], True
        self.trail, self.err = [], []


# ---------------------------------------------------------------- GUI
class MultiGui:
    W, H = 1600, 900

    def __init__(self, bounds, gt, runs):
        self.bounds, self.gt, self.runs = bounds, gt, runs
        pts = np.concatenate([v[:, :2] for v in bounds.values()])
        self.lo, self.hi = pts.min(0) - 20, pts.max(0) + 20
        self.sc = min(760 / (self.hi[0] - self.lo[0]), 820 / (self.hi[1] - self.lo[1]))
        self.base = np.full((self.H, self.W, 3), 24, np.uint8)
        for v in bounds.values():
            cv2.polylines(self.base, [self.px(v[:, :2])], False, (110, 110, 110), 1, cv2.LINE_AA)
        self.trail_gt, self.paused, self.cur = [], False, None
        cv2.namedWindow('simulate_multi', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('simulate_multi', self.W, self.H)

    def px(self, xy):
        xy = np.atleast_2d(xy)
        return np.c_[20 + (xy[:, 0] - self.lo[0]) * self.sc, 40 + (self.hi[1] - xy[:, 1]) * self.sc].astype(np.int32)

    def add(self, stamp, imgs, info):
        gx, gy, gyaw, _ = self.gt.at([stamp])
        self.trail_gt.append((gx[0], gy[0]))
        self.cur = (stamp, (gx[0], gy[0], gyaw[0]), imgs, info)

    def draw(self):
        stamp, (gx, gy, gyaw), imgs, info = self.cur
        im = self.base.copy()
        if len(self.trail_gt) > 1:
            cv2.polylines(im, [self.px(np.array(self.trail_gt[-3000:]))], False, GT_COL, 1, cv2.LINE_AA)
        for r in self.runs:
            if r.visible and len(r.trail) > 1:
                cv2.polylines(im, [self.px(np.array(r.trail[-3000:]))], False, r.color, 1, cv2.LINE_AA)
        cv2.circle(im, tuple(self.px([gx, gy])[0]), 6, GT_COL, 2)
        # zoom 40 m around GT
        Z, zc, zs = 380, np.array([1200, 200]), 9.0
        sub = np.zeros((Z, Z, 3), np.uint8)

        def zp(xy):
            d = (np.atleast_2d(xy)[:, :2] - [gx, gy]) * zs
            return (np.array([Z // 2, Z // 2]) + np.c_[d[:, 0], -d[:, 1]]).astype(np.int32)
        for v in self.bounds.values():
            cv2.polylines(sub, [zp(v)], False, (130, 130, 130), 1, cv2.LINE_AA)

        def car(x, y, yaw, col, rad):
            c = zp([x, y])[0]
            cv2.circle(sub, tuple(c), rad, col, 2)
            cv2.line(sub, tuple(c), tuple(c + (np.array([np.cos(yaw), -np.sin(yaw)]) * 22).astype(int)), col, 2)
        car(gx, gy, gyaw, GT_COL, 6)
        for r in self.runs:
            if r.visible and r.rows and r.rows[-1]['timestamp_ns'] == stamp and 'x' in r.rows[-1]:
                car(r.rows[-1]['x'], r.rows[-1]['y'], r.rows[-1]['yaw'], r.color, 4)
        im[zc[1] - Z // 2:zc[1] + Z // 2, zc[0] - Z // 2:zc[0] + Z // 2] = sub
        cv2.rectangle(im, tuple(zc - Z // 2), tuple(zc + Z // 2), (60, 60, 60), 1)
        cv2.putText(im, '40 m zoom', (zc[0] - Z // 2 + 4, zc[1] - Z // 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)
        for k, c in enumerate(('fl', 'r')):
            if imgs and imgs.get(c) is not None:
                im[400 + k * 198:400 + k * 198 + 193, 800:1200] = cv2.resize(imgs[c], (400, 193), interpolation=cv2.INTER_AREA)
                cv2.putText(im, c, (806, 420 + k * 198), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # |pos| error, last 60 s, one line per run
        px0, py0, pw, ph = 1210, 400, 380, 390
        cv2.rectangle(im, (px0, py0), (px0 + pw, py0 + ph), (60, 60, 60), 1)
        t_now = (stamp - self.gt.t0) / 1e9
        allv = [e for r in self.runs if r.visible for (t, e) in r.err[-700:] if t > t_now - 60]
        ymax = max(0.5, float(np.percentile(allv, 99)) * 1.2) if allv else 1.0
        for r in self.runs:
            if not r.visible or len(r.err) < 2:
                continue
            e = np.array([p for p in r.err[-700:] if p[0] > t_now - 60])
            if len(e) > 1:
                u = px0 + (e[:, 0] - (t_now - 60)) / 60 * pw
                v = py0 + ph - np.minimum(e[:, 1], ymax) / ymax * ph
                cv2.polylines(im, [np.c_[u, v].astype(np.int32)], False, r.color, 1, cv2.LINE_AA)
        cv2.putText(im, f'|pos error|, 0..{ymax:.2f} m, last 60 s', (px0 + 4, py0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1)
        # legend / live metrics
        cv2.putText(im, info[0], (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(im, 'GT', (800, 812), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COL, 1, cv2.LINE_AA)
        for k, (r, line) in enumerate(zip(self.runs, info[1:])):
            col = r.color if r.visible else (90, 90, 90)
            cv2.putText(im, f'{k + 1} {line}', (800, 832 + 17 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.imshow('simulate_multi', im)

    def wait(self, ms):
        t_end = time.perf_counter() + ms / 1000
        while True:
            k = cv2.waitKey(max(1, int(min(20, (t_end - time.perf_counter()) * 1000)))) & 0xFF
            if k == ord('q'):
                return False
            if k == ord(' '):
                self.paused = not self.paused
            if ord('1') <= k <= ord('9') and k - ord('1') < len(self.runs):
                r = self.runs[k - ord('1')]
                r.visible = not r.visible
                self.draw()
            if time.perf_counter() >= t_end and not self.paused:
                return True


# ---------------------------------------------------------------- results
def finish(run, gt, out, n_dropped):
    d = os.path.join(out, run.label)
    os.makedirs(d, exist_ok=True)
    fr = pd.DataFrame(run.rows)
    has = fr.x.notna() if 'x' in fr else pd.Series(False, index=fr.index)
    pr = fr[has]
    pr[['timestamp_ns', 'x', 'y', 'yaw']].round({'x': 4, 'y': 4, 'yaw': 6}).to_csv(os.path.join(d, 'pred.csv'), index=False)
    e = errors(gt, pr.timestamp_ns.values, pr.x.values, pr.y.values, pr.yaw.values)
    for k in ('pos', 'lat', 'lon', 'dyaw'):
        fr.loc[has, k] = e[k]
    fr.to_csv(os.path.join(d, 'frames.csv'), index=False)
    ok = e['ok']
    rms = lambda v: float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else float('nan')
    gw = gt.df[(gt.df.timestamp_ns >= fr.timestamp_ns.min() - 10**8) & (gt.df.timestamp_ns <= fr.timestamp_ns.max() + 10**8)]
    gw.to_csv(os.path.join(d, 'gt_window.csv'), index=False)
    ev = subprocess.run([sys.executable, EVAL, os.path.join(d, 'gt_window.csv'), os.path.join(d, 'pred.csv')],
                        capture_output=True, text=True).stdout
    open(os.path.join(d, 'evaluate.txt'), 'w').write(ev)
    S = dict(label=run.label, spec=run.spec, live=run.live, car_state=not run.nostate, poses=int(has.sum()),
             sets=len(fr), dropped=n_dropped + int((~has).sum()), pos_rmse=rms(e['pos']), lat_rmse=rms(e['lat']),
             lon_rmse=rms(e['lon']), yaw_rmse_deg=rms(e['dyaw']),
             within_025=float(np.mean(e['pos'][ok] < .25) * 100) if ok.any() else None,
             within_1m=float(np.mean(e['pos'][ok] < 1) * 100) if ok.any() else None, evaluate=ev)
    if run.live:
        S.update(step_ms_mean=float(fr.step_ms.mean()), step_ms_p95=float(np.percentile(fr.step_ms, 95)),
                 step_ms_max=float(fr.step_ms.max()))
        if hasattr(run.loc, 'summary'):
            S['localizer_stats'] = run.loc.summary()
    if ok.any():
        idx = np.where(ok)[0][np.argsort(-e['pos'][ok])]
        ts, worst = pr.timestamp_ns.values, []
        for i in idx:
            if all(abs(ts[i] - ts[j]) > 5e9 for j in worst):
                worst.append(i)
            if len(worst) == 5:
                break
        t_rel = (ts[worst] - gt.t0) / 1e9
        laps = lap_annotation(t_rel)
        S['worst'] = [dict(t_s=round(float(t_rel[k]), 1), lap=laps[k], pos=round(float(e['pos'][i]), 3),
                           lat=round(float(e['lat'][i]), 3), lon=round(float(e['lon'][i]), 3),
                           dyaw_deg=round(float(e['dyaw'][i]), 3)) for k, i in enumerate(worst)]
    json.dump(S, open(os.path.join(d, 'summary.json'), 'w'), indent=1, default=float)
    return S


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--mcap', required=True)
    ap.add_argument('--run', action='append', required=True, help='[LABEL=]file.py:Class[@nostate] or [LABEL=]pred.csv')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--start-s', type=float, default=None)
    ap.add_argument('--max-s', type=float, default=None)
    ap.add_argument('--state-topic', default=f'{NS}/state_estimation')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    runs = [Run(s, k) for k, s in enumerate(a.run)]
    assert len({r.label for r in runs}) == len(runs), 'labels must be unique (use LABEL=...)'
    live = [r for r in runs if r.live]
    gt = GT(a.gt)
    with open(a.mcap, 'rb') as f:
        st = make_reader(f).get_summary().statistics
    bag0, bag1 = st.message_start_time, st.message_end_time
    if a.start_s is None and not live:
        t0 = max(bag0, min(int(r.loc.t[0]) for r in runs) - SET_TOL_NS)
    else:
        t0 = bag0 + int((a.start_s or 0) * 1e9)
    if a.max_s is not None:
        t1 = t0 + int(a.max_s * 1e9)
    elif not live:
        t1 = min(bag1 + 1, max(int(r.loc.t[-1]) for r in runs) + SET_TOL_NS)
    else:
        t1 = bag1 + 1
    need_state = any(r.live and not r.nostate for r in runs)
    state_topic = a.state_topic if need_state else ''
    tag = a.tag or '_'.join(r.label for r in runs) + f'_s{a.start_s or 0:g}'
    out = os.path.join(HERE, 'runs', 'multi_' + tag)
    os.makedirs(out, exist_ok=True)
    gui_on = not a.no_gui
    print(f'[multi] {len(runs)} runs ({len(live)} live): {[r.label for r in runs]}; window '
          f'{(t0 - bag0) / 1e9:.1f}..{(min(t1, bag1) - bag0) / 1e9:.1f} s, speed {a.speed:g}, out {out}', flush=True)
    if len(live) > 1:
        print('[multi] note: live runs share CPU/GPU -> timings and accuracy differ from solo runs', flush=True)

    q, stop = queue.Queue(maxsize=200), threading.Event()
    th = threading.Thread(target=reader, args=(a.mcap, t0, t1, state_topic, q, stop, bool(live) or gui_on), daemon=True)
    th.start()
    cur = q.get()
    if cur is None:
        raise SystemExit('no camera sets in the window')
    info, tfs, bounds = read_static(a.mcap, t0)
    fx, fy, fyaw, _ = gt.at([cur[0]])
    first_pose = dict(stamp_ns=int(cur[0]), x=float(fx[0]), y=float(fy[0]), yaw=float(np.angle(np.exp(1j * fyaw[0]))))
    for r in live:
        t = time.perf_counter()
        r.loc = load_class(r.spec)(first_pose, info, tfs, bounds)
        print(f'[multi] {r.label}: init {time.perf_counter() - t:.1f} s', flush=True)

    gui = MultiGui(bounds, gt, runs) if gui_on else None
    pool = ThreadPoolExecutor(len(CAMS))
    decode = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
    dropped, carry, wall0, s0, last_draw, quit_ = 0, [], None, None, 0.0, False
    while cur is not None and not quit_:
        nxt = q.get()
        stamp = cur[0]
        if wall0 is None:
            wall0, s0 = time.perf_counter(), stamp
        if a.speed > 0:
            arrival = wall0 + (stamp - s0) / 1e9 / a.speed
            now = time.perf_counter()
            if nxt is not None and now >= wall0 + (nxt[0] - s0) / 1e9 / a.speed:
                carry += cur[2]; dropped += 1; cur = nxt
                continue
            if now < arrival:
                quit_ = not gui.wait((arrival - now) * 1000) if gui else (time.sleep(arrival - now) or False)
        imgs = dict(zip(cur[1], pool.map(decode, cur[1].values()))) if (live or gui) else {}
        imgs = {c: imgs.get(c) for c in CAMS}
        cs = carry + cur[2]
        carry = []
        for r in runs:
            tb = time.perf_counter()
            pred = r.loc.step(int(stamp), imgs, None if (r.nostate or not state_topic) else cs)
            rec = dict(timestamp_ns=int(stamp), step_ms=(time.perf_counter() - tb) * 1e3)
            if pred is not None:
                rec.update(x=float(pred[0]), y=float(pred[1]), yaw=float(np.angle(np.exp(1j * pred[2]))))
                r.trail.append(pred[:2])
                e = errors(gt, np.array([stamp]), np.array([pred[0]]), np.array([pred[1]]), np.array([pred[2]]))
                r.err.append(((stamp - gt.t0) / 1e9, float(e['pos'][0])))
                rec['_lat'] = float(e['lat'][0])
            r.rows.append(rec)
        if gui:
            if time.perf_counter() - last_draw > 0.1 or a.speed == 0:
                lines = [f't = {(stamp - gt.t0) / 1e9:7.1f} s   v = {gt.speed([stamp])[0]:5.1f} m/s   dropped {dropped}']
                for r in runs:
                    e = np.array([p[1] for p in r.err]) if r.err else np.array([np.nan])
                    lat = np.array([x['_lat'] for x in r.rows if '_lat' in x]) if r.err else np.array([np.nan])
                    stp = np.mean([x['step_ms'] for x in r.rows[-50:]])
                    lines.append(f'{r.label:14s} now {e[-1]:5.2f} m | RMSE {np.sqrt(np.mean(e ** 2)):.3f} '
                                 f'lat {np.sqrt(np.mean(lat ** 2)):.3f} | step {stp:5.1f} ms')
                gui.add(stamp, imgs, lines)
                gui.draw()
                quit_ = quit_ or not gui.wait(1)
                last_draw = time.perf_counter()
            else:
                gx, gy, _, _ = gt.at([stamp]); gui.trail_gt.append((gx[0], gy[0]))
        cur = nxt
    stop.set()
    while th.is_alive():
        try:
            q.get_nowait()
        except queue.Empty:
            time.sleep(0.05)
    if gui:
        cv2.destroyAllWindows()

    for r in runs:
        for x in r.rows:
            x.pop('_lat', None)
    res = [finish(r, gt, out, dropped) for r in runs]
    for r in live:
        if hasattr(r.loc, 'close'):
            r.loc.close()
    lines = ['| run | car state | pos RMSE | lat | lon | yaw° | <0.25 м | <1 м | poses | dropped | step ms (mean/p95) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for s in res:
        stp = f"{s['step_ms_mean']:.1f} / {s['step_ms_p95']:.1f}" if s['live'] else 'CSV'
        lines.append(f"| {s['label']} | {('yes' if s['car_state'] else 'no') if s['live'] else '—'} | **{s['pos_rmse']:.3f}** | {s['lat_rmse']:.3f} | "
                     f"{s['lon_rmse']:.3f} | {s['yaw_rmse_deg']:.3f} | {s['within_025']:.1f} | {s['within_1m']:.1f} | "
                     f"{s['poses']} | {s['dropped']} | {stp} |")
    lines.append('')
    for s in res:
        lines.append(f"worst {s['label']}: " + '; '.join(f"t={w['t_s']} {w['lap']} {w['pos']:.2f} m (lat {w['lat']:+.2f})"
                                                        for w in s.get('worst', [])))
    txt = '\n'.join(lines)
    open(os.path.join(out, 'compare.md'), 'w').write(txt + '\n')
    print(txt, flush=True)


if __name__ == '__main__':
    main()
