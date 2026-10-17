#!/usr/bin/env python3
"""Real-time simulation of a camera localizer on a camloc mcap, scored against GT.

Mode A — replay an existing prediction CSV (timestamp_ns,x,y,yaw):
    python3 sim/simulate.py --gt gt.csv --pred results/baseline_dead_reckoning.csv --mcap $MCAP --speed 0 --no-gui
Mode B — run a localizer live:
    python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer sim/my_loc.py:Localizer --speed 1 --start-s 300 --max-s 120
    ... --state-topic ''        # no car state at all

Localizer interface (mode B):
    class Localizer:
        def __init__(self, first_pose, camera_info, tf_static, track_bounds)
        def step(self, stamp_ns, images, car_state) -> (x, y, yaw)
  first_pose   dict(stamp_ns, x, y, yaw): GT at the first camera set of the window (= "first pose of the run")
  camera_info  {cam: dict(k, d, width, height, distortion_model)}            from the mcap
  tf_static    {"<parent> -> <child>": dict(translation=xyz, rotation=xyzw)}  from the mcap
  track_bounds {namespace: (N,3) array}                                       from the mcap
  images       {cam: BGR uint8 full-resolution array or None}                 one camera set (header stamps within 5 ms)
  car_state    list of dict(stamp_ns, <car_state.ALLOWED fields>) with stamp <= stamp_ns not delivered before
               (includes messages of dropped sets), or None with --state-topic ''.
/tf, GT and non-whitelisted state fields never reach the localizer.

Timing (--speed S > 0): set n "arrives" at wall0 + (stamp_n - stamp_0) / S. The consumer takes the newest set that has
arrived; older ones are dropped (no pose output for them). JPEG decoding happens in the simulator (thread pool) and is
reported separately from step(). --speed 0: as fast as possible, nothing is dropped.

Outputs in --out (default sim/runs/<tag>): pred.csv, frames.csv (per-set timings / errors), gt_window.csv,
evaluate.txt (official tools/evaluate.py on the window), summary.txt / summary.json.
GUI (OpenCV window): track + trajectories, 40 m zoom, cameras fl and r, error plot. Keys: q quit, space pause.
"""
import argparse, importlib.util, json, os, queue, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from car_state import ALLOWED  # noqa: E402  whitelist of motion fields

NS = '/constructor0'
CAMS = ['fl', 'fr', 'cl', 'cr', 'rl', 'rr', 'r']
IMG_T = {f'{NS}/sensor/camera_{c}/compressed_image': c for c in CAMS}
INFO_T = {f'{NS}/sensor/camera_{c}/camera_info': c for c in CAMS}
TF_STATIC, BOUNDS = '/tf_static', f'{NS}/track_bounds'
SET_TOL_NS = 5_000_000
STALE_NS = 150_000_000
STATE_LEAD_NS = 500_000_000    # car state streams before the first camera set of the window
EVAL = os.path.join(ROOT, 'data/c0_2026-10-13_22-53-42_camloc/tools/evaluate.py')
MAX_GT_GAP_NS = 50_000_000


def stamp_of(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


# ---------------------------------------------------------------- mcap
def read_static(path, t_start):
    """camera_info, tf_static, track_bounds: first messages at/after t_start (they repeat every 1-10 s)."""
    info, tfs, bounds = {}, {}, {}
    for start in (t_start, None):
        with open(path, 'rb') as f:
            for _, ch, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                    topics=list(INFO_T) + [TF_STATIC, BOUNDS], start_time=start, log_time_order=True):
                if ch.topic in INFO_T and INFO_T[ch.topic] not in info:
                    info[INFO_T[ch.topic]] = dict(k=list(msg.k), d=list(msg.d), width=msg.width, height=msg.height,
                                                  distortion_model=msg.distortion_model)
                elif ch.topic == TF_STATIC and not tfs:
                    for t in msg.transforms:
                        tr, r = t.transform.translation, t.transform.rotation
                        tfs[f'{t.header.frame_id} -> {t.child_frame_id}'] = dict(
                            translation=dict(x=tr.x, y=tr.y, z=tr.z), rotation=dict(x=r.x, y=r.y, z=r.z, w=r.w))
                elif ch.topic == BOUNDS and not bounds:
                    for m in msg.markers:
                        bounds[m.ns] = np.array([[p.x, p.y, p.z] for p in m.points])
                if len(info) == len(CAMS) and tfs and bounds:
                    return info, tfs, bounds
    raise RuntimeError(f'static topics missing: info {sorted(info)}, tf_static {bool(tfs)}, bounds {bool(bounds)}')


def reader(path, t0, t1, state_topic, q, stop, decode_imgs=True):
    """Camera sets in stamp order -> q: (stamp, {cam: jpeg bytes}, [state dicts with stamp <= set stamp]); None at end."""
    topics = list(IMG_T) + ([state_topic] if state_topic else [])
    pending, state_buf = {}, []

    def emit(key):
        s = pending.pop(key)
        stamp = s['fl'][0] if 'fl' in s else min(v[0] for v in s.values())
        n = 0
        while n < len(state_buf) and state_buf[n]['stamp_ns'] <= stamp:
            n += 1
        msgs, state_buf[:] = state_buf[:n], state_buf[n:]
        q.put((stamp, {c: v[1] for c, v in s.items()}, msgs))

    def flush(force=False):
        if not pending:
            return
        newest = max(pending)
        for key in sorted(pending):
            if force or len(pending[key]) == len(CAMS) or newest - key > STALE_NS:
                emit(key)
            else:
                break

    try:
        with open(path, 'rb') as f:
            for _, ch, m, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                    topics=topics, start_time=t0 - STATE_LEAD_NS, end_time=t1, log_time_order=True):
                if stop.is_set():
                    break
                if ch.topic in IMG_T:
                    if m.log_time < t0:       # state is read from STATE_LEAD_NS before the window, images are not
                        continue
                    s = stamp_of(msg)
                    key = next((k for k in pending if abs(k - s) < SET_TOL_NS), s)
                    pending.setdefault(key, {})[IMG_T[ch.topic]] = (s, bytes(msg.data) if decode_imgs else b'')
                    flush()
                else:
                    d = {'stamp_ns': stamp_of(msg)}
                    for k in ALLOWED:
                        d[k] = float(getattr(msg, k))
                    state_buf.append(d)
                    if len(state_buf) > 1 and state_buf[-1]['stamp_ns'] < state_buf[-2]['stamp_ns']:
                        state_buf.sort(key=lambda m: m['stamp_ns'])
            flush(force=True)
    finally:
        q.put(None)


# ---------------------------------------------------------------- GT / metrics
class GT:
    def __init__(self, path):
        g = pd.read_csv(path).sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
        self.df = g
        self.t = g.timestamp_ns.values.astype(np.int64)
        self.x, self.y = g.x.values, g.y.values
        self.yaw = np.unwrap(g.yaw.values)
        self.t0 = int(self.t[0])

    def at(self, stamps):
        """x, y, yaw, valid (evaluate.py rule: GT gap around the stamp <= 50 ms)."""
        s = np.asarray(stamps, np.int64)
        i = np.searchsorted(self.t, s)
        ok = (i > 0) & (i < len(self.t))
        i = np.clip(i, 1, len(self.t) - 1)
        ok &= (self.t[i] - self.t[i - 1]) <= MAX_GT_GAP_NS
        a = (s - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        f = lambda v: v[i - 1] + a * (v[i] - v[i - 1])
        return f(self.x), f(self.y), f(self.yaw), ok

    def speed(self, stamps):
        s = np.asarray(stamps, np.int64)
        x0, y0, _, _ = self.at(s - 50_000_000)
        x1, y1, _, _ = self.at(s + 50_000_000)
        return np.hypot(x1 - x0, y1 - y0) / 0.1


def errors(gt, stamps, x, y, yaw):
    gx, gy, gyaw, ok = gt.at(stamps)
    dx, dy = x - gx, y - gy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    dyaw = np.degrees(np.angle(np.exp(1j * (yaw - gyaw))))
    return dict(pos=np.hypot(dx, dy), lat=lat, lon=lon, dyaw=dyaw, ok=ok)


def lap_annotation(t_rel):
    p = os.path.join(ROOT, 'analysis', 'laps.csv')       # evaluation-only annotation
    if not os.path.exists(p):
        return [''] * len(t_rel)
    laps = pd.read_csv(p)
    i = np.clip(np.searchsorted(laps.t_start.values, t_rel, side='right') - 1, 0, len(laps) - 1)
    return [f'K{laps.lap.values[j]}' for j in i]


# ---------------------------------------------------------------- GUI
class Gui:
    W, H = 1600, 900

    def __init__(self, bounds, gt):
        self.bounds = bounds
        pts = np.concatenate([v[:, :2] for v in bounds.values()])
        lo, hi = pts.min(0) - 20, pts.max(0) + 20
        self.sc = min(760 / (hi[0] - lo[0]), 820 / (hi[1] - lo[1]))
        self.lo, self.hi = lo, hi
        self.base = np.full((self.H, self.W, 3), 24, np.uint8)
        for v in bounds.values():
            cv2.polylines(self.base, [self.to_px(v[:, :2])], False, (110, 110, 110), 1, cv2.LINE_AA)
        self.gt = gt
        self.trail_gt, self.trail_pr, self.err = [], [], []
        self.paused = False
        cv2.namedWindow('simulate', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('simulate', self.W, self.H)

    def to_px(self, xy):
        xy = np.atleast_2d(xy)
        u = 20 + (xy[:, 0] - self.lo[0]) * self.sc
        v = 40 + (self.hi[1] - xy[:, 1]) * self.sc
        return np.c_[u, v].astype(np.int32)

    def add(self, stamp, pred, imgs, info):
        gx, gy, gyaw, _ = self.gt.at([stamp])
        self.trail_gt.append((gx[0], gy[0]))
        if pred is not None:
            self.trail_pr.append(pred[:2])
            e = errors(self.gt, np.array([stamp]), *[np.array([v]) for v in pred])
            self.err.append(((stamp - self.gt.t0) / 1e9, e['pos'][0], e['lat'][0]))
        self.cur = (stamp, (gx[0], gy[0], gyaw[0]), pred, imgs, info)

    def draw(self):
        stamp, (gx, gy, gyaw), pred, imgs, info = self.cur
        im = self.base.copy()
        if len(self.trail_gt) > 1:
            cv2.polylines(im, [self.to_px(np.array(self.trail_gt[-3000:]))], False, (80, 200, 80), 1, cv2.LINE_AA)
        if len(self.trail_pr) > 1:
            cv2.polylines(im, [self.to_px(np.array(self.trail_pr[-3000:]))], False, (80, 80, 240), 1, cv2.LINE_AA)
        cv2.circle(im, tuple(self.to_px([gx, gy])[0]), 6, (80, 220, 80), 2)
        if pred is not None:
            cv2.circle(im, tuple(self.to_px(pred[:2])[0]), 4, (60, 60, 255), -1)
        # zoom: 40 m around GT, 9 px/m
        Z, zc, zs = 380, np.array([1200, 200]), 9.0
        cv2.rectangle(im, tuple(zc - Z // 2), tuple(zc + Z // 2), (60, 60, 60), 1)

        def zp(xy):
            d = (np.atleast_2d(xy)[:, :2] - [gx, gy]) * zs
            return (zc + np.c_[d[:, 0], -d[:, 1]]).astype(np.int32)
        sub = np.zeros((Z, Z, 3), np.uint8)
        for v in self.bounds.values():
            p = zp(v) - (zc - Z // 2)
            cv2.polylines(sub, [p], False, (130, 130, 130), 1, cv2.LINE_AA)

        def car(img, x, y, yaw, col):
            c = zp([x, y])[0] - (zc - Z // 2)
            tip = c + (np.array([np.cos(yaw), -np.sin(yaw)]) * 22).astype(int)
            cv2.circle(img, tuple(c), 5, col, 2)
            cv2.line(img, tuple(c), tuple(tip), col, 2)
        car(sub, gx, gy, gyaw, (80, 220, 80))
        if pred is not None:
            car(sub, *pred, (60, 60, 255))
        im[zc[1] - Z // 2:zc[1] + Z // 2, zc[0] - Z // 2:zc[0] + Z // 2] = sub
        # cameras
        y0 = 400
        for k, c in enumerate(('fl', 'r')):
            if imgs and imgs.get(c) is not None:
                ci = cv2.resize(imgs[c], (400, 193), interpolation=cv2.INTER_AREA)
                im[y0 + k * 198:y0 + k * 198 + 193, 800:1200] = ci
                cv2.putText(im, c, (806, y0 + k * 198 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # error plot (last 60 s)
        px0, py0, pw, ph = 1210, 400, 380, 390
        cv2.rectangle(im, (px0, py0), (px0 + pw, py0 + ph), (60, 60, 60), 1)
        if self.err:
            e = np.array(self.err)
            e = e[e[:, 0] > e[-1, 0] - 60]
            ymax = max(1.0, float(np.percentile(np.abs(e[:, 1:]), 99)) * 1.2)
            u = px0 + (e[:, 0] - (e[-1, 0] - 60)) / 60 * pw
            for col, (j, c) in enumerate(((1, (60, 60, 255)), (2, (255, 180, 60)))):
                v = py0 + ph / 2 - e[:, j] / ymax * ph / 2
                cv2.polylines(im, [np.c_[u, v].astype(np.int32)], False, c, 1, cv2.LINE_AA)
            cv2.line(im, (px0, py0 + ph // 2), (px0 + pw, py0 + ph // 2), (70, 70, 70), 1)
            cv2.putText(im, f'+-{ymax:.1f} m  red: |pos|  orange: lateral', (px0 + 4, py0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for k, line in enumerate(info):
            cv2.putText(im, line, (800, 812 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(im, 'green: GT   red: prediction', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        cv2.imshow('simulate', im)

    def wait(self, ms):
        """pump events for ms; returns False on quit."""
        t_end = time.perf_counter() + ms / 1000
        while True:
            k = cv2.waitKey(max(1, int(min(20, (t_end - time.perf_counter()) * 1000)))) & 0xFF
            if k == ord('q'):
                return False
            if k == ord(' '):
                self.paused = not self.paused
            if time.perf_counter() >= t_end and not self.paused:
                return True


# ---------------------------------------------------------------- replay "localizer" for mode A
class Replay:
    def __init__(self, path):
        p = pd.read_csv(path).sort_values('timestamp_ns')
        self.t = p.timestamp_ns.values.astype(np.int64)
        self.v = p[['x', 'y', 'yaw']].values

    def step(self, stamp_ns, images, car_state):
        i = np.searchsorted(self.t, stamp_ns)
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - stamp_ns) < SET_TOL_NS:
                return tuple(self.v[j])
        return None


def load_class(spec):
    path, cls = spec.rsplit(':', 1)
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    m = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    return getattr(mod, cls)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--mcap', required=True)
    ap.add_argument('--pred', help='mode A: prediction CSV to replay')
    ap.add_argument('--localizer', help='mode B: path/to/file.py:ClassName')
    ap.add_argument('--speed', type=float, default=1.0, help='1 = real time, 0 = as fast as possible (no drops)')
    ap.add_argument('--start-s', type=float, default=None, help='window start, s from the first message of the bag')
    ap.add_argument('--max-s', type=float, default=None, help='window length, s')
    ap.add_argument('--state-topic', default=f'{NS}/state_estimation', help="'' = no car state")
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    assert bool(a.pred) != bool(a.localizer), 'give exactly one of --pred (mode A) or --localizer (mode B)'

    gt = GT(a.gt)
    with open(a.mcap, 'rb') as f:
        st = make_reader(f).get_summary().statistics
    bag0, bag1 = st.message_start_time, st.message_end_time
    replay = Replay(a.pred) if a.pred else None
    t0 = bag0 + int((a.start_s or 0) * 1e9)
    if replay is not None and a.start_s is None:
        t0 = max(bag0, int(replay.t[0]) - SET_TOL_NS)
    t1 = bag1 + 1
    if a.max_s is not None:
        t1 = t0 + int(a.max_s * 1e9)
    elif replay is not None:
        t1 = min(bag1 + 1, int(replay.t[-1]) + SET_TOL_NS)
    state_topic = a.state_topic if a.localizer else ''
    tag = a.tag or ('A_' + os.path.splitext(os.path.basename(a.pred))[0] if a.pred else
                    'B_' + ('state' if state_topic else 'nostate') +
                    f'_s{a.start_s or 0:g}_{a.max_s if a.max_s is not None else "all"}_x{a.speed:g}')
    out = a.out or os.path.join(HERE, 'runs', tag)
    os.makedirs(out, exist_ok=True)
    gui_on = not a.no_gui
    print(f'[sim] {tag}: window {(t0 - bag0) / 1e9:.1f}..{(min(t1, bag1) - bag0) / 1e9:.1f} s, speed {a.speed:g}, '
          f'state {state_topic or "OFF"}, out {out}', flush=True)

    q, stop = queue.Queue(maxsize=200), threading.Event()
    th = threading.Thread(target=reader, args=(a.mcap, t0, t1, state_topic, q, stop, bool(a.localizer) or gui_on),
                          daemon=True)
    th.start()
    cur = q.get()
    if cur is None:
        raise SystemExit('no camera sets in the window')
    info, tfs, bounds = read_static(a.mcap, t0)
    t_init = time.perf_counter()
    if a.localizer:
        fx, fy, fyaw, _ = gt.at([cur[0]])
        first_pose = dict(stamp_ns=int(cur[0]), x=float(fx[0]), y=float(fy[0]),
                          yaw=float(np.angle(np.exp(1j * fyaw[0]))))
        loc = load_class(a.localizer)(first_pose, info, tfs, bounds)
    else:
        loc = replay
    t_init = time.perf_counter() - t_init
    print(f'[sim] init {t_init:.1f} s', flush=True)

    gui = Gui(bounds, gt) if gui_on else None
    pool = ThreadPoolExecutor(len(CAMS))
    decode = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
    rows, dropped, carry = [], [], []
    wall0, s0, last_draw, last_print = None, None, 0.0, time.perf_counter()
    quit_ = False
    while cur is not None and not quit_:
        nxt = q.get()
        stamp = cur[0]
        if wall0 is None:
            wall0, s0 = time.perf_counter(), stamp
        arrival = wall0 + (stamp - s0) / 1e9 / a.speed if a.speed > 0 else None
        if a.speed > 0:
            now = time.perf_counter()
            if nxt is not None and now >= wall0 + (nxt[0] - s0) / 1e9 / a.speed:
                carry += cur[2]
                dropped.append(stamp)
                cur = nxt
                continue
            if now < arrival:
                if gui:
                    quit_ = not gui.wait((arrival - now) * 1000)
                else:
                    time.sleep(arrival - now)
        ta = time.perf_counter()
        need_imgs = a.localizer or gui
        imgs = dict(zip(cur[1], pool.map(decode, cur[1].values()))) if need_imgs else {}
        imgs = {c: imgs.get(c) for c in CAMS}
        tb = time.perf_counter()
        cs = (carry + cur[2]) if state_topic else None
        carry = []
        pred = loc.step(int(stamp), imgs, cs)
        tc = time.perf_counter()
        rec = dict(timestamp_ns=int(stamp), decode_ms=(tb - ta) * 1e3, step_ms=(tc - tb) * 1e3,
                   lag_ms=(tc - arrival) * 1e3 if arrival else np.nan, n_cams=sum(v is not None for v in imgs.values()))
        if pred is not None:
            rec.update(x=float(pred[0]), y=float(pred[1]), yaw=float(np.angle(np.exp(1j * pred[2]))))
        elif replay is not None:
            dropped.append(stamp)          # mode A: camera set without a prediction
        rows.append(rec)
        if gui and (tc - last_draw > 0.1 or a.speed == 0):
            e_now = errors(gt, np.array([stamp]), *[np.array([v]) for v in pred]) if pred is not None else None
            info_txt = [f't = {(stamp - gt.t0) / 1e9:7.1f} s   v = {gt.speed([stamp])[0]:5.1f} m/s',
                        f'step {rec["step_ms"]:6.1f} ms  decode {rec["decode_ms"]:5.1f} ms  dropped {len(dropped)}',
                        (f'err {e_now["pos"][0]:.2f} m  lat {e_now["lat"][0]:+.2f}  lon {e_now["lon"][0]:+.2f}  '
                         f'yaw {e_now["dyaw"][0]:+.2f} deg') if e_now else 'no prediction']
            gui.add(stamp, pred, imgs, info_txt)
            gui.draw()
            quit_ = quit_ or not gui.wait(1)
            last_draw = time.perf_counter()
        elif gui:
            gui.add(stamp, pred, imgs, gui.cur[4] if hasattr(gui, 'cur') else [])
        if time.perf_counter() - last_print > 30:
            last_print = time.perf_counter()
            print(f'[sim] t={(stamp - gt.t0) / 1e9:7.1f} s  sets {len(rows)}  dropped {len(dropped)}  '
                  f'step mean {np.mean([r["step_ms"] for r in rows]):.1f} ms', flush=True)
        cur = nxt
    stop.set()
    while th.is_alive():          # unblock the reader
        try:
            q.get_nowait()
        except queue.Empty:
            time.sleep(0.05)
    if gui:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------ results
    fr = pd.DataFrame(rows)
    fr = fr[fr.timestamp_ns.notna()]
    has = fr.x.notna() if 'x' in fr else pd.Series(False, index=fr.index)
    pr = fr[has]
    pr[['timestamp_ns', 'x', 'y', 'yaw']].round({'x': 4, 'y': 4, 'yaw': 6}).to_csv(os.path.join(out, 'pred.csv'),
                                                                                    index=False)
    e = errors(gt, pr.timestamp_ns.values, pr.x.values, pr.y.values, pr.yaw.values)
    for k in ('pos', 'lat', 'lon', 'dyaw'):
        fr.loc[has, k] = e[k]
    fr.to_csv(os.path.join(out, 'frames.csv'), index=False)
    ok = e['ok']
    rms = lambda v: float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else float('nan')
    total = (fr.decode_ms + fr.step_ms).values
    S = dict(tag=tag, mode='A' if a.pred else 'B', localizer=a.localizer or a.pred, speed=a.speed,
             state=state_topic or None, window_s=[(t0 - bag0) / 1e9, (min(t1, bag1) - bag0) / 1e9],
             sets_in_window=len(fr) + (len(dropped) if a.localizer else 0), poses=int(has.sum()),
             dropped=len(dropped), evaluated=int(ok.sum()),
             pos_rmse=rms(e['pos']), lat_rmse=rms(e['lat']), lon_rmse=rms(e['lon']), yaw_rmse_deg=rms(e['dyaw']),
             pos_median=float(np.median(e['pos'][ok])) if ok.any() else None,
             pos_p95=pct(e['pos'][ok], 95), within_1m=float(np.mean(e['pos'][ok] < 1.0) * 100) if ok.any() else None,
             init_s=t_init)
    if a.localizer:
        S.update(step_ms_mean=float(fr.step_ms.mean()), step_ms_p95=pct(fr.step_ms, 95), step_ms_max=float(fr.step_ms.max()),
                 decode_ms_mean=float(fr.decode_ms.mean()), total_ms_mean=float(total.mean()),
                 total_ms_p95=pct(total, 95), total_ms_max=float(total.max()),
                 over_100ms=int((total > 100).sum()))
        if hasattr(loc, 'summary'):
            S['localizer_stats'] = loc.summary()
    # worst moments: largest position errors, >= 5 s apart
    worst = []
    if ok.any():
        idx = np.where(ok)[0][np.argsort(-e['pos'][ok])]
        ts = pr.timestamp_ns.values
        for i in idx:
            if all(abs(ts[i] - ts[j]) > 5e9 for j in worst):
                worst.append(i)
            if len(worst) == 5:
                break
        t_rel = (ts[worst] - gt.t0) / 1e9
        laps = lap_annotation(t_rel)
        v = gt.speed(ts[worst])
        S['worst'] = [dict(t_s=round(float(t_rel[k]), 1), lap=laps[k], stamp=int(ts[i]), v_mps=round(float(v[k]), 1),
                           pos=round(float(e['pos'][i]), 3), lat=round(float(e['lat'][i]), 3),
                           lon=round(float(e['lon'][i]), 3), dyaw_deg=round(float(e['dyaw'][i]), 3))
                      for k, i in enumerate(worst)]
    # official evaluate.py on the GT window of this run
    gw = gt.df[(gt.df.timestamp_ns >= fr.timestamp_ns.min() - 100_000_000) &
               (gt.df.timestamp_ns <= fr.timestamp_ns.max() + 100_000_000)]
    gw.to_csv(os.path.join(out, 'gt_window.csv'), index=False)
    ev = subprocess.run([sys.executable, EVAL, os.path.join(out, 'gt_window.csv'), os.path.join(out, 'pred.csv')],
                        capture_output=True, text=True)
    open(os.path.join(out, 'evaluate.txt'), 'w').write(ev.stdout + ev.stderr)
    S['evaluate'] = ev.stdout
    json.dump(S, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    lines = [f'== {tag}',
             f'poses {S["poses"]} / sets {S["sets_in_window"]}, dropped {S["dropped"]}, evaluated {S["evaluated"]}',
             f'pos RMSE {S["pos_rmse"]:.3f} m  (median {S["pos_median"]:.3f}, p95 {S["pos_p95"]:.3f})  '
             f'lat RMSE {S["lat_rmse"]:.3f}  lon RMSE {S["lon_rmse"]:.3f}  yaw RMSE {S["yaw_rmse_deg"]:.3f} deg  '
             f'<1 m {S["within_1m"]:.1f} %']
    if a.localizer:
        lines.append(f'step ms mean {S["step_ms_mean"]:.1f} p95 {S["step_ms_p95"]:.1f} max {S["step_ms_max"]:.1f} | '
                     f'decode {S["decode_ms_mean"]:.1f} | step+decode mean {S["total_ms_mean"]:.1f} '
                     f'p95 {S["total_ms_p95"]:.1f} max {S["total_ms_max"]:.1f}, >100 ms: {S["over_100ms"]} | '
                     f'init {t_init:.1f} s')
        if 'localizer_stats' in S:
            lines.append(f'localizer: {S["localizer_stats"]}')
    lines.append('worst moments:')
    for w in S.get('worst', []):
        lines.append(f'  t={w["t_s"]:7.1f} s {w["lap"]:3s} v={w["v_mps"]:5.1f} m/s  pos {w["pos"]:.2f}  '
                     f'lat {w["lat"]:+.2f}  lon {w["lon"]:+.2f}  yaw {w["dyaw_deg"]:+.2f} deg')
    lines += ['evaluate.py (GT window of this run):', ev.stdout.rstrip()]
    txt = '\n'.join(lines)
    open(os.path.join(out, 'summary.txt'), 'w').write(txt + '\n')
    print(txt, flush=True)
    if hasattr(loc, 'close'):
        loc.close()


if __name__ == '__main__':
    main()
