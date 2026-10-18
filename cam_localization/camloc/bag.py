"""Reading a run (ROS 2 mcap): static calibration, camera sets and car state, streamed in time order.

Car state, in order of preference (only motion fields; pose fields are never read):
  /constructor0/state_estimation   the fields of car_state.ALLOWED (training-run format)
  /constructor0/validation/twist   the same quantities from the body-frame twist (frame constructor0/base_link; format
                                   of the organisers' hidden-test example)
Each car-state message becomes dict(stamp_ns, vx_mps, vy_mps, wz_radps).
Track borders: /constructor0/track_bounds of the bag, else model/track_bounds.npz (same circuit).
"""
import os
import threading, queue
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from .car_state import ALLOWED
from .paths import MODEL

NS = '/constructor0'
CAMS = ['fl', 'fr', 'cl', 'cr', 'rl', 'rr', 'r']
IMG_T = {f'{NS}/sensor/camera_{c}/compressed_image': c for c in CAMS}
INFO_T = {f'{NS}/sensor/camera_{c}/camera_info': c for c in CAMS}
TF_STATIC, BOUNDS = '/tf_static', f'{NS}/track_bounds'
STATE, TWIST = f'{NS}/state_estimation', f'{NS}/validation/twist'
SET_TOL_NS = 5_000_000          # images of one camera set: header stamps within 5 ms (hardware-synced cameras)
STALE_NS = 150_000_000          # a set still missing cameras is emitted once a newer set is this much later
STATE_LEAD_NS = 500_000_000     # car state is read from this long before the first camera set


def stamp_of(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def _reader(path, **kw):
    f = open(path, 'rb')
    return f, make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(**kw)


def topics(path):
    with open(path, 'rb') as f:
        s = make_reader(f).get_summary()
    return {ch.topic for ch in s.channels.values()}, (s.statistics.message_start_time, s.statistics.message_end_time)


def read_static(path):
    """camera_info {cam: dict}, tf_static {'parent -> child': dict}, track_bounds {ns: (N, 3)} (bag or model)."""
    info, tfs, bounds = {}, {}, {}
    have = topics(path)[0]
    want = list(INFO_T) + [TF_STATIC] + ([BOUNDS] if BOUNDS in have else [])
    f, it = _reader(path, topics=want, log_time_order=True)
    with f:
        for _, ch, _, msg in it:
            if ch.topic in INFO_T and INFO_T[ch.topic] not in info:
                info[INFO_T[ch.topic]] = dict(k=list(msg.k), d=list(msg.d), width=msg.width,
                                              distortion_model=msg.distortion_model)
            elif ch.topic == TF_STATIC and not tfs:
                for t in msg.transforms:
                    tr, r = t.transform.translation, t.transform.rotation
                    tfs[f'{t.header.frame_id} -> {t.child_frame_id}'] = dict(
                        translation=dict(x=tr.x, y=tr.y, z=tr.z), rotation=dict(x=r.x, y=r.y, z=r.z, w=r.w))
            elif ch.topic == BOUNDS and not bounds:
                for m in msg.markers:
                    bounds[m.ns] = np.array([[p.x, p.y, p.z] for p in m.points])
            if len(info) == len(CAMS) and tfs and (bounds or BOUNDS not in want):
                break
    if len(info) < len(CAMS) or not tfs:
        raise RuntimeError(f'{path}: camera_info for {sorted(info)}, tf_static {bool(tfs)}')
    source = 'bag'
    if not bounds:
        z = np.load(os.path.join(MODEL, 'track_bounds.npz'))
        bounds, source = {k: z[k] for k in z.files}, 'model/track_bounds.npz'
    return info, tfs, bounds, source


def state_source(path):
    have = topics(path)[0]
    if STATE in have:
        return STATE
    if TWIST in have:
        return TWIST
    return None


def stream(path, t0=None, t1=None):
    """Yields (stamp_ns, {cam: jpeg bytes}, [car-state dicts with stamp <= stamp_ns, not yielded before]) in stamp
    order. t0 / t1: window in log time (ns), None = whole bag. Reads in a background thread."""
    src = state_source(path)
    tps = list(IMG_T) + ([src] if src else [])
    q = queue.Queue(maxsize=64)
    stop = threading.Event()

    def run():
        pending, state_buf = {}, []

        def emit(key):
            s = pending.pop(key)
            stamp = max(v[0] for v in s.values())    # newest image of the set: the pose uses no image after its stamp
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

        start = None if t0 is None else t0 - STATE_LEAD_NS
        f, it = _reader(path, topics=tps, start_time=start, end_time=t1, log_time_order=True)
        try:
            with f:
                for _, ch, m, msg in it:
                    if stop.is_set():
                        break
                    if ch.topic in IMG_T:
                        if t0 is not None and m.log_time < t0:
                            continue
                        s = stamp_of(msg)
                        key = next((k for k in pending if abs(k - s) < SET_TOL_NS), s)
                        pending.setdefault(key, {})[IMG_T[ch.topic]] = (s, bytes(msg.data))
                        flush()
                        continue
                    st = stamp_of(msg)
                    if ch.topic == STATE:
                        d = {'stamp_ns': st}
                        for k in ALLOWED:
                            d[k] = float(getattr(msg, k))
                    else:                                    # validation/twist: body-frame velocity, yaw rate
                        w = msg.twist
                        d = dict(stamp_ns=st, vx_mps=w.linear.x, vy_mps=w.linear.y, wz_radps=w.angular.z)
                    state_buf.append(d)
                    if len(state_buf) > 1 and state_buf[-1]['stamp_ns'] < state_buf[-2]['stamp_ns']:
                        state_buf.sort(key=lambda m: m['stamp_ns'])
                flush(force=True)
        finally:
            q.put(None)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            yield item
    finally:
        stop.set()
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                th.join(0.05)


def read_gt(path, t0=None, t1=None):
    """map -> constructor0/base_link from /tf, as the organisers' tools/extract_gt.py, sorted by stamp (as their
    tools/evaluate.py sorts): stamps (N,) int64 ns, poses (N, 4) x, y, z, yaw. Evaluation only - the localizer never
    sees it."""
    ts, rows = [], []
    f, it = _reader(path, topics=['/tf'], start_time=t0, end_time=t1, log_time_order=True)
    with f:
        for _, _, _, msg in it:
            for t in msg.transforms:
                if t.header.frame_id == 'map' and t.child_frame_id == 'constructor0/base_link':
                    tr, r = t.transform.translation, t.transform.rotation
                    yaw = np.arctan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y * r.y + r.z * r.z))
                    ts.append(stamp_of(t)); rows.append((tr.x, tr.y, tr.z, yaw))
    ts, rows = np.array(ts, np.int64), np.array(rows, float).reshape(-1, 4)
    o = np.argsort(ts, kind='stable')
    return ts[o], rows[o]
