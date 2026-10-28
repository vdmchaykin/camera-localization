#!/usr/bin/env python3
"""Real-time view of one or several localizer runs: track map, zoom on the car with the predicted heading,
cameras fl / fr / r.

Replays the camera frames of an mcap at their stamps together with the poses of prediction CSVs
(timestamp_ns,x,y,yaw). Each camera set is shown with the nearest pose of each model (within 5 ms for the first
model, within 60 ms for the others: some pipelines stamp a pose with the newest image of the set). Visualisation
only.

    # hidden-test example (daytime, no GT), MP4 + live window
    python3 live_vis.py --mcap hidden_test/data/rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap \
        --pred hidden_test/runs/v6_rerun/pred.csv --title "v6, hidden-test example (day)" --out hidden_test/runs/v6_rerun/live.mp4 --show
    # held-out lap K3 with GT, two models side by side
    python3 live_vis.py --mcap $MCAP --gt gt.csv --pred v6/runs_val/v6_K3/pred.csv --label v6 \
        --pred v11_Vlad/mix_v7.csv --label v11 --title "held-out lap K3 (night)" --out compare_K3.mp4

Layout (1920x1080): cameras fl | fr | r on top; full track map (bottom left, with an |error| plot of the last 30 s
when GT is given); zoom on the car, north up or heading up (bottom right): car footprint, heading arrow, path
projected 1 s ahead from the current speed and yaw rate, trail of the last 10 s, per model. GT: green outline.
--show: OpenCV window paced at --speed (q quit, space pause). --out: MP4 at the camera rate (10 fps = real time).
Speed shown: v_mps from frames.csv next to the first CSV if present, else from its poses.
"""
import argparse, json, os, subprocess, time
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = os.path.dirname(os.path.abspath(__file__))
CAMS = [('fl', 'front left'), ('fr', 'front right'), ('r', 'rear')]
TOPIC = {f'/constructor0/sensor/camera_{c}/compressed_image': c for c, _ in CAMS}
SET_TOL_NS = 5_000_000
OTHER_TOL_NS = 60_000_000
W, H = 1920, 1080
CW, CH = W // 3, round(W // 3 * 728 / 1506)          # camera tiles
PY, PH, PW = CH, H - CH, W // 2                      # bottom panels
CAR_L, CAR_W = 4.9, 1.9                              # m, footprint
# BGR
BG, ROAD, PIT, EDGE, TXT, DIM = (26, 24, 22), (62, 60, 58), (58, 62, 66), (165, 165, 165), (235, 235, 235), (150, 150, 150)
COLORS = [(40, 160, 255), (255, 200, 60), (200, 110, 255), (80, 230, 230)]     # orange, cyan, pink, yellow
GTC = (100, 215, 100)
RINGS = [(0.1, (110, 200, 110)), (0.25, (60, 205, 230)), (0.5, (80, 80, 235))]   # error target: green, yellow, red
FONT = cv2.FONT_HERSHEY_SIMPLEX
PLOT_S = 30.0


def stamp_of(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def camera_sets(path, t0=None, t1=None):
    """yield (stamp, {cam: jpeg bytes}) for fl / fr / r, grouped by header stamp, in stamp order."""
    pending = {}
    with open(path, 'rb') as f:
        for _, ch, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=list(TOPIC), start_time=t0, end_time=t1, log_time_order=True):
            s = stamp_of(msg)
            key = next((k for k in pending if abs(k - s) < SET_TOL_NS), s)
            pending.setdefault(key, {})[TOPIC[ch.topic]] = bytes(msg.data)
            for k in sorted(pending):
                if len(pending[k]) == len(CAMS) or s - k > 150_000_000:
                    yield k, pending.pop(k)
                else:
                    break
    for k in sorted(pending):
        yield k, pending[k]


class Track:
    def __init__(self, path):
        b = {k: np.array(v['points'])[:, :2] for k, v in json.load(open(path)).items()}
        self.loops = [b['track_left'], b['track_right']]
        self.pit = np.vstack([b['pitlane_left'], b['pitlane_right'][::-1]]) if 'pitlane_left' in b else None
        self.lines = list(b.values())
        pts = np.vstack(self.lines)
        self.lo, self.hi = pts.min(0), pts.max(0)

    def draw(self, img, tf, edge_w=1):
        """tf: (N,2) map xy -> (N,2) px float."""
        P = lambda a: np.round(tf(a) * 16).astype(np.int32)     # 4 fractional bits for smooth AA lines
        if self.pit is not None:
            cv2.fillPoly(img, [P(self.pit)], PIT, cv2.LINE_AA, 4)
        cv2.fillPoly(img, [P(l) for l in self.loops], ROAD, cv2.LINE_AA, 4)   # two loops -> the ring between them
        for l in self.lines:
            cv2.polylines(img, [P(l)], False, EDGE, edge_w, cv2.LINE_AA, 4)


def car_poly(x, y, yaw, length=CAR_L, width=CAR_W):
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[-0.5, -0.5], [0.3, -0.5], [0.5, 0.0], [0.3, 0.5], [-0.5, 0.5]]) * [length, width]
    return np.c_[x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1]]


def text(img, s, org, scale=0.6, col=TXT, th=1):
    """text on a translucent dark plate (an outline would widen the glyph advance and smear the letters)."""
    (w, h), b = cv2.getTextSize(s, FONT, scale, th)
    x0, y0, x1, y1 = max(org[0] - 5, 0), max(org[1] - h - 6, 0), min(org[0] + w + 5, img.shape[1]), min(org[1] + b + 3, img.shape[0])
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.45).astype(np.uint8)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def text_parts(img, parts, org, scale=0.4, th=1):
    """one line of differently coloured pieces [(words, colour), ...] on one translucent plate."""
    sizes = [cv2.getTextSize(s, FONT, scale, th) for s, _ in parts]
    w = sum(sz[0][0] for sz in sizes)
    h, b = max(sz[0][1] for sz in sizes), max(sz[1] for sz in sizes)
    x0, y0 = max(org[0] - 5, 0), max(org[1] - h - 6, 0)
    x1, y1 = min(org[0] + w + 5, img.shape[1]), min(org[1] + b + 3, img.shape[0])
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.45).astype(np.uint8)
    x = org[0]
    for (s, col), ((sw, _), _) in zip(parts, sizes):
        cv2.putText(img, s, (x, org[1]), FONT, scale, col, th, cv2.LINE_AA)
        x += sw


_MATH = {}


def math_img(tex, size=15, color=TXT):
    """LaTeX (matplotlib mathtext) rendered once to a BGRA image, transparent background."""
    key = (tex, size, color)
    if key not in _MATH:
        import io
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(dpi=100)
        FigureCanvasAgg(fig)
        fig.text(0, 0, tex, fontsize=size, color=tuple(c / 255 for c in color[::-1]))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.02)
        _MATH[key] = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    return _MATH[key]


def paste_math(img, tex, x, y_mid, size=15):
    """math_img on a translucent dark plate (like text()), left edge x, vertically centred on y_mid."""
    m = math_img(tex, size)
    h, w = m.shape[:2]
    y0 = int(round(y_mid - h / 2))
    y1, x1 = min(y0 + h, img.shape[0]), min(x + w, img.shape[1])
    if y1 <= max(y0, 0) or x1 <= x:
        return
    m = m[max(-y0, 0):y1 - y0, :x1 - x]
    y0 = max(y0, 0)
    a = m[..., 3:4].astype(np.float32) / 255
    roi = img[y0 - 3:y1 + 3, x - 5:x1 + 5]
    roi[:] = (roi * 0.45).astype(np.uint8)
    img[y0:y1, x:x1] = (img[y0:y1, x:x1] * (1 - a) + m[..., :3] * a).astype(np.uint8)


def arrow(img, p0, p1, col, th=2):
    cv2.arrowedLine(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), col, th, cv2.LINE_AA,
                    tipLength=0.25)


def P16(a):
    return np.round(np.asarray(a) * 16).astype(np.int32)


class Pred:
    def __init__(self, path, label, color, tol_ns):
        d = pd.read_csv(path).sort_values('timestamp_ns').drop_duplicates('timestamp_ns').reset_index(drop=True)
        self.label, self.color, self.tol = label, color, tol_ns
        self.t = d.timestamp_ns.values.astype(np.int64)
        self.x, self.y = d.x.values, d.y.values
        self.yaw = np.unwrap(d.yaw.values)
        ts = self.t / 1e9
        # speed / yaw rate from the poses, centred over +-0.3 s (display only)
        i0 = np.clip(np.searchsorted(self.t, self.t - 300_000_000), 0, len(ts) - 1)
        i1 = np.clip(np.searchsorted(self.t, self.t + 300_000_000) - 1, 0, len(ts) - 1)
        dt = np.maximum(ts[i1] - ts[i0], 1e-3)
        self.v = np.hypot(self.x[i1] - self.x[i0], self.y[i1] - self.y[i0]) / dt
        self.w = (self.yaw[i1] - self.yaw[i0]) / dt
        self.step_ms = None
        fr = os.path.join(os.path.dirname(os.path.abspath(path)), 'frames.csv')
        if os.path.exists(fr):
            f = pd.read_csv(fr).set_index('timestamp_ns')
            f = f[~f.index.duplicated()]
            if 'step_ms' in f:
                self.step_ms = f.step_ms.reindex(self.t).values
            if 'v_mps' in f and f.v_mps.notna().mean() > 0.5:
                self.v = f.v_mps.reindex(self.t).values

    def find(self, stamp):
        i = np.searchsorted(self.t, stamp)
        js = [j for j in (i - 1, i) if 0 <= j < len(self.t) and abs(self.t[j] - stamp) < self.tol]
        return min(js, key=lambda j: abs(self.t[j] - stamp)) if js else None


class GT:
    def __init__(self, path):
        g = pd.read_csv(path).sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
        self.t = g.timestamp_ns.values.astype(np.int64)
        self.x, self.y, self.yaw = g.x.values, g.y.values, np.unwrap(g.yaw.values)

    def at(self, t):
        return (np.interp(t, self.t, self.x), np.interp(t, self.t, self.y), np.interp(t, self.t, self.yaw))

    def window(self, t0, t1):
        a, b = np.searchsorted(self.t, [t0, t1])
        return np.c_[self.x[a:b:5], self.y[a:b:5]]

    def err(self, t, x, y, yaw):
        gx, gy, gyaw = self.at(t)
        dx, dy = x - gx, y - gy
        return (np.hypot(dx, dy), -np.sin(gyaw) * dx + np.cos(gyaw) * dy,
                np.degrees(np.angle(np.exp(1j * (yaw - gyaw)))), np.cos(gyaw) * dx + np.sin(gyaw) * dy)


class View:
    def __init__(self, track, preds, gt, title, zoom_m, heading_up):
        self.track, self.preds, self.gt, self.title = track, preds, gt, title
        self.zoom_m, self.heading_up = zoom_m, heading_up
        m = 30
        span = track.hi - track.lo
        self.msc = min((PW - 2 * m) / span[0], (PH - 2 * m - 30) / span[1])
        off = np.array([(PW - span[0] * self.msc) / 2, 30 + (PH - 30 - span[1] * self.msc) / 2])
        self.mtf = lambda a: np.c_[off[0] + (np.atleast_2d(a)[:, 0] - track.lo[0]) * self.msc,
                                   off[1] + (track.hi[1] - np.atleast_2d(a)[:, 1]) * self.msc]
        self.map_base = np.full((PH, PW, 3), BG, np.uint8)
        track.draw(self.map_base, self.mtf)
        self.zsc = PW / (2 * zoom_m)
        self.last_imgs = {}
        self.hist = [[] for _ in preds]          # (t, |err|) per model, for the plot
        self.tgt = [[] for _ in preds]           # (t, lateral, longitudinal) per model, for the target
        self.t_start = None

    def ztf(self, cx, cy, cyaw):
        rot = (np.pi / 2 - cyaw) if self.heading_up else 0.0
        c, s = np.cos(rot), np.sin(rot)

        def f(a):
            d = np.atleast_2d(a)[:, :2] - [cx, cy]
            r = np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]
            return np.c_[PW / 2 + r[:, 0] * self.zsc, PH / 2 - r[:, 1] * self.zsc]
        return f, rot

    def cameras(self, im, jpegs):
        for i, (c, name) in enumerate(CAMS):
            if jpegs.get(c):
                self.last_imgs[c] = cv2.resize(cv2.imdecode(np.frombuffer(jpegs[c], np.uint8), cv2.IMREAD_COLOR),
                                               (CW, CH), interpolation=cv2.INTER_AREA)
            if c in self.last_imgs:
                im[:CH, i * CW:(i + 1) * CW] = self.last_imgs[c]
            text(im, f'{name} ({c})', (i * CW + 12, 28), 0.7, TXT, 2)
            if i:
                cv2.line(im, (i * CW, 0), (i * CW, CH), BG, 3)

    def plot(self, mp, t):
        """|error| of each model over the last PLOT_S seconds (top right of the map panel)."""
        x0, y0, pw, ph = PW - 330, 46, 310, 170
        sub = mp[y0:y0 + ph, x0:x0 + pw]
        sub[:] = (sub * 0.5).astype(np.uint8)
        cv2.rectangle(mp, (x0, y0), (x0 + pw, y0 + ph), (70, 70, 70), 1)
        now = (t - self.t_start) / 1e9
        vals = [np.array(h) for h in self.hist if h]
        vals = [v[v[:, 0] > now - PLOT_S] for v in vals]
        ymax = max(0.2, max((float(np.percentile(v[:, 1], 99)) for v in vals if len(v)), default=0.2) * 1.25)
        for frac in (0.5,):
            yy = int(y0 + ph - frac * ph)
            cv2.line(mp, (x0, yy), (x0 + pw, yy), (55, 55, 55), 1)
        for v, p in zip([np.array(h) for h in self.hist], self.preds):
            if len(v) < 2:
                continue
            v = v[v[:, 0] > now - PLOT_S]
            u = x0 + (v[:, 0] - (now - PLOT_S)) / PLOT_S * pw
            w = y0 + ph - np.minimum(v[:, 1] / ymax, 1) * ph
            cv2.polylines(mp, [P16(np.c_[u, w])], False, p.color, 1, cv2.LINE_AA, 4)
        text(mp, f'|error|, last {PLOT_S:.0f} s (max {ymax:.2f} m)', (x0 + 6, y0 + 18), 0.45, DIM)
        # target: error in the car frame (lateral right/left, longitudinal ahead/behind), last 3 s
        R_M, sz = 0.5, 250
        tx, ty = x0 + (pw - sz) // 2 + sz // 2, y0 + ph + 30 + sz // 2
        sub = mp[ty - sz // 2:ty + sz // 2, tx - sz // 2:tx + sz // 2]
        sub[:] = (sub * 0.5).astype(np.uint8)
        sc = sz / 2 / R_M
        for r, col in RINGS:
            cv2.circle(mp, (tx, ty), int(r * sc), col, 1, cv2.LINE_AA)
        cv2.line(mp, (tx - sz // 2, ty), (tx + sz // 2, ty), (60, 60, 60), 1)
        cv2.line(mp, (tx, ty - sz // 2), (tx, ty + sz // 2), (60, 60, 60), 1)
        for h, p in zip(self.tgt, self.preds):
            v = np.array([e for e in h if e[0] > now - 3.0])
            if not len(v):
                continue
            pts = np.c_[tx - np.clip(v[:, 1], -R_M, R_M) * sc, ty - np.clip(v[:, 2], -R_M, R_M) * sc]
            if len(pts) > 1:
                cv2.polylines(mp, [P16(pts)], False, p.color, 1, cv2.LINE_AA, 4)
            cv2.circle(mp, tuple(np.round(pts[-1]).astype(int)), 6, p.color, -1, cv2.LINE_AA)
        text(mp, 'error in car frame, last 3 s', (tx - sz // 2 + 6, ty - sz // 2 + 18), 0.45, DIM)
        text(mp, 'ahead', (tx + 4, ty - sz // 2 + 40), 0.4, DIM)
        text(mp, 'left', (tx - sz // 2 + 6, ty - 6), 0.4, DIM)
        (_, g1), (_, g2), (_, g3) = RINGS
        text_parts(mp, [('rings ', DIM), ('0.1', g1), (' / ', DIM), ('0.25', g2), (' / ', DIM), ('0.5 m', g3)],
                   (tx - sz // 2 + 6, ty + sz // 2 - 8))

    def render(self, stamp, ks, jpegs):
        """ks: index into each model's poses (None = no pose near this camera set)."""
        live = [(p, k) for p, k in zip(self.preds, ks) if k is not None]
        if self.t_start is None:
            self.t_start = stamp
        if self.gt is not None:
            cx, cy, cyaw = self.gt.at(stamp)
        else:
            p, k = live[0]
            cx, cy, cyaw = p.x[k], p.y[k], p.yaw[k]
        im = np.full((H, W, 3), BG, np.uint8)
        self.cameras(im, jpegs)
        errs = {}
        if self.gt is not None:
            for i, (p, k) in enumerate(zip(self.preds, ks)):
                if k is not None:
                    errs[i] = self.gt.err(p.t[k], p.x[k], p.y[k], p.yaw[k])
                    self.hist[i].append(((stamp - self.t_start) / 1e9, errs[i][0]))
                    self.tgt[i].append(((stamp - self.t_start) / 1e9, errs[i][1], errs[i][3]))
        # ---- full map
        mp = self.map_base.copy()
        if self.gt is not None:
            g = self.gt.window(self.t_start, stamp)
            if len(g) > 1:
                cv2.polylines(mp, [P16(self.mtf(g))], False, GTC, 1, cv2.LINE_AA, 4)
        for p, k in live:
            j0 = np.searchsorted(p.t, self.t_start - OTHER_TOL_NS)
            if k - j0 > 1:
                cv2.polylines(mp, [P16(self.mtf(np.c_[p.x[j0:k + 1], p.y[j0:k + 1]]))], False, p.color, 2,
                              cv2.LINE_AA, 4)
        half = np.array([PW, PH]) / 2 / self.zsc
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
        _, rot = self.ztf(cx, cy, cyaw)
        c_, s_ = np.cos(-rot), np.sin(-rot)
        corners = np.c_[cx + c_ * corners[:, 0] - s_ * corners[:, 1], cy + s_ * corners[:, 0] + c_ * corners[:, 1]]
        cv2.polylines(mp, [np.round(self.mtf(corners)).astype(np.int32)], True, DIM, 1, cv2.LINE_AA)
        for p, k in live[::-1]:
            cp = self.mtf([p.x[k], p.y[k]])[0]
            cv2.circle(mp, tuple(np.round(cp).astype(int)), 7, p.color, -1, cv2.LINE_AA)
            cv2.circle(mp, tuple(np.round(cp).astype(int)), 7, (0, 0, 0), 1, cv2.LINE_AA)
            arrow(mp, cp, cp + 26 * np.array([np.cos(p.yaw[k]), -np.sin(p.yaw[k])]), p.color, 2)
        text(mp, 'track map', (14, 26), 0.7, TXT, 2)
        if self.gt is not None:
            self.plot(mp, stamp)
        im[PY:, :PW] = mp
        # ---- zoom
        f, rot = self.ztf(cx, cy, cyaw)
        zm = np.full((PH, PW, 3), BG, np.uint8)
        self.track.draw(zm, f, 2)
        sb = 10 * self.zsc
        cv2.line(zm, (PW - 30 - int(sb), PH - 30), (PW - 30, PH - 30), TXT, 2, cv2.LINE_AA)
        text(zm, '10 m', (PW - 30 - int(sb / 2) - 22, PH - 40), 0.55)
        for p, k in live:                                     # trails, last 10 s
            j0 = np.searchsorted(p.t, p.t[k] - 10e9)
            if k - j0 > 1:
                cv2.polylines(zm, [P16(f(np.c_[p.x[j0:k + 1], p.y[j0:k + 1]]))], False, p.color, 2, cv2.LINE_AA, 4)
        for p, k in live:                                     # projected path, 1 s (constant speed and yaw rate)
            v, w, x, y, yaw = p.v[k], p.w[k], p.x[k], p.y[k], p.yaw[k]
            if np.isfinite(v) and v > 0.5:
                ts = np.linspace(0, 1.0, 21)
                hh = yaw + w * ts
                dx = np.cumsum(np.r_[0, v * np.cos(hh[:-1]) * np.diff(ts)])
                dy = np.cumsum(np.r_[0, v * np.sin(hh[:-1]) * np.diff(ts)])
                pts = f(np.c_[x + dx, y + dy])
                for a, b in zip(pts[:-1:2], pts[1::2]):      # dashed
                    cv2.line(zm, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), p.color, 2,
                             cv2.LINE_AA)
        multi = len(self.preds) > 1
        for p, k in live[::-1]:                               # first model on top
            poly = P16(f(car_poly(p.x[k], p.y[k], p.yaw[k])))
            if multi:                                         # translucent fill so overlapping cars stay visible
                lay = zm.copy()
                cv2.fillPoly(lay, [poly], p.color, cv2.LINE_AA, 4)
                cv2.addWeighted(lay, 0.55, zm, 0.45, 0, zm)
                cv2.polylines(zm, [poly], True, p.color, 2, cv2.LINE_AA, 4)
            else:
                cv2.fillPoly(zm, [poly], p.color, cv2.LINE_AA, 4)
                cv2.polylines(zm, [poly], True, (0, 0, 0), 1, cv2.LINE_AA, 4)
            c0 = f([p.x[k], p.y[k]])[0]
            hd = np.array([np.cos(p.yaw[k] + rot), -np.sin(p.yaw[k] + rot)])
            arrow(zm, c0 + hd * CAR_L * 0.5 * self.zsc, c0 + hd * (CAR_L * 0.5 + 8) * self.zsc, p.color, 3)
        if self.gt is not None:                               # outline on top: visible even at cm-level error
            cv2.polylines(zm, [P16(f(car_poly(cx, cy, cyaw)))], True, GTC, 2, cv2.LINE_AA, 4)
        na = np.array([PW - 50, 70.0])
        nd = np.array([np.sin(rot), -np.cos(rot)])
        arrow(zm, na - nd * 22, na + nd * 22, DIM, 2)
        text(zm, 'N', tuple(np.round(na + nd * 36 - [6, -6]).astype(int)), 0.55, DIM)
        # info box
        p0, k0 = live[0]
        v0 = p0.v[k0]
        rows = [(self.title, TXT, 0.7, 2),
                (f't = {(stamp - self.t_start) / 1e9:6.1f} s    v = '
                 f'{v0 * 3.6 if np.isfinite(v0) else float("nan"):5.0f} km/h', TXT, 0.6, 1)]
        for i, (p, k) in enumerate(zip(self.preds, ks)):
            if k is None:
                rows.append((f'{p.label}: no pose', p.color, 0.6, 1))
                continue
            s = f'{p.label}: yaw {np.degrees(p.yaw[k]):7.1f} deg'
            if p.step_ms is not None and np.isfinite(p.step_ms[k]):
                s += f'   step {p.step_ms[k]:5.1f} ms'
            if i in errs:
                e, lat, dyaw, _ = errs[i]
                s = f'{p.label}: error {e:.2f} m   lateral {lat:+.2f} m   yaw {dyaw:+.2f} deg'
            rows.append((s, p.color, 0.6, 1))
        if not multi and self.gt is None:
            rows.insert(2, (f'x = {p0.x[k0]:8.2f} m   y = {p0.y[k0]:8.2f} m', TXT, 0.6, 1))
        for i, (s, col, sc, th) in enumerate(rows):
            text(zm, s, (14, 28 + 26 * i), sc, col, th)
        leg = [(p.label, p.color) for p in self.preds] + ([('GT (outline)', GTC)] if self.gt is not None else [])
        for i, (s, col) in enumerate(leg):
            yy = PH - 30 - 24 * (len(leg) - 1 - i)
            cv2.rectangle(zm, (14, yy - 12), (32, yy + 2), col, -1)
            text(zm, s, (40, yy), 0.55)
        yy = PH - 30 - 24 * len(leg)                          # dashed line sample = the projected path
        for x0 in (12, 22):
            cv2.line(zm, (x0, yy - 5), (x0 + 6, yy - 5), self.preds[0].color, 2, cv2.LINE_AA)
        paste_math(zm, r'path $1\,\mathrm{s}$ ahead:  $\psi(t)=\psi_0+\omega\,t,\;\; v=\mathrm{const}$', 40, yy - 5, size=12)
        text(zm, f'zoom {2 * self.zoom_m:.0f} m, ' + ('heading up' if self.heading_up else 'north up') +
             (', centred on GT' if self.gt is not None else ''), (14, yy - 24), 0.5, DIM)
        im[PY:, PW:] = zm
        cv2.line(im, (PW, PY), (PW, H), (0, 0, 0), 2)
        cv2.line(im, (0, PY), (W, PY), (0, 0, 0), 2)
        return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mcap', required=True)
    ap.add_argument('--pred', required=True, action='append', help='prediction CSV; repeat to compare models')
    ap.add_argument('--label', action='append', help='name of each --pred (same order)')
    ap.add_argument('--gt', help='GT CSV (timestamp_ns,x,y,...,yaw); optional')
    ap.add_argument('--bounds', default=os.path.join(ROOT, 'export', 'track_bounds.json'))
    ap.add_argument('--title', default='')
    ap.add_argument('--start-ns', type=int, default=None, help='only camera sets in [start, end] (default: first model)')
    ap.add_argument('--end-ns', type=int, default=None)
    ap.add_argument('--out', help='MP4 path')
    ap.add_argument('--show', action='store_true', help='live OpenCV window')
    ap.add_argument('--speed', type=float, default=1.0, help='--show playback speed (1 = real time)')
    ap.add_argument('--zoom-m', type=float, default=35.0, help='half width of the zoom view, m')
    ap.add_argument('--heading-up', action='store_true', help='rotate the zoom view so the car points up')
    a = ap.parse_args()
    assert a.out or a.show, 'give --out and/or --show'
    labels = a.label or []
    labels += [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in a.pred[len(labels):]]
    preds = [Pred(p, l, COLORS[i % len(COLORS)], SET_TOL_NS if i == 0 else OTHER_TOL_NS)
             for i, (p, l) in enumerate(zip(a.pred, labels))]
    t0 = a.start_ns if a.start_ns is not None else int(preds[0].t[0])
    t1 = a.end_ns if a.end_ns is not None else int(preds[0].t[-1])
    view = View(Track(a.bounds), preds, GT(a.gt) if a.gt else None, a.title or ' vs '.join(labels),
                a.zoom_m, a.heading_up)
    ff = None
    if a.out:
        ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                               '-s', f'{W}x{H}', '-r', '10', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                               '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out],
                              stdin=subprocess.PIPE)
    if a.show:
        cv2.namedWindow('live', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('live', W * 2 // 3, H * 2 // 3)
    wall0 = s0 = None
    n = 0
    paused = False
    for stamp, jpegs in camera_sets(a.mcap, t0 - 50_000_000, t1 + 50_000_000):
        if not t0 - SET_TOL_NS <= stamp <= t1 + SET_TOL_NS:
            continue
        ks = [p.find(stamp) for p in preds]
        if ks[0] is None:
            continue
        im = view.render(stamp, ks, jpegs)
        n += 1
        if ff:
            ff.stdin.write(im.tobytes())
        if a.show:
            if wall0 is None:
                wall0, s0 = time.perf_counter(), stamp
            due = wall0 + (stamp - s0) / 1e9 / a.speed
            cv2.imshow('live', im)
            while True:
                key = cv2.waitKey(max(1, int((due - time.perf_counter()) * 1000))) & 0xFF
                if key == ord('q'):
                    a.show = False
                    break
                if key == ord(' '):
                    paused = not paused
                    if not paused:
                        wall0 += time.perf_counter() - due
                        due = time.perf_counter()
                if not paused and time.perf_counter() >= due:
                    break
            if not a.show and not ff:
                break
    if ff:
        ff.stdin.close()
        ff.wait()
    cv2.destroyAllWindows()
    print(f'{n} frames' + (f' -> {a.out}' if a.out else ''))


if __name__ == '__main__':
    main()
