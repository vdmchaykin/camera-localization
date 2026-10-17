"""v7 localizer: v6 with the map points on bright blobs removed.

Same interface as sim/my_loc.py ... v6/my_loc_v6.py (inherited, none of them is modified):
    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

Hypothesis (v6 stress test: removing bright blobs from the images made K5 better): many bright map points are
lamps or reflections of the floodlights on the asphalt / car body. A reflection moves with the viewpoint, so its
triangulated 3D point is wrong and biases the pose; lamps also look different by day. v7 drops every map point
whose 5x5 raw-grey peak in its keyframe is >= V7_PEAK (v6/model/light_<cam>.npz, built by
v6/build_light_table.py). The pruning is applied to the loaded map (guided and global path alike).
  V7_PEAK=200   threshold (0 = no pruning, i.e. v6)
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'sim'), os.path.join(ROOT, 'v2'), os.path.join(ROOT, 'v3'),
          os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'v6')):
    sys.path.insert(0, p)
import my_loc_v6 as v6  # noqa: E402
from camera_model import CAMS  # noqa: E402

PEAK = int(os.environ.get('V7_PEAK', '200'))


class Localizer(v6.Localizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        t0 = time.perf_counter()
        self.pruned = {}
        if PEAK > 0:
            for c in CAMS:
                d = self.reloc.db.db[c]
                peak = np.load(os.path.join(ROOT, 'v6', 'model', f'light_{c}.npz'))['peak']
                assert len(peak) == len(d['xyz']), f'light table of {c} does not match the map'
                keep = peak < PEAK
                cnt = np.add.reduceat(keep.astype(np.int64), d['off'][:-1]) if len(keep) else np.zeros(0, int)
                cnt[np.diff(d['off']) == 0] = 0                       # reduceat on empty slices returns an element
                off = np.r_[0, np.cumsum(cnt)]
                for key in ('kps', 'desc', 'xyz', 'par'):
                    if key in d:
                        d[key] = d[key][keep]
                d['off'] = off
                self.pruned[c] = round(float(1 - keep.mean()) * 100, 1)
        self.init_s += time.perf_counter() - t0

    def summary(self):
        s = super().summary()
        s.update(v7=dict(peak=PEAK, pruned_pct=self.pruned))
        return s
