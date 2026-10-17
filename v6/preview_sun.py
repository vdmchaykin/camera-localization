#!/usr/bin/env python3
"""Preview of the synthetic sun perturbations (v6/sun_stress.py) on one camera set: v6/sun_preview.jpg
python3 v6/preview_sun.py [--t 300]   (t: seconds from the start of the bag)"""
import argparse, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'sim')]
os.environ['STRESS'] = ''
from simulate import read_static
from camera_model import CAMS, H, W, Camera, tf_mat
import sun_stress as S
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
ap = argparse.ArgumentParser(); ap.add_argument('--t', type=float, default=300); ap.add_argument('--yaw', type=float, default=0.0)
a = ap.parse_args()
M = os.path.join(ROOT, 'data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap')
with open(M, 'rb') as f:
    t0 = make_reader(f).get_summary().statistics.message_start_time + int(a.t * 1e9)
info, tfs, _ = read_static(M, t0)
imgs = {}
with open(M, 'rb') as f:
    for _, ch, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=[f'/constructor0/sensor/camera_{c}/compressed_image' for c in CAMS], start_time=t0):
        c = ch.topic.split('camera_')[1].split('/')[0]
        imgs.setdefault(c, cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR))
        if len(imgs) == len(CAMS): break
cams = {}
for c in CAMS:
    t = tfs[f'constructor0/base_link -> constructor0/camera_{c}']; r, p = t['rotation'], t['translation']
    cams[c] = Camera(c, info[c], tf_mat([r['x'], r['y'], r['z'], r['w']], [p['x'], p['y'], p['z']]))
rows = []
for c in ('fl', 'cl', 'cr', 'r'):
    h = cv2.resize(imgs[c], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    outs = [h, S.day(h), S.shadow(S.day(h), 7), S.glare(S.shadow(S.day(h), 7), cams[c], a.yaw)]
    rows.append(np.hstack([(np.clip(o, 0, 1) * 255).astype(np.uint8)[::2, ::2] for o in outs]))
im = np.vstack(rows)
for k, s in enumerate(('night (original)', 'day', 'day+shadow', 'day+shadow+glare')):
    cv2.putText(im, s, (k * (W // 2) + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(HERE, 'sun_preview.jpg'), im)
print('cams rows: fl, cl, cr, r ->', os.path.join(HERE, 'sun_preview.jpg'))
