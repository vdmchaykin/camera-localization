#!/usr/bin/env python3
"""One pass over the camloc mcap -> small export folder for analysis.

    pip install mcap mcap-ros2-support numpy
    python3 export_lite.py data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap export

Writes into <out>/:
  gt_tf.csv               timestamp_ns,log_ns,x,y,z,qx,qy,qz,qw,yaw      (/tf map->base_link)
  state_estimation.csv    every numeric field of StateEstimation, flattened
  image_stamps.csv        cam,stamp_ns,log_ns,bytes                      (all camera frames)
  track_bounds.json       LINE_STRIP points per marker namespace
  tf_static.json          base_link -> sensor transforms
  camera_info.json        first CameraInfo per camera
  schemas/*.msg           message definitions embedded in the mcap
  samples/<cam>/<stamp>.jpg   one frame per camera every SAMPLE_EVERY_S seconds
  summary.json            topic counts and time range
"""
import csv, json, math, os, sys, time
from collections import defaultdict
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

SAMPLE_EVERY_S = 60.0
NS = 'constructor0'


def stamp_ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def fields(o):
    sl = getattr(type(o), '__slots__', None)
    return list(sl) if sl else [k for k in vars(o) if not k.startswith('_')]


def flatten(o, prefix='', out=None):
    """Numeric leaves of a decoded ROS message -> {dotted.name: value}."""
    if out is None:
        out = {}
    if isinstance(o, bool):
        out[prefix] = int(o)
    elif isinstance(o, (int, float)):
        out[prefix] = o
    elif isinstance(o, (str, bytes, bytearray)):
        pass
    elif hasattr(o, '__len__') and not hasattr(o, '__slots__'):
        if len(o) <= 36:
            for i, v in enumerate(o):
                flatten(v, f'{prefix}[{i}]', out)
    else:
        for f in fields(o):
            flatten(getattr(o, f), f'{prefix}.{f}' if prefix else f, out)
    return out


def to_plain(o):
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    if isinstance(o, (bytes, bytearray)):
        return f'<{len(o)} bytes>'
    if hasattr(o, '__len__') and not hasattr(o, '__slots__'):
        return [to_plain(v) for v in o]
    return {f: to_plain(getattr(o, f)) for f in fields(o)}


def main(bag, out):
    os.makedirs(os.path.join(out, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(out, 'schemas'), exist_ok=True)
    f_gt = open(os.path.join(out, 'gt_tf.csv'), 'w', newline='')
    w_gt = csv.writer(f_gt)
    w_gt.writerow(['timestamp_ns', 'log_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw'])
    f_img = open(os.path.join(out, 'image_stamps.csv'), 'w', newline='')
    w_img = csv.writer(f_img)
    w_img.writerow(['cam', 'stamp_ns', 'log_ns', 'bytes'])
    f_se, w_se, se_cols = None, None, None

    counts = defaultdict(int)
    tf_static, cam_info, bounds = {}, {}, None
    next_sample = {}
    t0 = time.time()
    t_first = t_last = None

    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        summ = r.get_summary()
        for s in summ.schemas.values():
            with open(os.path.join(out, 'schemas', s.name.replace('/', '__') + '.msg'), 'wb') as fs:
                fs.write(s.data)

        for schema, ch, m, msg in r.iter_decoded_messages(log_time_order=False):
            topic = ch.topic
            counts[topic] += 1
            t_first = m.log_time if t_first is None else min(t_first, m.log_time)
            t_last = m.log_time if t_last is None else max(t_last, m.log_time)
            n = sum(counts.values())
            if n % 20000 == 0:
                print(f'{n} msgs, {time.time() - t0:.0f} s', flush=True)

            if topic.endswith('/compressed_image'):
                cam = topic.split('/')[-2].replace('camera_', '')
                st = stamp_ns(msg.header)
                w_img.writerow([cam, st, m.log_time, len(msg.data)])
                if st >= next_sample.get(cam, 0):
                    d = os.path.join(out, 'samples', cam)
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, f'{st}.jpg'), 'wb') as fj:
                        fj.write(bytes(msg.data))
                    next_sample[cam] = st + int(SAMPLE_EVERY_S * 1e9)
            elif topic == '/tf':
                for t in msg.transforms:
                    if t.header.frame_id != 'map' or t.child_frame_id != f'{NS}/base_link':
                        continue
                    q, p = t.transform.rotation, t.transform.translation
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    w_gt.writerow([stamp_ns(t.header), m.log_time, f'{p.x:.4f}', f'{p.y:.4f}', f'{p.z:.4f}',
                                   f'{q.x:.6f}', f'{q.y:.6f}', f'{q.z:.6f}', f'{q.w:.6f}', f'{yaw:.6f}'])
            elif topic == f'/{NS}/state_estimation':
                row = flatten(msg)
                if w_se is None:
                    se_cols = ['log_ns'] + list(row.keys())
                    f_se = open(os.path.join(out, 'state_estimation.csv'), 'w', newline='')
                    w_se = csv.writer(f_se)
                    w_se.writerow(se_cols)
                row['log_ns'] = m.log_time
                w_se.writerow([row.get(c, '') for c in se_cols])
            elif topic == '/tf_static':
                for t in msg.transforms:
                    k = f'{t.header.frame_id} -> {t.child_frame_id}'
                    if k not in tf_static:
                        tf_static[k] = to_plain(t.transform)
            elif topic.endswith('/camera_info'):
                cam = topic.split('/')[-2].replace('camera_', '')
                if cam not in cam_info:
                    cam_info[cam] = to_plain(msg)
            elif topic == f'/{NS}/track_bounds' and bounds is None:
                bounds = {}
                for mk in msg.markers:
                    key = mk.ns if mk.ns else str(mk.id)
                    bounds[key] = {'frame': mk.header.frame_id, 'type': mk.type,
                                   'points': [[p.x, p.y, p.z] for p in mk.points]}

    f_gt.close(); f_img.close()
    if f_se:
        f_se.close()
    for name, obj in (('tf_static.json', tf_static), ('camera_info.json', cam_info),
                      ('track_bounds.json', bounds)):
        with open(os.path.join(out, name), 'w') as fo:
            json.dump(obj, fo, indent=1)
    with open(os.path.join(out, 'summary.json'), 'w') as fo:
        json.dump({'counts': counts, 'log_time_first_ns': t_first, 'log_time_last_ns': t_last,
                   'export_seconds': round(time.time() - t0, 1)}, fo, indent=1)
    print(f'done in {time.time() - t0:.0f} s -> {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
