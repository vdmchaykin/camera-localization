#!/usr/bin/env python3
"""Audit camera synchronization and ground-truth timing before training.
Run in tools/: python audit_dataset.py sync_frames.csv frames.csv gt.csv laps.csv ../c0_2026-10-13_22-53-42_camloc_0.mcap
"""
import sys
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

CAMERAS = ('cl', 'cr', 'fl', 'fr', 'r', 'rl', 'rr')

def main():
    if len(sys.argv) != 6:
        raise SystemExit('Usage: python audit_dataset.py sync_frames.csv frames.csv gt.csv laps.csv recording.mcap')
    sync_path, frames_path, gt_path, laps_path, bag_path = sys.argv[1:]
    sync = pd.read_csv(sync_path)
    frames = pd.read_csv(frames_path)
    gt = pd.read_csv(gt_path).sort_values('timestamp_ns')
    laps = pd.read_csv(laps_path).sort_values('t_start')
    errors = []
    def check(test, msg):
        print(('PASS' if test else 'FAIL') + ': ' + msg)
        if not test: errors.append(msg)

    check(len(sync) > 0, f'{len(sync)} synchronized rows')
    check(sync['sample_id'].is_unique, 'unique sample_id')
    check(sync['timestamp_ns'].is_monotonic_increasing, 'chronological sample timestamps')
    check(sync[['x', 'y', 'z', 'yaw']].notna().all().all(), 'GT columns populated')
    check(sync['lap'].isin(laps['lap']).all(), 'lap IDs appear in laps.csv')

    t0_ns = int(np.median(frames['timestamp_ns'].to_numpy(np.int64) -
                          np.rint(frames['t_rel_s'].to_numpy(float)*1e9).astype(np.int64)))
    print('MCAP log-time origin:', t0_ns)
    print('GT first relative time (s):', round((gt['timestamp_ns'].iloc[0] - t0_ns)/1e9, 6))
    print('GT last relative time (s):', round((gt['timestamp_ns'].iloc[-1] - t0_ns)/1e9, 6))
    print('Laps time range (s):', laps['t_start'].min(), laps['t_end'].max())
    check(gt['timestamp_ns'].is_unique, 'unique GT timestamps')

    lap_limits = laps.set_index('lap')
    lim = lap_limits.loc[sync['lap']]
    rel = (sync['timestamp_ns'].to_numpy(np.int64) - t0_ns) / 1e9
    check(np.all((rel >= lim['t_start'].to_numpy()) &
                 (rel <= lim['t_end'].to_numpy())), 'reference frames within assigned lap')
    check(np.all(np.abs(sync['yaw'].to_numpy()) <= np.pi+1e-6), 'yaw within [-pi,pi]')
    for cam in CAMERAS:
        idcol, tcol = f'{cam}_frame_id', f'{cam}_timestamp_ns'
        check(sync[idcol].is_unique, f'camera_{cam}: no reused frame IDs')
        src = frames[frames['camera'].str.endswith(f'/camera_{cam}/compressed_image')].set_index('frame_id')
        ids = sync[idcol].to_numpy(np.int64)
        present = np.isin(ids, src.index.to_numpy(np.int64)).all()
        check(present, f'camera_{cam}: frame IDs found in frames.csv')
        if present:
            true_times = src.loc[ids, 'timestamp_ns'].to_numpy(np.int64)
            check(np.array_equal(true_times, sync[tcol].to_numpy(np.int64)),
                  f'camera_{cam}: timestamps match frames.csv')
        dt_ms = (sync[tcol].to_numpy(np.int64) - sync['timestamp_ns'].to_numpy(np.int64)) / 1e6
        print(f'  camera_{cam}: offset median={np.median(dt_ms):.2f} ms, max_abs={np.max(np.abs(dt_ms)):.2f} ms')
        cam_rel = (sync[tcol].to_numpy(np.int64) - t0_ns)/1e9
        check(np.all((cam_rel >= lim['t_start'].to_numpy()) &
                     (cam_rel <= lim['t_end'].to_numpy())), f'camera_{cam}: images inside assigned lap')

    # Check interpolation is supported by nearby GT measurements.
    times = gt['timestamp_ns'].to_numpy(np.int64)
    query = sync['timestamp_ns'].to_numpy(np.int64)
    ri = np.searchsorted(times, query, side='left')
    inside = (ri > 0) & (ri < len(times))
    check(inside.all(), 'reference times bracketed by GT')
    if inside.all():
        gaps_ms = (times[ri] - times[ri-1])/1e6
        print(f'GT bracket gap ms: median={np.median(gaps_ms):.2f}, max={np.max(gaps_ms):.2f}')
        check((gaps_ms <= 200).all(), 'GT interpolation gaps <=200 ms')

    print('\nSampled MCAP log_time versus message header.stamp offsets:')
    start = t0_ns
    end = int(frames['timestamp_ns'].max())
    probe_times = [start+int(5e9), (start+end)//2, end-int(5e9)]
    topics = [f'/constructor0/sensor/camera_{c}/compressed_image' for c in CAMERAS] + ['/tf']
    observed = {}
    with open(bag_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for center in probe_times:
            # Use MCAP index for small reading windows when present.
            for _, channel, record, message in reader.iter_decoded_messages(
                    topics=topics, start_time=center-int(0.6e9), end_time=center+int(0.6e9)):
                if channel.topic == '/tf':
                    stamp = None
                    for tr in message.transforms:
                        if tr.header.frame_id == 'map' and tr.child_frame_id == 'constructor0/base_link':
                            stamp = tr.header.stamp
                            break
                    if stamp is None: continue
                    key = '/tf map->base_link'
                else:
                    stamp = message.header.stamp
                    key = channel.topic.split('/')[3]
                offset_ms = (record.log_time - (stamp.sec*10**9 + stamp.nanosec))/1e6
                observed.setdefault(key, []).append(offset_ms)
    for topic in sorted(observed):
        arr = np.asarray(observed[topic])
        print(f'  {topic}: log_time - header.stamp: median={np.median(arr):.2f} ms; '
              f'range=[{arr.min():.2f}, {arr.max():.2f}] ms; n={len(arr)}')
    if not observed:
        print('  WARNING: no header-stamp samples decoded; inspect MCAP separately')

    print('\nAUDIT:', 'FAIL: review listed errors' if errors else 'PASS for structural checks; review time offsets above')
    print('NOTE: lap boundaries still need confirmation from their original source or track crossing analysis.')
    if errors: raise SystemExit(1)

if __name__ == '__main__':
    main()
