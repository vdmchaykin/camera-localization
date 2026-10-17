#!/usr/bin/env python3
"""Synchronize seven indexed MCAP camera streams with GT and lap labels.
Usage: python sync_frames.py frames.csv gt.csv laps.csv
"""
import argparse
import numpy as np
import pandas as pd

CAMERAS = ('cl', 'cr', 'fl', 'fr', 'r', 'rl', 'rr')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('frames')
    p.add_argument('gt')
    p.add_argument('laps')
    p.add_argument('--output', default='sync_frames.csv')
    p.add_argument('--tolerance-ms', type=float, default=50)
    p.add_argument('--max-gt-gap-ms', type=float, default=200)
    args = p.parse_args()

    frames = pd.read_csv(args.frames)
    gt = pd.read_csv(args.gt).sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
    laps = pd.read_csv(args.laps).sort_values('t_start')
    if frames.empty or len(gt) < 2:
        raise ValueError('frames.csv is empty or gt.csv has fewer than 2 timestamps')

    # index_frames.py defines t_rel_s against MCAP's earliest log_time.
    t0_candidates = frames['timestamp_ns'].to_numpy(dtype=np.int64) - np.rint(
        frames['t_rel_s'].to_numpy(dtype=float) * 1e9
    ).astype(np.int64)
    t0_ns = int(np.median(t0_candidates))
    if np.ptp(t0_candidates) > 3000:
        raise ValueError('Inconsistent t_rel_s/timestamp_ns: inspect frame indexing')
    print(f'MCAP time origin: {t0_ns} ns')

    streams = {}
    for name in CAMERAS:
        suffix = f'/camera_{name}/compressed_image'
        part = frames[frames['camera'].str.endswith(suffix)].copy()
        if part.empty:
            raise ValueError(f'No images from camera_{name}')
        part = part.sort_values('timestamp_ns').reset_index(drop=True)
        part = part.rename(columns={'timestamp_ns': f'{name}_timestamp_ns', 'frame_id': f'{name}_frame_id'})
        streams[name] = part[[f'{name}_timestamp_ns', f'{name}_frame_id']]

    out = streams['cl'].copy()
    out['timestamp_ns'] = out['cl_timestamp_ns']
    tolerance = int(args.tolerance_ms * 1e6)
    for name in CAMERAS:
        if name == 'cl':
            continue
        right = streams[name].rename(columns={f'{name}_timestamp_ns': 'match_ns'})
        out = pd.merge_asof(out.sort_values('timestamp_ns'), right.sort_values('match_ns'),
                            left_on='timestamp_ns', right_on='match_ns', direction='nearest',
                            tolerance=tolerance)
        out = out.rename(columns={'match_ns': f'{name}_timestamp_ns'})

    before = len(out)
    frame_columns = [f'{c}_frame_id' for c in CAMERAS]
    out = out.dropna(subset=frame_columns).copy()
    for name in CAMERAS:
        out[f'{name}_frame_id'] = out[f'{name}_frame_id'].astype('int64')
        out[f'{name}_timestamp_ns'] = out[f'{name}_timestamp_ns'].astype('int64')
    # Prevent an image being reused in adjacent bundles.
    for name in CAMERAS:
        used = out[f'{name}_frame_id'].duplicated(keep=False)
        if used.any():
            print(f'Warning: {used.sum()} bundles reuse camera_{name}; dropping duplicates')
            out = out.loc[~used].copy()

    out['t_rel_s'] = (out['timestamp_ns'] - t0_ns) / 1e9
    out['lap'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out['complete'] = False
    for i, row in laps.iterrows():
        # Final endpoint is included so the last exact-timestamp frame is retained.
        inside = ((out['t_rel_s'] >= row['t_start']) &
                  (out['t_rel_s'] <= row['t_end'] if i == laps.index[-1]
                   else out['t_rel_s'] < row['t_end']))
        out.loc[inside, 'lap'] = int(row['lap'])
        out.loc[inside, 'complete'] = bool(row['complete'])
    out = out.dropna(subset=['lap']).copy()
    out['lap'] = out['lap'].astype(int)

    # Do not combine camera images across lap boundaries.
    for name in CAMERAS:
        rel = (out[f'{name}_timestamp_ns'] - t0_ns) / 1e9
        limits = laps.set_index('lap').loc[out['lap']]
        out = out[(rel >= limits['t_start'].to_numpy()) &
                  (rel <= limits['t_end'].to_numpy())].copy()

    times = gt['timestamp_ns'].to_numpy(dtype=np.int64)
    query = out['timestamp_ns'].to_numpy(dtype=np.int64)
    right = np.searchsorted(times, query, side='left')
    valid = (right > 0) & (right < len(times))
    ri = np.clip(right, 1, len(times) - 1)
    li = ri - 1
    max_gap = int(args.max_gt_gap_ms * 1e6)
    valid &= (times[ri] - times[li]) <= max_gap
    out = out.loc[valid].copy()
    query, li, ri = query[valid], li[valid], ri[valid]
    alpha = (query - times[li]) / (times[ri] - times[li]).astype(float)
    for col in ('x', 'y', 'z'):
        values = gt[col].to_numpy(dtype=float)
        out[col] = values[li] + alpha * (values[ri] - values[li])
    yaw = np.unwrap(gt['yaw'].to_numpy(dtype=float))
    interpolated = yaw[li] + alpha * (yaw[ri] - yaw[li])
    out['yaw'] = (interpolated + np.pi) % (2 * np.pi) - np.pi
    out['max_camera_offset_ms'] = np.maximum.reduce([
        np.abs(out[f'{name}_timestamp_ns'].to_numpy() - out['timestamp_ns'].to_numpy()) / 1e6
        for name in CAMERAS
    ])
    out = out.sort_values('timestamp_ns').reset_index(drop=True)
    out.insert(0, 'sample_id', np.arange(len(out)))
    out.to_csv(args.output, index=False)
    print(f'Complete 7-camera samples: {len(out)} / {before}')
    print('Missing/unsynchronized/invalid-GT samples:', before - len(out))
    print('Samples by lap:\n', out.groupby('lap').size().to_string())
    print('Worst camera offset, ms:', round(out['max_camera_offset_ms'].max(), 3) if len(out) else 'n/a')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
