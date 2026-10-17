#!/usr/bin/env python3
"""Per-lap metrics of a simulate.py run (evaluation only: uses GT and analysis/laps.csv).

    python3 sim/per_lap.py sim/runs/B_state_full [more run dirs ...]

K3, K6 are held out (never in the map) -> honest numbers. Other laps: the map contains other laps of the same
recording but not the query's own pass (map_config.json: self_exclude_s), i.e. leave-pass-out.
"""
import os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from simulate import EVAL, GT, ROOT, errors  # noqa: E402


def main(dirs):
    gt = GT(os.path.join(ROOT, 'gt.csv'))
    laps = pd.read_csv(os.path.join(ROOT, 'analysis', 'laps.csv'))
    for d in dirs:
        p = pd.read_csv(os.path.join(d, 'pred.csv'))
        t = (p.timestamp_ns.values - gt.t0) / 1e9
        e = errors(gt, p.timestamp_ns.values, p.x.values, p.y.values, p.yaw.values)
        lap = np.clip(np.searchsorted(laps.t_start.values, t, side='right') - 1, 0, len(laps) - 1)
        print(f'== {os.path.basename(d)}')
        print(f'{"lap":4s} {"n":>5s} {"pos":>7s} {"lat":>7s} {"lon":>7s} {"yaw°":>6s} {"<1m%":>6s} {"max":>7s}')
        rms = lambda v: np.sqrt(np.mean(v ** 2))
        for L in np.unique(lap):
            m = (lap == L) & e['ok']
            tag = f'K{L}' + ('*' if L in (3, 6) else '')
            print(f'{tag:4s} {m.sum():5d} {rms(e["pos"][m]):7.3f} {rms(e["lat"][m]):7.3f} {rms(e["lon"][m]):7.3f} '
                  f'{rms(e["dyaw"][m]):6.3f} {100 * np.mean(e["pos"][m] < 1):6.1f} {e["pos"][m].max():7.2f}')
            if L in (3, 6):
                sub = os.path.join(d, f'pred_K{L}.csv')
                p[lap == L].to_csv(sub, index=False)
                g = gt.df[(gt.df.timestamp_ns >= p.timestamp_ns[lap == L].min() - 10**8) &
                          (gt.df.timestamp_ns <= p.timestamp_ns[lap == L].max() + 10**8)]
                g.to_csv(os.path.join(d, f'gt_K{L}.csv'), index=False)
                r = subprocess.run([sys.executable, EVAL, os.path.join(d, f'gt_K{L}.csv'), sub],
                                   capture_output=True, text=True).stdout
                open(os.path.join(d, f'evaluate_K{L}.txt'), 'w').write(r)
        print('* held-out lap (official evaluate.py output in evaluate_K3.txt / evaluate_K6.txt)')


if __name__ == '__main__':
    main(sys.argv[1:])
