#!/usr/bin/env python3
"""v1 vs v2, local PC vs cluster: one table from the simulate.py summaries.

    rsync -a alder.informatik.uni-bonn.de:better_gpu/v2/runs_cluster/ v2/runs_cluster/
    python3 v2/compare.py      -> table on stdout + v2/compare.md
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOCAL = {('v1', 'K1'): 'v2/runs/v1_K1', ('v2', 'K1'): 'v2/runs/v2_K1',
         ('v1', 'K5'): 'v2/runs/v1_K5', ('v2', 'K5'): 'v2/runs/v2_K5',
         ('v1', 'K3*'): 'sim/runs/val_K3', ('v2', 'K3*'): 'v2/runs/v2_val_K3',
         ('v1', 'K6*'): 'sim/runs/val_K6', ('v2', 'K6*'): 'v2/runs/v2_val_K6'}
CLUSTER = {k: 'v2/runs_cluster/' + {'K1': '{}_K1', 'K5': '{}_K5', 'K3*': '{}_val_K3', 'K6*': '{}_val_K6'}[k[1]].format(k[0])
           for k in LOCAL}
HOSTS = [('PC, GTX 1080', LOCAL), ('cluster, RTX 5060 Ti', CLUSTER)]


def load(d):
    p = os.path.join(ROOT, d, 'summary.json')
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    hdr = ('| машина | версия | круг | pos RMSE | lat | lon | yaw° | <0.25 м | поправка, мс (p95) | поправок/с | '
           'возраст поправки, мс | шаг+jpeg, мс (p95) | выброшено |')
    rows = [hdr, '|' + '---|' * 13]
    for host, runs in HOSTS:
        for lap in ('K1', 'K5', 'K3*', 'K6*'):
            for ver in ('v1', 'v2'):
                s = load(runs[(ver, lap)])
                if s is None:
                    rows.append(f'| {host} | {ver} | {lap} | — нет прогона | | | | | | | | | |')
                    continue
                L = s.get('localizer_stats', {})
                ev = s.get('evaluate', '')
                w25 = next((l.split()[-2] for l in ev.splitlines() if l.startswith('within 0.25')), '')
                dur = s['window_s'][1] - s['window_s'][0]
                rows.append(f"| {host} | {ver} | {lap} | **{s['pos_rmse']:.3f}** | {s['lat_rmse']:.3f} | {s['lon_rmse']:.3f} | "
                            f"{s['yaw_rmse_deg']:.3f} | {w25} % | {L.get('reloc_ms_mean', float('nan')):.0f} "
                            f"({L.get('reloc_ms_p95', float('nan')):.0f}) | {L.get('relocalizations', 0) / dur:.1f} | "
                            f"{L.get('fuse_delay_ms_mean', float('nan')):.0f} | {s['total_ms_mean']:.1f} ({s['total_ms_p95']:.1f}) | "
                            f"{s['dropped']} |")
    txt = '\n'.join(rows) + '\n\n`*` held-out lap (not in the map, not used for tuning).\n'
    open(os.path.join(HERE, 'compare.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
