#!/usr/bin/env python3
"""
Dump /constructor0/state_estimation to data/state.parquet (all numeric fields, flattened)
and find which columns are speed and yaw rate by comparing with GT motion.

    python extract_state.py --mcap ../c0_..._camloc_0.mcap --data data/

The GT comparison is offline analysis only (to pick columns and units).
Position / orientation / s / n fields of this topic are ground truth: never feed them
to the localizer. Only use the speed / acceleration / yaw-rate columns.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

TOPIC = "/constructor0/state_estimation"


def fields(o):
    names = getattr(o, "__slots__", None) or list(getattr(o, "__dict__", {}))
    return [n for n in names if not n.startswith("_")]


def flatten(o, prefix, out):
    if isinstance(o, (bool, int, float, np.integer, np.floating)):
        out[prefix] = float(o)
    elif isinstance(o, (list, tuple, np.ndarray)):
        if 0 < len(o) <= 16 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in o):
            for i, v in enumerate(o):
                out[f"{prefix}[{i}]"] = float(v)
    elif not isinstance(o, (str, bytes)):
        for n in fields(o):
            flatten(getattr(o, n), f"{prefix}.{n}" if prefix else n, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    data = Path(a.data)

    rows = []
    for i, m in enumerate(read_ros2_messages(a.mcap, topics=[TOPIC])):
        r = {}
        flatten(m.ros_msg, "", r)
        hs = r.get("header.stamp.sec")
        r["stamp_ns"] = (int(hs) * 1_000_000_000 + int(r["header.stamp.nanosec"])
                         if hs is not None else m.log_time_ns)
        rows.append(r)
        if i == 0:
            print(f"message type: {type(m.ros_msg).__name__}, {len(r)} numeric fields")
    st = pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True)
    st.to_parquet(data / "state.parquet")
    print(f"saved {len(st)} rows -> {data / 'state.parquet'}")

    # --- which column is speed / yaw rate?  compare with motion derived from GT
    gt = pd.read_parquet(data / "gt_tf.parquet")
    t = gt.stamp_ns.values
    from scipy.spatial.transform import Rotation as R
    yaw = np.unwrap(R.from_quat(gt[["qx", "qy", "qz", "qw"]].values).as_euler("zyx")[:, 0])
    ts = (t - t[0]) * 1e-9
    k = 10  # 0.1 s baseline at 100 Hz, less noise
    v_ref = np.hypot(gt.x.values[k:] - gt.x.values[:-k], gt.y.values[k:] - gt.y.values[:-k]) / (ts[k:] - ts[:-k])
    r_ref = (yaw[k:] - yaw[:-k]) / (ts[k:] - ts[:-k])
    tm = t[:-k] + (t[k:] - t[:-k]) // 2

    cols = [c for c in st.columns if c != "stamp_ns" and st[c].std() > 0]
    res = []
    for c in cols:
        x = np.interp(tm, st.stamp_ns.values, st[c].values)
        for name, ref in (("speed", v_ref), ("yaw_rate", r_ref)):
            cc = np.corrcoef(x, ref)[0, 1]
            scale = float(x @ ref / max(x @ x, 1e-12))
            res.append((name, c, cc, scale))
    res = pd.DataFrame(res, columns=["target", "column", "corr", "scale_to_si"])
    print("\nall numeric fields:\n  " + "\n  ".join(cols))
    for name in ("speed", "yaw_rate"):
        top = res[res.target == name].reindex(
            res[res.target == name]["corr"].abs().sort_values(ascending=False).index).head(5)
        print(f"\nbest matches for {name} (scale_to_si: multiply column by this to get m/s or rad/s):")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()