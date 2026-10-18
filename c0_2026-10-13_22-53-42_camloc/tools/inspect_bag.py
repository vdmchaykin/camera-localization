#!/usr/bin/env python3
"""
Quick look at a new bag before running the localizer on it.

    python inspect_bag.py --mcap ../rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap \
        --train-mcap ../c0_2026-10-13_22-53-42_camloc_0.mcap --out bag_check/

Prints: time span, camera rates and stamp sync, the motion topics (frame_id, field ranges),
whether /tf or a first pose exists, and saves one contact sheet of the 7 cameras (to see
day / night) plus the twist/accel as CSV.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages


def stamp(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--train-mcap", default=None, help="to check if it has the same motion topics")
    ap.add_argument("--out", default="bag_check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    with open(a.mcap, "rb") as f:
        topics = sorted({c.topic for c in make_reader(f).get_summary().channels.values()})
    print("topics:\n  " + "\n  ".join(topics))
    print(f"/tf present: {'/tf' in topics} | state_estimation present: "
          f"{any('state_estimation' in t for t in topics)} | track_bounds present: "
          f"{any('track_bounds' in t for t in topics)}")

    cams = defaultdict(list); first_img = {}
    twist, accel = [], []
    for m in read_ros2_messages(a.mcap):
        t, msg = m.channel.topic, m.ros_msg
        if t.endswith("/compressed_image"):
            cam = t.split("camera_")[1].split("/")[0]
            cams[cam].append(stamp(msg.header))
            if cam not in first_img and len(cams[cam]) == 50:      # 5 s in
                first_img[cam] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), 1)
        elif t.endswith("validation/twist"):
            tw = msg.twist
            twist.append(dict(stamp_ns=stamp(msg.header), frame=msg.header.frame_id,
                              vx=tw.linear.x, vy=tw.linear.y, vz=tw.linear.z,
                              wx=tw.angular.x, wy=tw.angular.y, wz=tw.angular.z))
        elif t.endswith("validation/accel"):
            ac = msg.accel
            accel.append(dict(stamp_ns=stamp(msg.header), frame=msg.header.frame_id,
                              ax=ac.linear.x, ay=ac.linear.y, az=ac.linear.z))

    t0 = min(min(v) for v in cams.values())
    print(f"\ncameras ({len(cams)}):")
    for c, s in sorted(cams.items()):
        s = np.array(s)
        print(f"  {c:3s} {len(s)} frames, {(s[-1]-s[0])/1e9:.1f} s, median step {np.median(np.diff(s))/1e6:.1f} ms")
    # sync: spread of stamps inside one 100 ms set
    sets = defaultdict(list)
    for c, s in cams.items():
        for x in s:
            sets[round((x - t0) / 1e8)].append(x)
    spread = [(max(v) - min(v)) / 1e6 for v in sets.values() if len(v) > 1]
    print(f"  stamp spread inside a set: median {np.median(spread):.3f} ms, max {np.max(spread):.3f} ms")

    for name, rows in (("twist", twist), ("accel", accel)):
        if not rows:
            print(f"\n{name}: none"); continue
        d = pd.DataFrame(rows)
        d.to_csv(out / f"{name}.csv", index=False)
        dt = np.median(np.diff(d.stamp_ns)) / 1e6
        print(f"\n{name}: {len(d)} msgs, ~{1000/dt:.0f} Hz, frame_id {d.frame.unique().tolist()}")
        print(d.drop(columns=["stamp_ns", "frame"]).describe().loc[["mean", "min", "max"]].round(3).to_string())

    if first_img:
        tiles = [cv2.resize(first_img[c], (500, 242)) for c in sorted(first_img)]
        for c, im in zip(sorted(first_img), tiles):
            cv2.putText(im, c, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / "cameras.jpg"), sheet)
        print(f"\nsaved {out / 'cameras.jpg'} (day or night?)")
        print("mean brightness per camera:", {c: int(first_img[c].mean()) for c in sorted(first_img)})

    if a.train_mcap:
        with open(a.train_mcap, "rb") as f:
            tt = sorted({c.topic for c in make_reader(f).get_summary().channels.values()})
        print("\ntraining mcap has validation/twist:", any("validation/twist" in t for t in tt))


if __name__ == "__main__":
    main()