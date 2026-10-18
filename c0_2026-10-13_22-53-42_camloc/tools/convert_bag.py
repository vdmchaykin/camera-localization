#!/usr/bin/env python3
"""
Turn a new bag (cameras + validation/twist, no /tf, no state_estimation) into a data folder
that the localizers read like the training data (no GT inside).

    python convert_bag.py --mcap ../rosbag_.../rosbag_..._0.mcap --out data_bag --train-data data

Writes data_bag/:
  images/<cam>/<stamp_ns>.jpg     original JPEG bytes
  calib.json                      camera_info + /tf_static of THIS bag
  state.parquet                   vx_mps, vy_mps, wz_radps from /constructor0/validation/twist
                                  (frame constructor0/base_link -> already at base_link)
  motion_calib.json               identity: the twist is at base_link, no lever-arm correction
  frames.parquet, laps.csv        one "lap" (1) covering the whole bag; pose columns are NaN
                                  except the first frame set, filled later by init_pose.py
  track_bounds.json, masks/       copied from the training data folder (same track, same car)
"""
import argparse, json, re, shutil
from pathlib import Path

import numpy as np
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

IMG = re.compile(r"/camera_([^/]+)/compressed_image$")
INFO = re.compile(r"/camera_([^/]+)/camera_info$")


def stamp(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-data", default="data", help="for track_bounds.json and masks/")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    calib = {"cameras": {}, "tf_static": {}}
    rows, tw = [], []
    for m in read_ros2_messages(a.mcap):
        t, msg = m.channel.topic, m.ros_msg
        if (mi := IMG.search(t)):
            cam, ns = mi.group(1), stamp(msg.header)
            rel = Path("images") / cam / f"{ns}.jpg"
            (out / rel).parent.mkdir(exist_ok=True)
            (out / rel).write_bytes(bytes(msg.data))
            rows.append((ns, cam, str(rel)))
        elif (mi := INFO.search(t)):
            cam = mi.group(1)
            calib["cameras"].setdefault(cam, {
                "frame_id": msg.header.frame_id, "width": msg.width, "height": msg.height,
                "K": list(msg.k), "D": list(msg.d), "distortion_model": msg.distortion_model})
        elif t == "/tf_static":
            for e in msg.transforms:
                tr, q = e.transform.translation, e.transform.rotation
                calib["tf_static"][e.child_frame_id] = {
                    "parent": e.header.frame_id, "t": [tr.x, tr.y, tr.z],
                    "q_xyzw": [q.x, q.y, q.z, q.w]}
        elif t.endswith("validation/twist"):
            v, w = msg.twist.linear, msg.twist.angular
            tw.append((stamp(msg.header), v.x, v.y, w.z, float(np.hypot(v.x, v.y))))

    (out / "calib.json").write_text(json.dumps(calib, indent=2))
    st = pd.DataFrame(tw, columns=["stamp_ns", "vx_mps", "vy_mps", "wz_radps", "v_mps"])
    st.sort_values("stamp_ns").to_parquet(out / "state.parquet")
    (out / "motion_calib.json").write_text(json.dumps(
        {"vx_k": 1.0, "vy_a": 1.0, "vy_b": 0.0, "vy_c": 0.0, "wz_s": 1.0, "wz_d": 0.0,
         "note": "identity: validation/twist is given in base_link"}, indent=2))

    f = pd.DataFrame(rows, columns=["stamp_ns", "cam", "path"]).sort_values(["stamp_ns", "cam"])
    t0 = int(f.stamp_ns.min())
    f["t_rel"] = (f.stamp_ns - t0) * 1e-9
    f["lap"] = 1
    for c in ["x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]:
        f[c] = np.nan                       # no GT in this bag; init_pose.py fills the first set
    f["gt_ok"] = True
    f.reset_index(drop=True).to_parquet(out / "frames.parquet")
    dur = float(f.t_rel.max()) + 1.0
    pd.DataFrame({"lap": [1], "t_start": [0.0], "t_end": [dur], "duration_s": [dur],
                  "complete": [True]}).to_csv(out / "laps.csv", index=False)

    td = Path(a.train_data)
    if (td / "track_bounds.json").exists():
        shutil.copy(td / "track_bounds.json", out / "track_bounds.json")
    if (td / "masks").exists() and not (out / "masks").exists():
        shutil.copytree(td / "masks", out / "masks")

    # sanity: same cameras / extrinsics as the training data?
    tc = td / "calib.json"
    if tc.exists():
        old = json.loads(tc.read_text())
        for cam, c in calib["cameras"].items():
            o = old["cameras"].get(cam)
            if o is None:
                print(f"  {cam}: not in the training calibration!")
                continue
            dK = np.abs(np.array(c["K"]) - np.array(o["K"])).max()
            print(f"  {cam}: max |dK| vs training {dK:.3f}")
        for k, e in calib["tf_static"].items():
            o = old["tf_static"].get(k)
            if o:
                dt = np.abs(np.array(e["t"]) - np.array(o["t"])).max()
                dq = np.abs(np.array(e["q_xyzw"]) - np.array(o["q_xyzw"])).max()
                if dt > 1e-4 or dq > 1e-4:
                    print(f"  tf_static {k}: differs from training (dt {dt:.4f} m, dq {dq:.5f})")
    print(f"{len(f)} images, {f.cam.nunique()} cameras, {dur:.1f} s, twist {len(st)} msgs "
          f"(speed {st.v_mps.min():.1f}..{st.v_mps.max():.1f} m/s) -> {out}")
    print("next: init_pose.py (first pose), then the localizer with --data", out)


if __name__ == "__main__":
    main()