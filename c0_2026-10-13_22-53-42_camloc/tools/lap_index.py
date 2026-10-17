#!/usr/bin/env python3
"""
One-time indexing of the training mcap, split by lap.

    python lap_index.py extract --mcap train.mcap --laps laps.csv --out data/
    python lap_index.py info --out data/

Produces (under --out):
    images/<cam>/<stamp_ns>.jpg   original JPEG bytes, no re-encoding
    frames.parquet                one row per image: stamp_ns, cam, path, lap,
                                  GT pose (6-DoF) interpolated at the image's own stamp
    gt_tf.parquet                 raw map->base_link poses (100 Hz)
    calib.json                    K, D, size per camera + raw /tf_static edges
    laps.csv                      copy of the lap table

Lap selection afterwards is just a filter:
    from lap_index import load_frames
    map_frames = load_frames("data", laps=[1, 2, 3, 5, 6], cams=["fl", "fr", "r"])

GT poses here are for map building and validation only - never at inference.
"""
import argparse, json, re, shutil
from pathlib import Path

import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial.transform import Rotation, Slerp

CAM_IMG = re.compile(r"/camera_([^/]+)/compressed_image$")
CAM_INFO = re.compile(r"/camera_([^/]+)/camera_info$")
MAX_GAP_NS = 50_000_000  # same rule as the scorer: skip if GT gap > 50 ms


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def list_topics(mcap_path):
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    return sorted({ch.topic for ch in summary.channels.values()})


def extract(args):
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cams = set(args.cams) if args.cams else None

    topics = list_topics(args.mcap)
    img_topics = [t for t in topics if CAM_IMG.search(t)
                  and (cams is None or CAM_IMG.search(t).group(1) in cams)]
    info_topics = [t for t in topics if CAM_INFO.search(t)]
    wanted = img_topics + info_topics + ["/tf", "/tf_static"]
    print(f"cameras: {[CAM_IMG.search(t).group(1) for t in img_topics]}")

    img_rows, tf_rows = [], []
    calib = {"cameras": {}, "tf_static": {}}

    for i, m in enumerate(read_ros2_messages(args.mcap, topics=wanted)):
        topic, msg = m.channel.topic, m.ros_msg

        if topic == "/tf":
            for t in msg.transforms:
                if t.child_frame_id.endswith(args.base_frame):
                    tr, q = t.transform.translation, t.transform.rotation
                    tf_rows.append((stamp_ns(t.header), tr.x, tr.y, tr.z,
                                    q.x, q.y, q.z, q.w))
        elif topic == "/tf_static":
            for t in msg.transforms:  # parent -> child edges, keep latest
                tr, q = t.transform.translation, t.transform.rotation
                calib["tf_static"][t.child_frame_id] = {
                    "parent": t.header.frame_id,
                    "t": [tr.x, tr.y, tr.z], "q_xyzw": [q.x, q.y, q.z, q.w]}
        elif (mi := CAM_INFO.search(topic)):
            cam = mi.group(1)
            if cam not in calib["cameras"]:
                calib["cameras"][cam] = {
                    "frame_id": msg.header.frame_id,
                    "width": msg.width, "height": msg.height,
                    "K": list(msg.k), "D": list(msg.d),
                    "distortion_model": msg.distortion_model}
        else:
            cam = CAM_IMG.search(topic).group(1)
            ns = stamp_ns(msg.header)
            rel = Path("images") / cam / f"{ns}.jpg"
            (out / rel).parent.mkdir(exist_ok=True)
            (out / rel).write_bytes(bytes(msg.data))
            img_rows.append((ns, cam, str(rel)))

        if i % 20000 == 0:
            print(f"  {i} msgs, {len(img_rows)} images, {len(tf_rows)} tf")

    gt = pd.DataFrame(tf_rows, columns=["stamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"])
    gt = gt.drop_duplicates("stamp_ns").sort_values("stamp_ns").reset_index(drop=True)
    gt.to_parquet(out / "gt_tf.parquet")

    frames = pd.DataFrame(img_rows, columns=["stamp_ns", "cam", "path"])
    frames = frames.sort_values(["stamp_ns", "cam"]).reset_index(drop=True)
    frames = attach_gt(frames, gt)

    laps = pd.read_csv(args.laps)
    t0 = args.t0_ns if args.t0_ns is not None else int(gt.stamp_ns.iloc[0])
    frames["t_rel"] = (frames.stamp_ns - t0) * 1e-9
    idx = np.searchsorted(laps.t_start.values, frames.t_rel.values, side="right") - 1
    inside = (idx >= 0) & (frames.t_rel.values < laps.t_end.values[idx.clip(0)])
    frames["lap"] = np.where(inside, laps.lap.values[idx.clip(0)], -1)

    frames.to_parquet(out / "frames.parquet")
    (out / "calib.json").write_text(json.dumps(calib, indent=2))
    shutil.copy(args.laps, out / "laps.csv")
    print(f"done: {len(frames)} images, t0={t0}")
    info(args)


def attach_gt(frames, gt):
    """Interpolate the 6-DoF GT pose at each image's own exposure stamp."""
    ts = gt.stamp_ns.values
    q = frames.stamp_ns.values
    j = np.searchsorted(ts, q).clip(1, len(ts) - 1)
    t_lo, t_hi = ts[j - 1], ts[j]
    ok = (q >= ts[0]) & (q <= ts[-1]) & (t_hi - t_lo <= MAX_GAP_NS)

    a = np.where(ok, (q - t_lo) / np.maximum(t_hi - t_lo, 1), 0.0)
    for c in ["x", "y", "z"]:
        v = gt[c].values
        frames[c] = np.where(ok, v[j - 1] + a * (v[j] - v[j - 1]), np.nan)

    rot = Rotation.from_quat(gt[["qx", "qy", "qz", "qw"]].values)
    slerp = Slerp(ts.astype(np.float64), rot)
    quat = np.full((len(q), 4), np.nan)
    quat[ok] = slerp(q[ok].astype(np.float64)).as_quat()
    frames[["qx", "qy", "qz", "qw"]] = quat
    frames["yaw"] = np.nan
    frames.loc[ok, "yaw"] = Rotation.from_quat(quat[ok]).as_euler("zyx")[:, 0]
    frames["gt_ok"] = ok
    return frames


def load_frames(root, laps=None, cams=None, require_gt=True):
    """Select frames by lap and camera. laps=None -> all laps (excluding -1)."""
    f = pd.read_parquet(Path(root) / "frames.parquet")
    f = f[f.lap >= 0]
    if laps is not None:
        f = f[f.lap.isin(laps)]
    if cams is not None:
        f = f[f.cam.isin(cams)]
    if require_gt:
        f = f[f.gt_ok]
    return f.reset_index(drop=True)


def info(args):
    f = pd.read_parquet(Path(args.out) / "frames.parquet")
    print(f.groupby(["lap", "cam"]).size().unstack(fill_value=0))
    print(f"no GT: {(~f.gt_ok).sum()} images, outside laps: {(f.lap < 0).sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--mcap", required=True)
    e.add_argument("--laps", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--cams", nargs="*", help="e.g. fl fr r (default: all 7)")
    e.add_argument("--base-frame", default="base_link")
    e.add_argument("--t0-ns", type=int, default=None,
                   help="absolute ns that laps.csv t=0 refers to (default: first /tf stamp)")
    i = sub.add_parser("info")
    i.add_argument("--out", required=True)
    a = p.parse_args()
    extract(a) if a.cmd == "extract" else info(a)