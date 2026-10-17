#!/usr/bin/env python3
"""
Method 3a with COLMAP/hloc: triangulate a 3D feature map from training laps using GT poses.

    python colmap_map.py build --data data/ --laps 1 2 3 5 6 --cams fl fr r --out maps/colmap_v1
    python colmap_map.py build ... --cams fr --laps 1 2 --limit 300 --out maps/colmap_dbg   # debug

Steps (each is resumable; delete the output file to redo a step):
  1. keyframes per camera from frames.parquet (every --kf-dist m / --kf-yaw deg)
  2. reference COLMAP model in ref/: cameras (FULL_OPENCV from camera_info) + images with
     FIXED poses cam_from_world = inv(T_map_base_GT @ T_base_cam); no 3D points yet
  3. image pairs from poses: close camera centres + similar viewing direction,
     across all laps and all cameras (fl and fr overlap, so they get paired too)
  4. SuperPoint features + LightGlue matches (hloc)
  5. hloc triangulation: COLMAP point_triangulator + bundle adjustment of points only,
     poses stay fixed -> the map lives exactly in the track frame
  6. export to the same per-camera format as featmap.py, so localization/validation is
         python featmap.py validate --data data/ --map maps/colmap_v1 --laps 4 7 --out val.csv
"""
import argparse, json
from copy import deepcopy
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from hloc import extract_features, match_features, triangulation
import pycolmap

from lap_index import load_frames
from featmap import T_base_cam, T_map_base_row, select_keyframes


# ----------------------------------------------------------------------------- 1-2
def write_reference(kfs_by_cam, calib, ref_dir):
    """COLMAP text model with GT camera poses and no points."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    cam_lines, img_lines, rows = [], [], []
    for cid, (cam, kfs) in enumerate(kfs_by_cam.items(), start=1):
        c = calib["cameras"][cam]
        K = np.array(c["K"], dtype=float).reshape(3, 3)
        k1, k2, p1, p2, k3 = (list(c["D"]) + [0.0] * 5)[:5]
        # ROS/OpenCV pixel centres are at integers, COLMAP's at +0.5 (hloc shifts keypoints too)
        cam_lines.append(f"{cid} FULL_OPENCV {c['width']} {c['height']} "
                         f"{K[0, 0]} {K[1, 1]} {K[0, 2] + 0.5} {K[1, 2] + 0.5} "
                         f"{k1} {k2} {p1} {p2} {k3} 0 0 0")
        T_bc = T_base_cam(calib, cam)
        for r in kfs.itertuples():
            T_mc = T_map_base_row(r) @ T_bc
            T_cm = np.linalg.inv(T_mc)
            qx, qy, qz, qw = R.from_matrix(T_cm[:3, :3]).as_quat()
            iid = len(rows) + 1
            name = Path(r.path).relative_to("images").as_posix()      # e.g. fr/1791...jpg
            img_lines.append(f"{iid} {qw} {qx} {qy} {qz} {T_cm[0, 3]} {T_cm[1, 3]} {T_cm[2, 3]} "
                             f"{cid} {name}\n\n")
            rows.append(dict(image_id=iid, name=name, cam=cam, stamp_ns=r.stamp_ns, lap=r.lap,
                             x=r.x, y=r.y, z=r.z, yaw=r.yaw,
                             cx=T_mc[0, 3], cy=T_mc[1, 3], cz=T_mc[2, 3],
                             dx=T_mc[0, 2], dy=T_mc[1, 2], dz=T_mc[2, 2]))
    (ref_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (ref_dir / "images.txt").write_text("".join(img_lines))
    (ref_dir / "points3D.txt").write_text("")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3
def pairs_from_poses(index, out_path, num, max_dist, max_angle_deg):
    C = index[["cx", "cy", "cz"]].values
    V = index[["dx", "dy", "dz"]].values
    tree = cKDTree(C)
    cos_thr = np.cos(np.deg2rad(max_angle_deg))
    pairs = set()
    for i, nb in enumerate(tree.query_ball_point(C, max_dist)):
        nb = [j for j in nb if j != i and V[i] @ V[j] >= cos_thr]
        nb.sort(key=lambda j: np.sum((C[j] - C[i]) ** 2))
        pairs.update((min(i, j), max(i, j)) for j in nb[:num])
    names = index.name.values
    out_path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in sorted(pairs)))
    return len(pairs)


# ----------------------------------------------------------------------------- 4b
def filter_features(features, names, data, mask_dir, drop_ids, use_body, dilate_px):
    """Drop keypoints on unwanted SAM 3 classes (e.g. car, sky, floodlight) and on the own
    car body, in place, BEFORE matching. Marked per image so it is never applied twice."""
    data = Path(data)
    body = {}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2) if dilate_px else None
    n_before = n_after = n_nomask = 0
    with h5py.File(features, "r+") as h:
        for name in names:
            g = h[name]
            if g.attrs.get("masked", False):
                continue
            cam = name.split("/")[0]
            H, W = [int(v) for v in g["image_size"][()][::-1]]
            drop = np.zeros((H, W), bool)
            if use_body:
                if cam not in body:
                    bm = cv2.imread(str(data / "masks" / f"{cam}.png"), cv2.IMREAD_GRAYSCALE)
                    body[cam] = (bm == 0) if bm is not None else np.zeros((H, W), bool)
                drop |= body[cam]
            if mask_dir:
                lab = cv2.imread(str(data / mask_dir / name.replace(".jpg", ".png")),
                                 cv2.IMREAD_GRAYSCALE)
                if lab is None:
                    n_nomask += 1
                else:
                    m = np.isin(lab, drop_ids).astype(np.uint8)
                    if k is not None:
                        m = cv2.dilate(m, k)
                    drop |= m > 0
            kp = g["keypoints"][()]
            u = np.clip(np.round(kp[:, 0]).astype(int), 0, W - 1)
            v = np.clip(np.round(kp[:, 1]).astype(int), 0, H - 1)
            keep = ~drop[v, u]
            n_before += len(kp); n_after += int(keep.sum())
            for key in ("keypoints", "scores", "descriptors"):
                if key not in g:
                    continue
                arr = g[key][()]
                arr = arr[:, keep] if key == "descriptors" else arr[keep]   # desc is (D, N)
                del g[key]
                g.create_dataset(key, data=arr)
            g.attrs["masked"] = True
    if n_before:
        print(f"masks: kept {n_after}/{n_before} keypoints ({n_after / n_before:.0%}), "
              f"{n_nomask} images had no SAM mask (body mask only)")


# ----------------------------------------------------------------------------- 6
def export(out, index, features, rec, a):
    good = {pid: p.xyz for pid, p in rec.points3D.items()
            if p.error < a.max_point_err and p.track.length() >= a.min_track}
    name2id = {img.name: iid for iid, img in rec.images.items()}
    print(f"export: {len(good)} / {len(rec.points3D)} points pass "
          f"(error < {a.max_point_err}px, track >= {a.min_track})")

    with h5py.File(features, "r") as h:
        for cam in a.cams:
            sub = index[index.cam == cam].reset_index(drop=True)
            st = {"kp": [], "desc": [], "xyz": [], "score": [], "kf": []}
            n_skip = 0
            for k, r in enumerate(sub.itertuples()):
                iid = name2id.get(r.name)
                if iid is None:
                    continue
                g = h[r.name]
                kp = g["keypoints"][()].astype(np.float32)          # OpenCV pixel convention
                desc = g["descriptors"][()].T.astype(np.float16)    # hloc stores (D, N)
                sc = g["scores"][()].astype(np.float32)
                p2d = rec.images[iid].points2D
                if len(p2d) != len(kp):
                    # image without any verified match: COLMAP keeps it with no keypoints
                    n_skip += 1
                    continue
                idx = [i for i, p in enumerate(p2d) if p.has_point3D() and p.point3D_id in good]
                if not idx:
                    continue
                idx = np.array(sorted(idx, key=lambda i: -sc[i])[:a.max_pts_per_kf])
                st["kp"].append(kp[idx]); st["desc"].append(desc[idx]); st["score"].append(sc[idx])
                st["xyz"].append(np.array([good[p2d[i].point3D_id] for i in idx], np.float32))
                st["kf"].append(np.full(len(idx), k))
            if not st["kf"]:
                print(f"[{cam}] no points!")
                continue
            if n_skip:
                print(f"[{cam}] skipped {n_skip} keyframes without keypoints in the model")
            kf = np.concatenate(st["kf"])
            np.savez(out / f"{cam}.npz", kp=np.concatenate(st["kp"]),
                     desc=np.concatenate(st["desc"]), xyz=np.concatenate(st["xyz"]),
                     score=np.concatenate(st["score"]), kf=kf)
            kt = sub[["stamp_ns", "lap", "x", "y", "z", "yaw", "cx", "cy"]].copy()
            kt["cam_heading"] = np.arctan2(sub.dy, sub.dx)
            kt["n_pts"] = np.bincount(kf, minlength=len(sub))
            kt.to_parquet(out / f"{cam}_kf.parquet")
            print(f"[{cam}] {len(kf)} points, {(kt.n_pts > 0).mean():.0%} keyframes with points, "
                  f"median {int(kt.n_pts.median())} per keyframe")

    (out / "meta.json").write_text(json.dumps(
        {"source": "colmap", "laps": a.laps, "cams": a.cams}, indent=2))


def plot_points(out, data, rec):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xyz = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    if len(xyz) == 0:
        print("no 3D points, skipping plot")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    tb = Path(data) / "track_bounds.json"
    if tb.exists():
        for _, pts in json.loads(tb.read_text()):
            p = np.asarray(pts)
            ax.plot(p[:, 0], p[:, 1], color="k", lw=0.5)
    ax.scatter(xyz[:, 0], xyz[:, 1], s=0.2, c=xyz[:, 2], cmap="viridis", vmin=-2, vmax=15)
    ax.set_aspect("equal")
    fig.savefig(out / "map_points.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'map_points.png'}")


# ----------------------------------------------------------------------------- main
def build(a):
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = json.loads((data / "calib.json").read_text())

    kfs = {}
    for cam in a.cams:
        f = load_frames(data, laps=a.laps, cams=[cam])
        k = select_keyframes(f, a.kf_dist, a.kf_yaw)
        kfs[cam] = k.iloc[:a.limit] if a.limit else k
        print(f"[{cam}] {len(f)} frames -> {len(kfs[cam])} keyframes")

    ref = out / "ref"
    index = write_reference(kfs, calib, ref)
    index.to_parquet(out / "index.parquet")

    pairs = out / "pairs.txt"
    if not pairs.exists():
        n = pairs_from_poses(index, pairs, a.num_pairs, a.pair_dist, a.pair_angle)
        print(f"{n} pairs for {len(index)} images")

    features, matches = out / "features.h5", out / "matches.h5"
    fconf = deepcopy(extract_features.confs["superpoint_aachen"])
    fconf["model"]["max_keypoints"] = a.max_kp
    extract_features.main(fconf, data / "images", image_list=list(index.name),
                          feature_path=features)
    if a.mask_dir or a.body_masks:
        if matches.exists():
            with h5py.File(features, "r") as h:
                if not all(h[n].attrs.get("masked", False) for n in index.name):
                    raise SystemExit("matches.h5 was computed on unmasked features: "
                                     "use a new --out folder when switching masks on")
        drop_ids = []
        if a.mask_dir:
            cls = json.loads((data / a.mask_dir / "classes.json").read_text())
            drop_ids = [c["id"] for c in cls if c["name"] in a.drop_classes]
        print(f"masking keypoints: classes {a.drop_classes} from data/{a.mask_dir}, "
              f"body masks {'on' if a.body_masks else 'off'}")
        filter_features(features, list(index.name), data, a.mask_dir, drop_ids,
                        a.body_masks, a.mask_dilate)
    match_features.main(match_features.confs["superpoint+lightglue"], pairs,
                        features=features, matches=matches)

    sfm = out / "sfm"
    if (sfm / "points3D.bin").exists():
        rec = pycolmap.Reconstruction(sfm)
    else:
        rec = triangulation.main(sfm_dir=sfm, reference_model=ref, image_dir=data / "images",
                                 pairs=pairs, features=features, matches=matches)
    print(rec.summary())

    export(out, index, features, rec, a)
    plot_points(out, data, rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--laps", nargs="+", type=int, required=True)
    b.add_argument("--cams", nargs="+", default=["fl", "fr", "r"])
    b.add_argument("--out", required=True)
    b.add_argument("--kf-dist", type=float, default=4.0, help="m between keyframes")
    b.add_argument("--kf-yaw", type=float, default=5.0, help="deg between keyframes")
    b.add_argument("--limit", type=int, default=0, help="debug: first N keyframes per camera")
    b.add_argument("--max-kp", type=int, default=2048)
    b.add_argument("--num-pairs", type=int, default=10, help="neighbours per image")
    b.add_argument("--pair-dist", type=float, default=25.0, help="m between camera centres")
    b.add_argument("--pair-angle", type=float, default=35.0, help="deg between viewing dirs")
    b.add_argument("--max-point-err", type=float, default=2.0, help="px, export filter")
    b.add_argument("--min-track", type=int, default=3, help="views per point, export filter")
    b.add_argument("--max-pts-per-kf", type=int, default=600)
    b.add_argument("--mask-dir", default=None, help="SAM 3 masks under data/: sam3 or sam3_video")
    b.add_argument("--drop-classes", nargs="+", default=["car", "sky", "light"])
    b.add_argument("--mask-dilate", type=int, default=3, help="px margin around dropped classes")
    b.add_argument("--body-masks", action="store_true", help="also use data/masks/<cam>.png")
    build(ap.parse_args())