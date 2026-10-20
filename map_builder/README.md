# Map builder: synthetic day data, night map and day descriptor layer

All the code that builds the model (`model/`) from the data: the night keyframe map (keyframes, SuperPoint keypoints,
descriptors, 3D points, DINOv2 global descriptors, car-body masks), the keyframe poses / motion calibration / night
brightness / track borders, and the day descriptor of every map point.

```bash
pip install pandas                                         # in addition to ../requirements.txt
map_builder/build_map.sh TRAIN.mcap DAY.mcap               # ~3 h on one GPU, writes model/
```

* `TRAIN.mcap`: the training run `c0_2026-10-13_22-53-42_camloc_0.mcap` (night, with GT `/tf`).
* `DAY.mcap`: a day-styled copy of the training run: the same bag with every camera image re-lit night -> day by
  Qwen-Image-Edit-2511 + the Lightning 4-step LoRA (geometry, header stamps and all other messages unchanged).
  Made by `qwen_day.py` (own environment, see its docstring); several parts may be given
  (`DAY_part1.mcap DAY_part2.mcap`).

```bash
python map_builder/qwen_day.py TRAIN.mcap DAY.mcap         # step 0: synthetic day copy (GPU, resumable)
```

## Steps

| step | script | output (in `model/`) |
|---|---|---|
| 1 | `export_lite.py TRAIN.mcap export` | (work) calibration, GT `/tf`, image stamps |
| 2 | `extract_frames.py TRAIN.mcap frames` | (work) every frame of the 7 cameras at half resolution (753x364) |
| 3 | `make_masks.py` | `masks/<cam>.png`: hand-drawn car-body polygons |
| 4 | `global_desc.py` | `features/global_<cam>.npy`: DINOv2 ViT-S/14 descriptor ([CLS ; GeM]) of every frame |
| 5 | `db_build.py` | `features/db_<cam>.npz`: the night keyframe map (below) |
| 6 | `write_set_stamps.py` | `features/set_stamps.npy`: row order of `global_<cam>.npy` |
| 7 | `build_model.py --mcap TRAIN.mcap` | `kf_poses.npz`, `motion_calib.json`, `night_stats.json`, `track_bounds.npz` |
| 8 | `build_day_layer.py --mcap DAY.mcap` | `day/day_<cam>.npy`, `day/day_meta.json`: the day descriptor layer |

Steps 1-6 are the scripts that built the night map on 2026-10-16, kept unchanged in their original standalone
layout (own `camera_model.py`, `vislib.py`; work files `export/`, `frames/`, `features/`, `masks/` next to them);
`write_set_stamps.py` is a packaging helper added later. Steps 7-8 use the localizer package `camloc/`, so the day
descriptors go through exactly the query path of the localizer.

**Night map** (`db_build.py`): a keyframe every 4 m of GT travel (5956 keyframes); per camera, SuperPoint (1024 kp,
CLAHE, body mask) on each keyframe, LightGlue matches with the next keyframe, triangulation with the GT poses and the
`/tf_static` extrinsics (reprojection < 1.5 px, parallax > 0.75 deg, depth < 150 m); each keypoint keeps its
largest-parallax 3D point, up to 500 points with the highest score per keyframe. `db_<cam>.npz` holds `kf_stamp`,
`off` (points of keyframe k: `off[k]:off[k+1]`), `kps`, `desc` (float16), `xyz` (map frame), `par`. GT is used only
here and in step 7, at training time.

**Day layer** (`build_day_layer.py`): for every map keyframe and camera, the image with the same header stamp in the
day bag goes through the localizer's query path (grey, exposure normalisation to the night median, CLAHE, FP16
SuperPoint backbone, L2-normalised dense descriptors) and the descriptor is sampled at the map keypoints. So every map
point keeps its 3D position and gets a second, day descriptor (`day_<cam>.npy`, aligned row by row with
`db_<cam>.npz['desc']`); the localizer matches with the better of the two.

The validation model (K3 / K6 held out of the motion calibration and the day layer) and all validation runs:
`tools/reproduce.sh`, see the main [README](../README.md#building-the-model).

## Check against the submission model

Rebuilt on 2026-10-20 with the code in this folder and compared with the submission model:

* night map (steps 1-6, frames re-extracted from the mcap): the first 109 keyframes of all 7 cameras give the same
  keyframes, keypoints, descriptors and 3D points, byte for byte; the masks are the same files; DINOv2 descriptors of
  64 sampled frames are identical. The full night map was not rebuilt (~3 h);
* steps 7-8 in full (`build_model.py`, `build_day_layer.py` on `qwen_full.mcap`, ~4 min): `kf_poses.npz`,
  `motion_calib.json`, `night_stats.json`, `track_bounds.npz` and all seven `day/day_<cam>.npy` are byte-identical.

Results may differ in the last bits on another GPU / library version.
