# v6: preparing for sunlight

The map (`features/`) and all tuning come from one **night** session. The hidden test run "may include higher
speeds and different lighting conditions". There is no daylight data, so v6 (1) adds defences against the
changes sunlight brings, each switchable for ablation, and (2) adds a synthetic stress test to see how the
pipeline reacts. The stress test does not predict day accuracy.

Nothing outside `v6/` was changed. v6 inherits v5 → v3 → v2 → v1.

## What sunlight changes and what v6 does

| what changes by day | risk for the pipeline | v6 answer | flag |
|---|---|---|---|
| images much brighter, different exposure | descriptors see other contrast | per-camera gamma: median raw grey → median of the night keyframes (`model/night_stats.json`); at night gamma ≈ 1 | `V6_EXPO` |
| sun in the image, glare, specular reflections | false keypoints / matches on saturated areas | saturated pixels (≥ 250, dilated) masked for keypoints and dense matches; camera skipped if > 60 % saturated | `V6_SAT` |
| surfaces lit from another direction, lamps off, bright sky | descriptor similarity drops, guided matching fails | on failure, retry once with dense search on and lower thresholds (SIM 0.70→0.60, dense 0.75→0.65), the prior window keeps geometry tight | `V6_RELAX` |
| a wrong match is more likely | a bad pose fix | pose must be on the drivable area (track + pit lane from `/track_bounds`, +1.5 m) | `V6_BOUNDS` |
| matching may fail for seconds | v2 goes global after 3 failures (0.3 s), ~220 ms each, night-trained retrieval may jump to a look-alike straight | global only after 10 failures (1 s), at most once per s, only keyframes within max(30 m, 4σ) of the prior; between attempts keep trying the guided path | `V6_GLOBAL` |

All flags default to on; `V6_xxx=0` turns one off.

How night-dependent the map is (`build_light_table.py`): only 5–12 % of the map points per camera lie on bright
blobs (5×5 peak ≥ 200), the median grey under a point is ~90. The map stands mostly on lit structure (stands,
walls, kerbs, markings), not on the lamps themselves.

## Synthetic sun (`sun_stress.py`)

Wraps any localizer (`STRESS_BASE=path:Class`) and perturbs the images before `step()`:
`day` (brighter, lamps removed, bright sky behind dark silhouettes), `shadow` (hard cast shadows on the ground,
moving every 2 s), `glare` (sun at a fixed map azimuth/elevation: saturated disk + veiling glare in cameras that
look towards it, depending on heading), `relight` (smooth random gain field = light from another direction, the
harshest proxy), `overexp`. The perturbation time is reported as `stress_ms` and must be subtracted from `step_ms`.
`preview_sun.py` → `sun_preview.jpg`.

## Files

- `my_loc_v6.py` — the localizer.
- `build_light_table.py` — training-time artefacts in `model/`: `night_stats.json`, `light_<cam>.npz`.
- `sun_stress.py`, `preview_sun.py` — stress test and its preview.
- `run_stress.sh` — stress matrix (v3 / v5 / v6 × night / day_all / hard) on dev laps K1, K5.
- `runs/` — results.

## What cannot be prepared without day data

- Real night→day descriptor change. SuperPoint descriptors are trained on mixed lighting, but the map holds
  night descriptors only. With a day run you would add a second map layer (same 3D points, day descriptors), or
  re-describe the map points with a lighting-invariant descriptor.
- Lighting-independent cues, e.g. track borders and kerbs detected in the images, matched to `/track_bounds`
  (lateral + yaw). They would also remove the dependency on the night map.
