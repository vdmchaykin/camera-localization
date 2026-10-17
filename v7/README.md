# v7: v6 without the map points on bright blobs

Hypothesis (from the v6 stress test, where the `day` perturbation removed bright blobs and K5 got better): bright
map points are lamps or reflections of the floodlights. A reflection moves with the viewpoint, so its 3D point is
wrong and biases the pose. v7 drops every map point whose 5×5 raw-grey peak in its keyframe is ≥ `V7_PEAK`
(`v6/model/light_<cam>.npz`). Nothing outside `v7/` was changed.

`v7/run_dev.sh` → `v7/runs/` (development laps K1, K5; sync; no drops). K3/K6 not run.

| run | pruned points | K1 pos | K5 pos | K5 lat | K5 hairpin 843–850 s, max \|lat\| | inl (K5) |
|---|---|---|---|---|---|---|
| v6 night | 0 | 0.104 | 0.091 | 0.054 | 0.440 | 980 |
| v7, peak ≥ 250 | 1–7 % per camera | 0.103 | 0.092 | 0.054 | 0.441 | 975 |
| v7, peak ≥ 200 | 4–12 % per camera | 0.103 | 0.092 | 0.055 | 0.444 | 949 |
| v6 under synthetic `day_all` (reference) | — | 0.106 | 0.088 | 0.046 | 0.337 | 756 |

**Hypothesis not confirmed.** Removing the bright points changes nothing (±1 mm), and the inliers barely drop.
Those points were rarely used in the pose. The hairpin error comes from something else, and the gain under the
synthetic day has another cause (contrast / sky / exposure). v7 is neutral at night; it can be kept only as a
hedge for daylight, where these points would not match anyway.
