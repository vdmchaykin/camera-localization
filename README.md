# camera-localization
## cam_localization

The final localizer (keyframe map + day descriptor layer + EKF, one run script) is in
[cam_localization/](cam_localization/README.md): `./setup.sh`, `./download_model.sh`, `./run.sh RUN.mcap OUT_DIR`.

`c0_2026-10-13_22-53-42_camloc/` is the organisers' dataset folder without the data: their README and their scoring
tools `tools/extract_gt.py` and `tools/evaluate.py`, unchanged. `cam_localization/tools/run.py` scores with this
`evaluate.py`.
