#!/bin/bash
# Same comparison as on the local PC (v2/run_dev.sh + v2/run_val.sh + v1 on K3/K6), for the cluster GPU.
# Results: v2/runs_cluster/<tag>. Run from the project root:  nohup v2/run_cluster.sh > v2/runs_cluster/run.log 2>&1 &
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home"
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v2/runs_cluster
run() { python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $1 --speed 1 --start-s $2 --max-s $3 --no-gui --tag $4 --out v2/runs_cluster/$4 2>&1 | grep -v Warning; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
run sim/my_loc.py:Localizer   185.5 180.1 v1_K1
run v2/my_loc_v2.py:Localizer 185.5 180.1 v2_K1
run sim/my_loc.py:Localizer   752.2 143.9 v1_K5
run v2/my_loc_v2.py:Localizer 752.2 143.9 v2_K5
run sim/my_loc.py:Localizer   523.4 110.6 v1_val_K3
run v2/my_loc_v2.py:Localizer 523.4 110.6 v2_val_K3
run sim/my_loc.py:Localizer   896.1 111.6 v1_val_K6
run v2/my_loc_v2.py:Localizer 896.1 111.6 v2_val_K6
python3 v2/profile_v2.py --n 40 --start-s 680 2>&1 | grep -v Warning | tail -6 > v2/runs_cluster/profile_v2.txt
echo ALL_DONE
