#!/bin/bash -l
# =============================================================================
# slurm/shower_grid_array.sh  --  FULL-STATS FSD Cube shower multiplicity grid,
# run as a Slurm JOB ARRAY over disjoint shower slices so every task finishes
# inside a short wall (and they run concurrently), then merged with merge_grids.sh.
#
# Each array task k inducts a NSLICE-shower slice of the SAME edep file
# (events [k*NSLICE : (k+1)*NSLICE]) over the full threshold x reset grid, and
# writes its own partial ti_mult_grid.npz. The one-time induction of a slice fits
# 'shared' memory (~25 GB for 250) and a 6 h wall, so nothing is lost to the wall.
#
# Submit (4 tasks x 250 = 1000 showers):
#   sbatch slurm/shower_grid_array.sh
#   EDEP=/path/showers.h5 NSLICE=250 sbatch --array=0-3 slurm/shower_grid_array.sh
# Then, after all tasks finish, merge:
#   sbatch --dependency=afterok:<arrayJobID> slurm/merge_grids.sh    # or run merge_grids.sh args by hand
#
# Watch:  squeue --me ;  tail -f grid_part_*_*.log
# Output: $ND_WORK/tistudy_grid_full/part_<k>/ti_mult_grid.npz  (merged into .../ti_mult_grid.npz)
#
# Scale up by adding tasks: --array=0-7 with NSLICE=250 -> 2000 showers, etc.
# =============================================================================
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus 1 -c 32 -N 1
#SBATCH -t 6:00:00
#SBATCH -J fsdcube_grid
#SBATCH --array=0-3
#SBATCH -o grid_part_%A_%a.log

NSLICE="${NSLICE:-250}"
THRESHOLDS="2500 3750 5000 6250 7500"
RESETS="-1 1024 512 256 128 64"

source ~/dune_sim.sh
nd_conda
cd "$ND_SRC"
GIT_TERMINAL_PROMPT=0 git pull --ff-only 2>&1 || echo "note: git pull skipped/failed -- running current checkout"

EDEP="${EDEP:-$ND_WORK/fsdcube_induction.EDEPSIM.h5}"
[ -f "$EDEP" ] || { echo "!! no edep file at $EDEP -- set EDEP=<path> and resubmit"; exit 1; }
k="$SLURM_ARRAY_TASK_ID"
OFFSET=$(( k * NSLICE ))
OUT="$ND_WORK/tistudy_grid_full/part_${k}"
mkdir -p "$OUT"
echo "task $k: edep=$EDEP  slice=[${OFFSET}:$(( OFFSET + NSLICE ))]  out=$OUT"

python -u tests/threshold_induction_study.py --config fsd_cube --mult-grid \
  --edep-h5 "$EDEP" --recenter-showers --edep-offset "$OFFSET" --n-events "$NSLICE" \
  --seed $(( 12345 + k )) --outdir "$OUT" \
  --thresholds $THRESHOLDS --resets $RESETS

echo "PART $k DONE -> $OUT/ti_mult_grid.npz"
