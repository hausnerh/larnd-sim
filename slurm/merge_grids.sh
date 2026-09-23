#!/bin/bash -l
# =============================================================================
# slurm/merge_grids.sh  --  merge the shower_grid_array.sh partials into ONE
# full-stats grid + plots. Pure numpy/matplotlib (no GPU), so it can run on a
# login node directly, or as a tiny CPU job after the array with a dependency:
#   sbatch --dependency=afterok:<arrayJobID> slurm/merge_grids.sh
# or just run the python line below on a login node once the parts exist.
# =============================================================================
#SBATCH -A dune
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1 -c 8 -t 0:30:00
#SBATCH -J fsdcube_merge
#SBATCH -o grid_merge_%j.log

source ~/dune_sim.sh
nd_conda
cd "$ND_SRC"
OUT="${OUT:-$ND_WORK/tistudy_grid_full}"

python tests/threshold_induction_study.py --config fsd_cube \
  --merge-grids "$OUT"/part_*/ti_mult_grid.npz --outdir "$OUT"

echo "MERGED -> $OUT/ti_mult_grid.npz (+ mult_*.png, incl. _perhit)"
