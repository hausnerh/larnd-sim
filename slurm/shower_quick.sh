#!/bin/bash -l
# =============================================================================
# slurm/shower_quick.sh  --  QUICK FSD Cube shower multiplicity-grid test, as a
# detached Slurm batch job (survives SSH drops; no terminal needed).
#
# A small, fast check on an EXISTING edep file: N=150 showers over the 5-threshold
# grid with a reduced reset set (-1, 512, 128). Use it to eyeball the multiplicity
# distributions before committing to the full N=1000 x all-resets grid.
#
# Submit:
#   sbatch slurm/shower_quick.sh                        # edep from $EDEP default below
#   EDEP=/path/to/showers.h5 sbatch slurm/shower_quick.sh   # point at your edep file
#   sbatch -A "$ND_ACCT_GPU" slurm/shower_quick.sh          # override the GPU account
#
# Watch:   squeue --me ;  tail -f shower_quick_<jobid>.log
# Output:  $ND_WORK/tistudy_mult_quick/ti_mult_grid.npz
#
# 'shared' GPU QOS = 1 GPU + 32 cores (~64 GB): N=150 needs ~15 GB, comfortable.
# (The full N=1000 grid needs '-q regular' for memory -- see run_fsdcube_induction.sh.)
# Edit -A to your GPU account (echo $ND_ACCT_GPU) if it is not dune_g.
# =============================================================================
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus 1 -c 32 -N 1
#SBATCH -t 2:00:00
#SBATCH -J fsdcube_shwr_q
#SBATCH -o shower_quick_%j.log

source ~/dune_sim.sh
nd_conda
cd "$ND_SRC"
GIT_TERMINAL_PROMPT=0 git pull --ff-only 2>&1 || echo "note: git pull skipped/failed -- running current checkout"

# edep input: export EDEP=<path> before sbatch to override; default = orchestrator's file
EDEP="${EDEP:-$ND_WORK/fsdcube_induction.EDEPSIM.h5}"
if [ ! -f "$EDEP" ]; then
  echo "!! no edep file at: $EDEP"
  echo "!! set one and resubmit:  EDEP=/path/to/showers.h5 sbatch slurm/shower_quick.sh"
  exit 1
fi
echo "edep: $EDEP"

python -u tests/threshold_induction_study.py --config fsd_cube --mult-grid \
  --edep-h5 "$EDEP" --recenter-showers --n-events 150 \
  --outdir "$ND_WORK/tistudy_mult_quick" \
  --thresholds 2500 3750 5000 6250 7500 --resets -1 512 128

echo "DONE -> $ND_WORK/tistudy_mult_quick/ti_mult_grid.npz"
