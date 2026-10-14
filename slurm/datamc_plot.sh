#!/bin/bash -l
# =============================================================================
# slurm/datamc_plot.sh  --  the PLOTTING stage of the data/MC scan harness:
# overlay the per-prc FEE sweeps (produced + merged by run_datamc_scan.sh) on the
# real data / mkramer-MC kramer curves with tests/fsdcube_hitq_datamc.py. Pure
# h5py/numpy/matplotlib (no GPU), usually run as the final dependent job of
# run_datamc_scan.sh, but can be run by hand too:
#
#   PLOT_OUT=$ND_WORK/datamc_plots NORM="single shower" THRBINS=1 \
#     sbatch slurm/datamc_plot.sh 2=/path/grid_prc2/ti_mult_grid.npz 16=/path/grid_prc16/ti_mult_grid.npz
#
# The per-prc 'prc=path' grid tokens are the POSITIONAL args ($@) -> --grid.
# fsdcube_hitq_datamc.py needs h5py; if `nd_conda` lacks it, set PLOT_ENV_CMD to the
# activation that has it, e.g. PLOT_ENV_CMD='conda activate larnd'.
# =============================================================================
#SBATCH -A dune
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1 -c 16 -t 2:00:00
#SBATCH -J datamc_plot
#SBATCH -o datamc_plot_%j.log

set -e
source ~/dune_sim.sh
if [ -n "$PLOT_ENV_CMD" ]; then eval "$PLOT_ENV_CMD"; else nd_conda; fi
cd "$ND_SRC"

PRCS="${PRCS:-2 16}"
SIM_R="${SIM_R:-3 4}"
NORM="${NORM:-single shower density}"
PLOT_OUT="${PLOT_OUT:-$ND_WORK/datamc_plots}"
[ "${THRBINS:-0}" = "1" ] && THRBINS_FLAG="--thr-bins" || THRBINS_FLAG=""

if [ "$#" -eq 0 ]; then
  echo "!! no 'prc=path' grid tokens given as positional args -- plotting data/MC with NO sweep overlay"
  GRID_FLAG=""
else
  GRID_FLAG="--grid $*"
fi

echo "plotting: prc=[$PRCS] sim-r=[$SIM_R] norm=[$NORM] thrbins=${THRBINS:-0} out=$PLOT_OUT"
echo "  grids: $*"
python tests/fsdcube_hitq_datamc.py \
  --prc $PRCS --sim-r $SIM_R --norm $NORM $THRBINS_FLAG $GRID_FLAG \
  --outdir "$PLOT_OUT" $PLOT_EXTRA

echo "PLOTS -> $PLOT_OUT"
