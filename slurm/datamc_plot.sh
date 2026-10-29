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
#
# Produces, per prc and norm scheme: the split Hit-Q plots, pixel hit-multiplicity,
# per-pixel last-hit Q, AND the time-between-hits plot (hitq_dt.png = Δt between a pixel's
# consecutive hits). Tune the Δt axis with DT_MAX / DT_BINS / DT_FIELD (all optional).
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

# optional time-between-hits axis knobs (hitq_dt.png); unset -> the script's own defaults
DT_FLAG=""
[ -n "${DT_FIELD:-}" ] && DT_FLAG="$DT_FLAG --dt-field $DT_FIELD"
[ -n "${DT_MAX:-}" ]   && DT_FLAG="$DT_FLAG --dt-max $DT_MAX"
[ -n "${DT_BINS:-}" ]  && DT_FLAG="$DT_FLAG --dt-bins $DT_BINS"

# prc IS a periodic-reset window, so by default compare each data prc to the sim node at ITS OWN
# reset (reset cyc = prc * PRC_STEP_US / ts; prc2 -> 128, prc16 -> 1024). The grid's reset sweep
# already carries those nodes, so no rerun is needed. GRID_RESET_PER_PRC=0 reverts to the grid base.
[ "${GRID_RESET_PER_PRC:-1}" = "1" ] && GRID_RESET_FLAG="--grid-reset-per-prc" || GRID_RESET_FLAG=""
[ -n "${PRC_STEP_US:-}" ] && GRID_RESET_FLAG="$GRID_RESET_FLAG --prc-step-us $PRC_STEP_US"

if [ "$#" -eq 0 ]; then
  echo "!! no 'prc=path' grid tokens given as positional args -- plotting data/MC with NO sweep overlay"
  GRID_FLAG=""
else
  GRID_FLAG="--grid $*"
fi

echo "plotting: prc=[$PRCS] sim-r=[$SIM_R] norm=[$NORM] thrbins=${THRBINS:-0} out=$PLOT_OUT"
echo "  grids: $*"
echo "  dt-knobs:${DT_FLAG:- (defaults)}"
echo "  grid-reset:${GRID_RESET_FLAG:- grid-base (GRID_RESET_PER_PRC=0)}"
python tests/fsdcube_hitq_datamc.py \
  --prc $PRCS --sim-r $SIM_R --norm $NORM $THRBINS_FLAG $GRID_FLAG $DT_FLAG $GRID_RESET_FLAG \
  --outdir "$PLOT_OUT" $PLOT_EXTRA

echo "PLOTS -> $PLOT_OUT"
