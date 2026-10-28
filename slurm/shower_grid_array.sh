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
#
# EDEP may be ONE edep file, a quoted glob, or a directory -- e.g. mkramer's larnd-sim
# OUTPUT (it carries its input 'segments' through), so the sweep re-runs HIS showers:
#   EDEP='/pscratch/sd/m/mkramer/test/larnd-sim.segFFE3/output/skim_more/prc*_r*_patch_noDrop/*.hdf5' \
#     NSLICE=250 sbatch --array=0-3 slurm/shower_grid_array.sh
# Set RECENTER=0 to keep his in-frame positions (default recenters, as for own showers).
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
THRESHOLDS="${THRESHOLDS:-2500 3750 5000 6250 7500}"
RESETS="${RESETS:--1 1024 512 256 128 64}"
# BASE_RESET pins the grid's NOMINAL reset (PERIODIC_RESET_CYCLES) -> --base-reset. The periodic
# reset is a readout (FEE) setting applied downstream of the shared edep deposits, so this is how a
# sample is read out at its chip reset-register value (e.g. prc2 -> 2, prc16 -> 16). Empty = use the
# loaded config's own default. The base value is always added to the grid's reset nodes.
BASE_RESET="${BASE_RESET:-}"
# OUTBASE lets an orchestrator point each run (e.g. per-prc) at its own grid dir; parts land in
# $OUTBASE/part_<k> and merge_grids.sh (OUT=$OUTBASE) collapses them to $OUTBASE/ti_mult_grid.npz.
OUTBASE="${OUTBASE:-$ND_WORK/tistudy_grid_full}"

source ~/dune_sim.sh
nd_conda
cd "$ND_SRC"
# Name the branch explicitly: a bare `git pull --ff-only` here fails with "Cannot
# fast-forward to multiple branches" and leaves a STALE checkout running silently.
BR="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)"
if GIT_TERMINAL_PROMPT=0 git pull --ff-only origin "$BR" 2>&1; then
  echo "run tree updated to $(git rev-parse --short HEAD) ($BR)"
else
  echo "!! git pull failed -- RUNNING STALE CHECKOUT $(git rev-parse --short HEAD) ($BR)."
  echo "!! update $ND_SRC by hand (git fetch && git checkout $BR && git pull) and resubmit."
fi

EDEP="${EDEP:-$ND_WORK/fsdcube_induction.EDEPSIM.h5}"
# EDEP may be a file, a glob, or a directory (Python-side expansion); accept any that resolves.
case "$EDEP" in
  *[*?[]*) compgen -G "$EDEP" >/dev/null || { echo "!! --edep-h5 glob matched nothing: $EDEP"; exit 1; } ;;
  *) [ -e "$EDEP" ] || { echo "!! no edep path at $EDEP -- set EDEP=<file|glob|dir> and resubmit"; exit 1; } ;;
esac
RECENTER="${RECENTER:-1}"
[ "$RECENTER" = "0" ] && RECENTER_FLAG="" || RECENTER_FLAG="--recenter-showers"
k="$SLURM_ARRAY_TASK_ID"
OFFSET=$(( k * NSLICE ))
OUT="$OUTBASE/part_${k}"
mkdir -p "$OUT"
echo "task $k: edep=$EDEP  recenter=$RECENTER  slice=[${OFFSET}:$(( OFFSET + NSLICE ))]  out=$OUT"
echo "        thresholds=[$THRESHOLDS]  resets=[$RESETS]  base_reset=${BASE_RESET:-config-default}"

# A crash here MUST fail the task: this array feeds an `afterok` merge, and a merge that
# fires on a missing/partial part silently produces a wrong grid. So gate the "DONE" on
# BOTH a zero python exit AND a non-empty grid file, and propagate the failure otherwise.
if python -u tests/threshold_induction_study.py --config fsd_cube --mult-grid \
     --edep-h5 "$EDEP" $RECENTER_FLAG --edep-offset "$OFFSET" --n-events "$NSLICE" \
     --seed $(( 12345 + k )) --outdir "$OUT" \
     --thresholds $THRESHOLDS --resets $RESETS ${BASE_RESET:+--base-reset $BASE_RESET}; then
  [ -s "$OUT/ti_mult_grid.npz" ] || { echo "!! PART $k: python exited 0 but no $OUT/ti_mult_grid.npz" >&2; exit 2; }
  echo "PART $k DONE -> $OUT/ti_mult_grid.npz"
else
  rc=$?
  echo "!! PART $k FAILED (python exit $rc) -- no grid written; downstream merge will NOT run" >&2
  exit "$rc"
fi
