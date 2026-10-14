#!/bin/bash -l
# =============================================================================
# slurm/run_datamc_scan.sh  --  ONE harness that orchestrates the whole FSDCube
# data/MC threshold-sweep comparison end to end, as a chain of dependent Slurm jobs:
#
#   for each prc:  shower_grid_array.sh  (GPU array, re-runs mkramer's showers through
#                    the threshold x reset grid)   -->  merge_grids.sh  (CPU, afterok)
#   then:          datamc_plot.sh  (CPU, afterok on ALL merges) overlays each prc's own
#                    sweep on the real data / mkramer-MC kramer curves.
#
# Run it on a LOGIN NODE (it only submits; the work runs in the queued jobs):
#   slurm/run_datamc_scan.sh                      # defaults: prc 2 16, mkramer's edep, 4x250 showers
#   PRCS="2 16" NARRAY=8 NSLICE=250 THRBINS=1 NORM="single shower" slurm/run_datamc_scan.sh
#   DRYRUN=1 slurm/run_datamc_scan.sh            # print the sbatch chain WITHOUT submitting
#
# Knobs (all env, all optional):
#   PRCS        parity configs to run, one grid + panel each        (default "2 16")
#   SIM_BASE    mkramer's larnd-sim OUTPUT base (carries 'segments') (default his segFFE3 skim_more)
#   EDEP_R      readout-radius dir to pull segments from (r-indep.)  (default 3)
#   EDEP_VAR    variant suffix in the dir name, e.g. _noFFE          (default none)
#   EDEP_TMPL   full edep path template with {prc}                   (default from SIM_BASE/EDEP_R/VAR)
#   NARRAY      array tasks per prc  (--array=0..NARRAY-1)           (default 4)
#   NSLICE      showers per task                                     (default 250)
#   RECENTER    1 = recenter showers; 0 = keep mkramer's positions   (default 0, his files are in-frame)
#   THRESHOLDS  override the sweep thresholds (e-, space list)        (default array-script's)
#   RESETS      override the sweep resets (cycles, -1=off)            (default array-script's)
#   OUTROOT     per-prc grids under $OUTROOT/grid_prc<P>             (default $ND_WORK/tistudy_datamc)
#   NORM        plot norm scheme(s)                                   (default "single shower density")
#   THRBINS     1 = also make the per-pixel-threshold-binned plots    (default 0)
#   SIM_R       MC readout radii to draw as dashed curves in plots    (default "3 4")
#   PLOT_OUT    where the plots land                                  (default $OUTROOT/plots)
#   PLOT_ENV_CMD env activation for the plot job if `nd_conda` lacks h5py (e.g. 'conda activate larnd')
#   DRYRUN      1 = print the chain instead of submitting
# Watch:  squeue --me ;  tail -f $OUTROOT/logs/*.log
# =============================================================================
set -euo pipefail
[ -f ~/dune_sim.sh ] && source ~/dune_sim.sh || true

PRCS="${PRCS:-2 16}"
SIM_BASE="${SIM_BASE:-/pscratch/sd/m/mkramer/test/larnd-sim.segFFE3/output/skim_more}"
EDEP_R="${EDEP_R:-3}"
EDEP_VAR="${EDEP_VAR:-}"
# NB: build the default in a plain assignment, NOT inside ${EDEP_TMPL:-...}; a literal {prc} inside a
# ${:-default} throws off bash's brace matching. {prc} here is a literal placeholder substituted per prc.
if [ -z "${EDEP_TMPL:-}" ]; then
  EDEP_TMPL="$SIM_BASE/prc{prc}_r${EDEP_R}${EDEP_VAR}_patch_noDrop/*.hdf5"
fi
NARRAY="${NARRAY:-4}"
NSLICE="${NSLICE:-250}"
RECENTER="${RECENTER:-0}"
OUTROOT="${OUTROOT:-${ND_WORK:-.}/tistudy_datamc}"
LOGDIR="${LOGDIR:-$OUTROOT/logs}"
NORM="${NORM:-single shower density}"
THRBINS="${THRBINS:-0}"
SIM_R="${SIM_R:-3 4}"
PLOT_OUT="${PLOT_OUT:-$OUTROOT/plots}"
DRYRUN="${DRYRUN:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit() {  # $1 = label (for the synthetic dry-run jobid); rest = sbatch args + script [+ script args]
  local label="$1"; shift
  if [ "$DRYRUN" = "1" ]; then
    echo "  DRY> sbatch --parsable $*" >&2
    echo "DRY_${label}"
  else
    sbatch --parsable "$@"
  fi
}

echo "=== FSDCube data/MC scan harness ==="
echo "prc=[$PRCS]  array=0-$((NARRAY-1)) x $NSLICE showers  recenter=$RECENTER"
echo "edep template: $EDEP_TMPL"
echo "grids -> $OUTROOT/grid_prc<P>    plots -> $PLOT_OUT    logs -> $LOGDIR"
[ "$DRYRUN" = "1" ] && echo "*** DRYRUN: printing the sbatch chain, submitting nothing ***"
[ "$DRYRUN" = "1" ] || mkdir -p "$OUTROOT" "$LOGDIR" "$PLOT_OUT"

MERGE_DEPS=""
GRID_TOKENS=()
for P in $PRCS; do
  EDEP="${EDEP_TMPL//\{prc\}/$P}"
  OUTBASE="$OUTROOT/grid_prc$P"
  export EDEP OUTBASE NSLICE RECENTER
  [ -n "${THRESHOLDS:-}" ] && export THRESHOLDS || true
  [ -n "${RESETS:-}" ] && export RESETS || true
  echo "-- prc$P: edep=$EDEP"
  ARR=$(submit "arr$P" --export=ALL -J "grid_prc$P" -o "$LOGDIR/grid_prc${P}_%A_%a.log" \
        --array="0-$((NARRAY-1))" "$HERE/shower_grid_array.sh")
  export OUT="$OUTBASE"
  MRG=$(submit "mrg$P" --export=ALL -J "merge_prc$P" -o "$LOGDIR/merge_prc${P}_%j.log" \
        --dependency="afterok:$ARR" "$HERE/merge_grids.sh")
  echo "   array=$ARR  merge=$MRG (afterok:$ARR)  grid=$OUTBASE/ti_mult_grid.npz"
  MERGE_DEPS="${MERGE_DEPS:+$MERGE_DEPS:}$MRG"
  GRID_TOKENS+=("$P=$OUTBASE/ti_mult_grid.npz")
done

export PRCS SIM_R NORM THRBINS PLOT_OUT
[ -n "${PLOT_ENV_CMD:-}" ] && export PLOT_ENV_CMD || true
[ -n "${PLOT_EXTRA:-}" ] && export PLOT_EXTRA || true
PLOT=$(submit "plot" --export=ALL -J datamc_plot -o "$LOGDIR/datamc_plot_%j.log" \
       --dependency="afterok:$MERGE_DEPS" "$HERE/datamc_plot.sh" "${GRID_TOKENS[@]}")
echo "-- plot=$PLOT (afterok:$MERGE_DEPS)  ->  $PLOT_OUT"
echo "=== submitted: ${PRCS// /,} prc x (array+merge) + 1 plot.  watch: squeue --me ; tail -f $LOGDIR/*.log ==="
