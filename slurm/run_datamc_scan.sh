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
#   slurm/run_datamc_scan.sh                      # defaults: prc 2 16, mkramer's edep, 16x50 showers
#   PRCS="2 16" NARRAY=8 NSLICE=250 THRBINS=1 NORM="single shower" slurm/run_datamc_scan.sh
#   DRYRUN=1 slurm/run_datamc_scan.sh            # print the sbatch chain WITHOUT submitting
#
# Knobs (all env, all optional):
#   PRCS        chip reset-register values to run (prc<P> = PERIODIC_   (default "2 16")
#               RESET_CYCLES = P), one grid + panel each. prc is a READOUT setting applied
#               downstream of the deposits, NOT an input selector: every prc reads the SAME
#               shared showers (SHOWER_EDEP) and differs ONLY by the reset it is read out at,
#               so grid_prc2 vs grid_prc16 is a clean controlled reset-rate comparison.
#               (mkramer's prc* output dirs all carry the identical input 'segments'.)
#   SIM_BASE    mkramer's larnd-sim OUTPUT base (carries 'segments') (default his segFFE3 skim_more)
#   EDEP_R      readout-radius dir to pull segments from (r-indep.)  (default 3)
#   EDEP_VAR    variant suffix in the dir name, e.g. _noFFE          (default none)
#   EDEP_TMPL   full edep path template with {prc}                   (default from SIM_BASE/EDEP_R/VAR)
#   SHOWER_PRC  which prc* dir to pull the (shared) showers from      (default 2; any works --
#               they carry the same segments)
#   SHOWER_EDEP the one shared shower source glob for ALL prc         (default EDEP_TMPL@SHOWER_PRC)
#   NARRAY      array tasks per prc  (--array=0..NARRAY-1)           (default 16)
#   NSLICE      showers per task (keep each task under WALL; real    (default 50)
#               showers are ~seconds each to induce, so 50/task fits a 6 h GPU easily)
#   WALL        per-array-task wall limit (sbatch -t)                (default 6:00:00)
#   AGG_RESERVE_S  wall seconds held back from induction for the      (default 9000 = 2.5 h)
#               aggregation+checkpoint phase (a few monster showers can eat a whole wall;
#               the induction phase stops at WALL-AGG_RESERVE_S and writes a PARTIAL grid,
#               which merge_mult_grids sums across tasks -- so nothing is lost, it's spread
#               over more slices, and no task dies to a timeout writing nothing).
#   TI_WALL_BUDGET_S  override the induction cap directly (seconds; default WALL-AGG_RESERVE_S)
#   RECENTER    1 = recenter showers; 0 = keep mkramer's positions   (default 0, his files are in-frame)
#   THRESHOLDS  override the sweep thresholds (e-, space list)        (default array-script's)
#   RESETS      per-prc reset nodes are set from PRCS (prc<P> -> reset P); this knob is no longer
#               used by this harness (drive a custom reset sweep via shower_grid_array.sh directly)
#   OUTROOT     per-prc grids under $OUTROOT/grid_prc<P>             (default $ND_WORK/tistudy_datamc)
#   NORM        plot norm scheme(s)                                   (default "single shower density")
#   THRBINS     1 = also make the per-pixel-threshold-binned plots    (default 0)
#   SIM_R       MC readout radii to draw as dashed curves in plots    (default "3 4")
#   DT_MAX/DT_BINS/DT_FIELD  time-between-hits (hitq_dt.png) axis knobs (default: auto/50/t_drift)
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
# prc = the chip PERIODIC_RESET_CYCLES the showers are read out at (prc2 -> 2, prc16 -> 16), a READOUT
# setting the sweep applies downstream of the deposits. mkramer's prc* output dirs all carry the SAME
# input 'segments', so every prc reads ONE shared shower source and differs ONLY by the reset -- a
# controlled comparison. (Reading per-prc dirs instead would compare the same showers to themselves.)
SHOWER_PRC="${SHOWER_PRC:-2}"
SHOWER_EDEP="${SHOWER_EDEP:-${EDEP_TMPL//\{prc\}/$SHOWER_PRC}}"
NARRAY="${NARRAY:-16}"
NSLICE="${NSLICE:-50}"
WALL="${WALL:-6:00:00}"
RECENTER="${RECENTER:-0}"
OUTROOT="${OUTROOT:-${ND_WORK:-.}/tistudy_datamc}"
LOGDIR="${LOGDIR:-$OUTROOT/logs}"
NORM="${NORM:-single shower density}"
THRBINS="${THRBINS:-0}"
SIM_R="${SIM_R:-3 4}"
PLOT_OUT="${PLOT_OUT:-$OUTROOT/plots}"
DRYRUN="${DRYRUN:-0}"

# Induction is heavy-tailed (a few monster showers take ~1 h each), so instead of praying a slice
# fits the wall we cap the INDUCTION phase and let each task write a PARTIAL grid for whatever it
# finished -- merge_mult_grids sums n_shower across tasks, so nothing is lost, it's just spread over
# more slices (a timed-out task writes NOTHING). TI_WALL_BUDGET_S (seconds) is that cap; by default
# it's WALL minus AGG_RESERVE_S held back for the cheaper aggregation+checkpoint phase that follows.
AGG_RESERVE_S="${AGG_RESERVE_S:-9000}"            # default 2.5 h reserved for aggregation
wall_to_secs() {                                  # SLURM time -> seconds: [D-]H:M:S, M:S, or bare minutes
  local t="$1" d=0 rest; case "$t" in *-*) d="${t%%-*}"; rest="${t#*-}";; *) rest="$t";; esac
  local IFS=:; set -- $rest
  case "$#" in
    3) echo $(( d*86400 + 10#$1*3600 + 10#$2*60 + 10#$3 ));;
    2) echo $(( d*86400 + 10#$1*60 + 10#$2 ));;
    1) echo $(( d*86400 + 10#$1*60 ));;
    *) echo 0;;
  esac
}
if [ -z "${TI_WALL_BUDGET_S:-}" ]; then
  _ws="$(wall_to_secs "$WALL")"
  if [ "$_ws" -gt 0 ]; then
    _b=$(( _ws - AGG_RESERVE_S )); _half=$(( _ws / 2 ))
    [ "$_b" -lt "$_half" ] && _b="$_half"         # never give aggregation more than half the wall
    TI_WALL_BUDGET_S="$_b"
  fi
fi
export TI_WALL_BUDGET_S="${TI_WALL_BUDGET_S:-0}"

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
echo "prc=[$PRCS] (= PERIODIC_RESET_CYCLES per grid)  array=0-$((NARRAY-1)) x $NSLICE showers  recenter=$RECENTER"
echo "wall=$WALL  induction budget=${TI_WALL_BUDGET_S}s (then partial-grid; reserve ${AGG_RESERVE_S}s for aggregation)"
echo "shared shower source (all prc): $SHOWER_EDEP"
echo "grids -> $OUTROOT/grid_prc<P>    plots -> $PLOT_OUT    logs -> $LOGDIR"
[ "$DRYRUN" = "1" ] && echo "*** DRYRUN: printing the sbatch chain, submitting nothing ***"
[ "$DRYRUN" = "1" ] || mkdir -p "$OUTROOT" "$LOGDIR" "$PLOT_OUT"

MERGE_DEPS=""
GRID_TOKENS=()
for P in $PRCS; do
  EDEP="$SHOWER_EDEP"                 # shared shower source for EVERY prc (same deposits)
  OUTBASE="$OUTROOT/grid_prc$P"
  RESETS="$P"                         # prc = PERIODIC_RESET_CYCLES: read these showers out at reset P
  BASE_RESET="$P"                     # ... and make it the grid's nominal node
  export EDEP OUTBASE NSLICE RECENTER RESETS BASE_RESET
  [ -n "${THRESHOLDS:-}" ] && export THRESHOLDS || true
  echo "-- prc$P: reset=$P cyc  edep=$EDEP"
  ARR=$(submit "arr$P" --export=ALL -J "grid_prc$P" -o "$LOGDIR/grid_prc${P}_%A_%a.log" \
        -t "$WALL" --array="0-$((NARRAY-1))" "$HERE/shower_grid_array.sh")
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
[ -n "${DT_MAX:-}" ] && export DT_MAX || true
[ -n "${DT_BINS:-}" ] && export DT_BINS || true
[ -n "${DT_FIELD:-}" ] && export DT_FIELD || true
PLOT=$(submit "plot" --export=ALL -J datamc_plot -o "$LOGDIR/datamc_plot_%j.log" \
       --dependency="afterok:$MERGE_DEPS" "$HERE/datamc_plot.sh" "${GRID_TOKENS[@]}")
echo "-- plot=$PLOT (afterok:$MERGE_DEPS)  ->  $PLOT_OUT"
echo "=== submitted: ${PRCS// /,} prc x (array+merge) + 1 plot.  watch: squeue --me ; tail -f $LOGDIR/*.log ==="
