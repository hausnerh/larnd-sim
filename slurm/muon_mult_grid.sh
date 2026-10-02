#!/bin/bash -l
# =============================================================================
# slurm/muon_mult_grid.sh  --  VERTICAL-MUON hit-multiplicity + per-hit-Q grid,
# the muon counterpart to the shower grid. Downward (cosmic-like) muons: in-plane
# azimuth = 90 deg, so each track runs along the short/vertical y anode axis over
# ONE x-column of pixels, at fixed drift depth (out-of-plane theta = 0, isochronous).
# Full threshold x reset grid, tagged sample='muons'. Muons are sparse (~80 pixels
# each, one column) so the pre-signal cache is small -> one short 'shared' GPU job.
#
# Submit:  sbatch slurm/muon_mult_grid.sh
# Watch:   squeue --me ;  tail -f muon_mult_*.log
# Output:  $ND_WORK/tistudy_muon_mult/ti_mult_grid.npz  (+ mult_*, mult_spectrum_*,
#          shoulder_qsplit_* plots, all labelled 'muons')
#
# (To instead sweep other in-plane angles, pass e.g. --muon-azimuths 0 45 90.)
# =============================================================================
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus 1 -c 32 -N 1
#SBATCH -t 6:00:00
#SBATCH -J fsdcube_mumult
#SBATCH -o muon_mult_%j.log

source ~/dune_sim.sh
nd_conda
cd "$ND_SRC"
GIT_TERMINAL_PROMPT=0 git pull --ff-only 2>&1 || echo "note: git pull skipped/failed -- running current checkout"

OUT="$ND_WORK/tistudy_muon_mult"
mkdir -p "$OUT"
echo "vertical (downward) muons: in-plane azimuth 90 deg, one x-column -> $OUT"

python -u tests/threshold_induction_study.py --config fsd_cube --mult-grid --mult-grid-muons \
  --n-events 300 --muon-azimuths 90 --muon-length 200 --outdir "$OUT" \
  --thresholds 2500 3750 5000 6250 7500 --resets -1 1024 512 256 128 64
echo "DONE -> $OUT/ti_mult_grid.npz (sample=muons, vertical / in-plane azimuth 90 deg)"
