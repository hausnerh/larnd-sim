#!/bin/bash -l
# =============================================================================
# slurm/muon_burst.sh  --  FSD Cube through-going muon BURST study, as a detached
# Slurm batch job (survives SSH drops; no terminal needed).
#
# Produces the muon threshold/reset scans + burst-mode band capture AND dumps the
# full-resolution waveforms along the WHOLE muon strip -- every on-track collector
# (coll_*.txt) and every +-1-pitch pure-induction neighbor (induction_*.txt) --
# for --wf-txt events. Muons are VERTICAL (downward): in-plane azimuth = 90 deg, i.e. the track
# runs along the short/vertical y anode axis over ONE x-column of pixels, at fixed drift depth
# (out-of-plane theta = 0, isochronous) -- cosmic-like, not the horizontal/beam orientation.
#
# Submit:
#   sbatch slurm/muon_burst.sh                    # uses the -A account below
#   sbatch -A "$ND_ACCT_GPU" slurm/muon_burst.sh  # override the GPU account
#
# Watch:   squeue --me ;  tail -f muon_burst_<jobid>.log
# Output:  $ND_WORK/tistudy_muon_burst   (+ induction_waveforms/{coll,induction}_*.txt)
#
# 'shared' GPU QOS = 1 GPU + 32 cores (~64 GB); the muon run fits easily. Change
# the -A line to your GPU account (check with: echo $ND_ACCT_GPU) if it is not dune_g.
# =============================================================================
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus 1 -c 32 -N 1
#SBATCH -t 6:00:00
#SBATCH -J fsdcube_muon
#SBATCH -o muon_burst_%j.log

source ~/dune_sim.sh          # ND_* env vars + nd_conda/nd_* helpers
nd_conda
cd "$ND_SRC"
# pull the latest study code; ff-only + no prompt so a batch job can never hang on it
GIT_TERMINAL_PROMPT=0 git pull --ff-only 2>&1 || echo "note: git pull skipped/failed -- running current checkout"

python -u tests/threshold_induction_study.py --config fsd_cube --burst-only \
  --burst-thetas 0 --muon-length 200 --muon-events 150 --burst-events 40 \
  --burst-neighbors 3 --burst-inductions 0.5 1.0 1.5 --muon-scan-events 60 \
  --muon-thresholds 2000 2500 3000 4000 5000 \
  --muon-azimuth 90 --wf-txt 2 --outdir "$ND_WORK/tistudy_muon_burst"

echo "DONE -> $ND_WORK/tistudy_muon_burst"
