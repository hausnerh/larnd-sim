#!/bin/bash -l
# =============================================================================
# tests/bi207/bi207_bigstats.sh -- large-stats Bi-207 FSD-cube generation as a SLURM JOB ARRAY.
#
# Each array task k generates NCHUNK Bi-207 decays with a DISTINCT RNG seed (BI207_SEED, so the
# chunks are statistically independent), translates them onto the fsd_cube cathode, and runs
# simulate_pixels -> a per-chunk LARND. NO ndlar_flow (the calib/larnd analyses read the LARND +
# EDEPSIM directly, not the FLOW file), so each task is just edep-sim + simulate_pixels and fits a
# short wall. Files land in $ND_WORK as bi207big_<k>.EDEPSIM.h5 and bi207big_<k>_fsd.LARND.h5.
#
# AGGREGATE across chunks by globbing the per-chunk files (check_bi207_calib.py takes many). NOTE the
# --edep glob ends in [0-9] so it matches ONLY the raw per-chunk edep (bi207big_<k>.EDEPSIM.h5) and NOT
# the translated bi207big_<k>_fsd.EDEPSIM.h5 (which would double-count emitted decays):
#   python $ND_SRC/tests/bi207/check_bi207_calib.py "$ND_WORK"/bi207big_*_fsd.LARND.h5 \
#       --edep "$ND_WORK"/bi207big_*[0-9].EDEPSIM.h5 --outdir $ND_WORK/bi207_calib_big
#
# SUBMIT (validate a few tasks first, then scale the array):
#   NCHUNK=10000 sbatch -A $ND_ACCT_GPU --array=0-2  $ND_SRC/tests/bi207/bi207_bigstats.sh  # 3e4 (test)
#   NCHUNK=10000 sbatch -A $ND_ACCT_GPU --array=0-99 $ND_SRC/tests/bi207/bi207_bigstats.sh  # 1e6
# Knobs: NCHUNK (decays/task, default 10000), BI207_INSET (cm off cathode, default 0.5).
# Watch: squeue --me ;  tail -f $ND_WORK/bi207big_<arrayID>_*.out
#
# Re-runnable: a task whose _fsd.LARND.h5 already exists is skipped, so a partial array can be
# resubmitted to fill the gaps (same --array).
# =============================================================================
#SBATCH -J bi207big
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 06:00:00
#SBATCH -o bi207big_%A_%a.out
#SBATCH --image=mjkramer/sim2x2:ndlar011
set -o pipefail

source "${ND_SETUP:-$HOME/dune_sim.sh}"      || { echo "cannot source dune_sim.sh (set ND_SETUP)"; exit 1; }
source "${ND_SRC}/tests/bi207/bi207_sim.sh"  || { echo "cannot source bi207_sim.sh (is ND_SRC set?)"; exit 1; }
nd_conda                                     || { echo "nd_conda failed"; exit 1; }

k="${SLURM_ARRAY_TASK_ID:-0}"
NCHUNK="${NCHUNK:-10000}"
INSET="${BI207_INSET:-0.5}"
tag="bi207big_${k}"
export BI207_SEED=$(( 20001 + k * 2 ))        # distinct odd seed per task (read by bi207_gun)
cd "${ND_WORK}" || exit 1
echo "### bi207 big-stats task ${k}: ${NCHUNK} decays, seed ${BI207_SEED}  host=$(hostname)  $(date)"

larnd="${ND_WORK}/${tag}_fsd.LARND.h5"
if [ -s "${larnd}" ]; then
  echo "### ${larnd} already exists -- skipping task ${k}"; exit 0
fi

# [0] generate NCHUNK seeded decays + convert -> ${tag}.EDEPSIM.h5  (bi207_gen cd's to ND_WORK)
[ -f "${ND_WORK}/${tag}.EDEPSIM.h5" ] || bi207_gen "${NCHUNK}" "${tag}" || exit 3
# [1] translate onto the fsd_cube cathode
python "${ND_SRC}/tests/bi207/bi207_to_fsd.py" "${ND_WORK}/${tag}.EDEPSIM.h5" "${ND_WORK}/${tag}_fsd.EDEPSIM.h5" \
    --inset "${INSET}" || exit 4
# [2] simulate_pixels (FEE, conda+GPU) -> per-chunk LARND (distinct FEE seed too)
python "${ND_SRC}/cli/simulate_pixels.py" \
    --input_filename="${ND_WORK}/${tag}_fsd.EDEPSIM.h5" --output_filename="${larnd}" \
    --config=fsd_cube --rand_seed=$(( k + 1 )) || exit 5
echo "### DONE task ${k} -> ${larnd}   $(date)"
