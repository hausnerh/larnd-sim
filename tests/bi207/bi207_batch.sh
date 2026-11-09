#!/bin/bash
# =============================================================================
# tests/bi207/bi207_batch.sh -- end-to-end Bi-207 FSD-cube pipeline as a SLURM batch job on
# Perlmutter, so the GPU + ndlar_flow legs run unattended instead of timing out on an
# interactive node. Runs: [generate edep] -> translate onto the fsd_cube cathode ->
# simulate_pixels (FEE, GPU) -> ndlar_flow (h5flow, container) -> check_bi207_fee analysis.
#
# SUBMIT (after `source ~/dune_sim.sh` + `source $ND_SRC/tests/bi207/bi207_sim.sh`):
#     ND_ACCT=<your nersc GPU account, e.g. m1234_g>  bi207_submit
#     BI207_N=5000 ND_ACCT=m1234_g  bi207_submit        # also (re)generate 5000 decays first
# or submit by hand:
#     sbatch -A <acct>_g -o $ND_WORK/bi207_%j.out $ND_SRC/tests/bi207/bi207_batch.sh
# or just run it on an interactive GPU node (nd_node):
#     bash $ND_SRC/tests/bi207/bi207_batch.sh
# Watch it:  squeue --me        Log:  $ND_WORK/bi207_<jobid>.out
#
# Knobs (env vars; bi207_submit forwards them via --export=ALL):
#   BI207_N     if >0 AND no <tag>.EDEPSIM.h5 exists yet, generate that many decays first
#   BI207_TAG   file tag                     (default bi207)
#   BI207_INSET source inset into LAr, cm    (default 0.5)
#   ND_SETUP    path to your dune_sim.sh     (default $HOME/dune_sim.sh)
# Override SLURM at submit time: ND_QOS / ND_TIME (bi207_submit passes -q/-t), e.g.
#   ND_QOS=debug ND_TIME=00:30:00 ND_ACCT=m1234_g bi207_submit   # quick smoke test
#
# Defaults below are for a single-GPU "shared" job (cheapest for 1 GPU). A CLI -q/-t from
# bi207_submit overrides these directives.
#SBATCH -J bi207
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -o bi207_%j.out
#SBATCH --image=mjkramer/sim2x2:ndlar011
# =============================================================================
set -o pipefail

source "${ND_SETUP:-$HOME/dune_sim.sh}"      || { echo "cannot source dune_sim.sh (set ND_SETUP)"; exit 1; }
source "${ND_SRC}/tests/bi207/bi207_sim.sh"  || { echo "cannot source bi207_sim.sh (is ND_SRC set?)"; exit 1; }
nd_conda                                     || { echo "nd_conda failed"; exit 1; }

TAG="${BI207_TAG:-bi207}"
INSET="${BI207_INSET:-0.5}"
N="${BI207_N:-0}"
cd "${ND_WORK}" || exit 1
echo "### Bi-207 FSD-cube pipeline  tag=${TAG} inset=${INSET}  host=$(hostname)  $(date)"

# ndlar_flow clone (needs yamls/fsdcube_flow; the container's bundled copy is too old) --------
ndflow="${ND_FLOW:-${ND_WORK}/ndlar_flow}"
if [ ! -d "${ndflow}/yamls/fsdcube_flow" ]; then
  echo "### cloning ndlar_flow -> ${ndflow}"
  git clone --depth 1 https://github.com/DUNE/ndlar_flow "${ndflow}" || exit 1
fi
export ND_FLOW="${ndflow}"

# [0] generate edep only if missing and explicitly requested (BI207_N>0) ---------------------
if [ ! -f "${ND_WORK}/${TAG}.EDEPSIM.h5" ]; then
  if [ "${N}" -gt 0 ] 2>/dev/null; then
    echo "### [0] generate ${N} Bi-207 decays"
    bi207_gen "${N}" "${TAG}" || exit 3
  else
    echo "no ${ND_WORK}/${TAG}.EDEPSIM.h5 and BI207_N not set -- set BI207_N=<ndecays> to generate"; exit 2
  fi
fi

# [1-3] translate -> simulate_pixels (GPU) -> ndlar_flow -------------------------------------
echo "### [1-3] translate + simulate_pixels + ndlar_flow"
bi207_fsd "${TAG}" "${INSET}" || exit 4

# [4] FEE Q-spectrum analysis ---------------------------------------------------------------
echo "### [4] analysis -> ${ND_WORK}/${TAG}_fee"
python "${ND_SRC}/tests/bi207/check_bi207_fee.py" "${ND_WORK}/${TAG}_fsd.FLOW.hdf5" \
       --outdir "${ND_WORK}/${TAG}_fee" || exit 5

echo "### ALL DONE -> ${ND_WORK}/${TAG}_fee/bi207_fee.png + bi207_fee.txt   $(date)"
