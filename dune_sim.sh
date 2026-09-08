# ~/dune_sim.sh  —  DUNE ND-LAr shower-sim helpers

# Exports
export ND_IMG=mjkramer/sim2x2:ndlar011
export ND_ACCT_GPU=dune_g
export ND_ACCT_CPU=dune
export ND_WORK=${SCRATCH:-$HOME}/showers
export ND_GEOM=$ND_WORK/ND_Production/geometry/single_module0.gdml
export ND_ACTIVE_VOL=volTPCActive
export ND_VERTEX="-15.295 -21.824 0.000"
export ND_SD=TPCActive_shape
export ND_CUDA=$ND_WORK/cuda
export ND_LS=/opt/generators/larnd-sim/larndsim
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
export NUMBA_CUDA_WARN_ON_IMPLICIT_COPY=0

# Containerize
incont()
{
  local mods=cvmfs
  nvidia-smi -L >/dev/null 2>&1 && mods=cvmfs,gpu
  shifter --image="${ND_IMG}" --module="${mods}" -- bash -lc "$*"
}

# Setup
nd_setup()
{
  mkdir -p "${ND_WORK}" && cd "${ND_WORK}" || return 1
  [ -d ND_Production ] || git clone https://github.com/DUNE/ND_Production.git
  shifterimg images 2>/dev/null | grep -q sim2x2 || shifterimg pull "${ND_IMG}"
  echo "setup done in ${ND_WORK}"
}

nd_cuda_setup()
{
  command -v module >/dev/null 2>&1 && module load cudatoolkit/12.4 2>/dev/null
  mkdir -p "$ND_CUDA/lib64" "$ND_CUDA/nvvm/lib64" "$ND_CUDA/nvvm/libdevice"
  cp -auP "$CUDA_HOME"/lib64/lib*.so*                    "$ND_CUDA/lib64/"           2>/dev/null
  cp -auP "$CUDA_HOME"/../../math_libs/*/lib64/lib*.so*  "$ND_CUDA/lib64/"           2>/dev/null
  cp -auP "$CUDA_HOME"/nvvm/lib64/lib*.so*               "$ND_CUDA/nvvm/lib64/"      2>/dev/null
  cp -auP "$CUDA_HOME"/nvvm/libdevice/*                  "$ND_CUDA/nvvm/libdevice/"  2>/dev/null
  echo "staged CUDA in $ND_CUDA ($(ls "$ND_CUDA/lib64" | wc -l) libs)"
}

# Node Noodling
nd_node()     { salloc -N 1 -C gpu -q interactive -t "${1:-2}:00:00" -A "${ND_ACCT_GPU}"; }
nd_node_cpu() { salloc -N 1 -C cpu -q interactive -t "${1:-1}:00:00" -A "${ND_ACCT_CPU}"; }
nd_cd()       { cd "${ND_WORK}"; }

# Active Volume
nd_active()
{
  incont "python -c \"import ROOT; ROOT.TGeoManager.Import('${ND_GEOM}'); [print(v.GetName()) for v in ROOT.gGeoManager.GetListOfVolumes() if any(k in v.GetName() for k in ('Active','TPC','LAr'))]\""
}

# Gun
nd_gun()
{
  local particle=${1:-e-} ene=${2:-300} mac=${3:-e_gum.mac}
  cat > "${mac}" <<EOF
/edep/hitSeparation ${ND_SD} -1 mm
/edep/hitSagitta    ${ND_SD} 1.0 mm
/edep/hitLength     ${ND_SD} 1.0 mm
/edep/update
/gps/particle ${particle}
/gps/ene/type Mono
/gps/ene/mono ${ene} MeV
/gps/position ${ND_VERTEX} cm
/gps/direction 0 0 1
EOF
echo "wrote ${mac} (${particle} @ ${ene} MeV at (${ND_VERTEX}) cm, SD=${ND_SD})"
}

# Pipeline
nd_edep() # nd_edep <macro> <out.root> [nEvents]
{
  incont "edep-sim -C -g '${ND_GEOM}' -o '${2}' -e ${3:-200} '${1}'"
}

nd_convert()  # nd_convert <in.root> <out.h5>
{
  incont "export CPATH=/opt/generators/edep-sim/install/include/EDepSim:\$CPATH; \
	  export ROOT_INCLUDE_PATH=/opt/generators/edep-sim/install/include/EDepSim:\$ROOT_INCLUDE_PATH; \
	  dumpTree.py '${1}' '${2}' --keep_all_dets"
}

nd_verify()
{
  incont "python -c \"import h5py; s=h5py.File('${1}')['segments']; d=float(s['dE'][...].sum()); print('segments', s.shape, 'total_dE_MeV', d)\""
}

nd_larnd() # nd_run [particle] [MeV] [nEvents] [tag]
{
  command -v module >/dev/null 2>&1 && module load cudatoolkit/12.4 2>/dev/null
  [ -e "${ND_CUDA}/lib64/libcudart.so.12" ] || nd_cuda_setup
  incont "export CUDA_HOME=${ND_CUDA}; \
	  export LD_LIBRARY_PATH=${ND_CUDA}/lib64:${ND_CUDA}/nvvm/lib64:\$LD_LIBRARY_PATH; \
	  simulate_pixels.py '${1}' \
	  '${ND_LS}/pixel_layouts/multi_tile_layout-2.4.16.yaml' \
	  '${ND_LS}/detector_properties/module0.yaml' \
	  '${ND_LS}/simulation_properties/singles_sim.yaml' \
	  '${2}' \
	  --response_file '${ND_LS}/bin/response_44.npy' \
	  --light_simulated False --rand_seed ${3:-1}"
}

nd_run() # nd_run [particle] [MeV] [nEvents] [tag]
{
  local particle=${1:-e-} ene=${2:-300} n=${3:-200} tag=${4:-shower}
  cd "${ND_WORK}" || return 1
  nd_gun "${particle}" "${ene}" "${tag}.mac"           || return 2
  nd_edep "${tag}.mac" "${tag}.EDEPSIM.root" "${n}"    || return 3
  nd_convert "${tag}.EDEPSIM.root" "${tag}.EDEPSIM.h5" || return 4
  nd_verify "${tag}.EDEPSIM.h5"
  nd_larnd "${tag}.EDEPSIM.h5" "${tag}.LARND.h5"       || return 5
  echo "DONE -> ${ND_WORK}/${tag}.LARND.h5"
}

# Conda env
export ND_CONDA_ENV=${SCRATCH:-$HOME}/conda_envs/larnd
export ND_SRC=$ND_WORK/larnd-sim-dev
export ND_BRANCH=testing/hhausner
export ND_CUDA_VERSION=12.2

_nd_conda_hook()
{
  module load python >/dev/null 2>&1
  local base; base=$(conda info --base 2>/dev/null)
  [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ] && source "${base}/etc/profile.d/conda.sh"
}

nd_conda()
{
  _nd_conda_hook
  conda activate "${ND_CONDA_ENV}" 2>/dev/null || \
    { echo "env ${ND_CONDA_ENV} missing -- run: nd_conda_create"; return 1; }
}

nd_conda_create()
{
  _nd_conda_hook
  mkdir -p "$(dirname "${ND_CONDA_ENV}")"
  if [ ! -d "${ND_CONDA_ENV}" ]; then
    echo ">> creating conda env (${ND_CONDA_ENV}, cuda ${ND_CUDA_VERSION})"
    conda create -y -p "${ND_CONDA_ENV}" -c conda-forge \
      python=3.11 numba cupy "cuda-version=${ND_CUDA_VERSION}" cuda-nvcc cuda-nvrtc \
      h5py scipy pyyaml fire matplotlib tqdm || return 1
  fi
  if [ ! -d "${ND_SRC}/.git" ]; then
    echo ">> cloning larnd-sim (${ND_BRANCH})"
    git clone https://github.com/hausnerh/larnd-sim.git "${ND_SRC}" || return 1
    ( cd "${ND_SRC}" && git checkout "${ND_BRANCH}" 2>/dev/null \
      || echo "  !! branch ${ND_BRANCH} not on remote -- push it or scp your files" )
  fi
  conda activate "${ND_CONDA_ENV}" || return 1
  echo ">> pip install -e larnd-sim"
  ( cd "${ND_SRC}" && SKIP_CUPY_INSTALL=1 pip install -e . ) || return 1
  echo ">> done. On a GPU node, verify with: nd_conda_check"
}

nd_conda_check() # GPU
{
  nd_conda || return 1
  python - <<'PY'
import numpy as np, numba
from numba import cuda
print("numba", numba.__version__)
@cuda.jit
def add1(a):
    i = cuda.grid(1)
    if i < a.size: a[i] += 1.0
a = cuda.to_device(np.zeros(16, dtype=np.float32)); add1[1, 16](a)
print("plain kernel OK:", a.copy_to_host()[:3])
from numba.cuda.random import create_xoroshiro128p_states
print("rng OK:", create_xoroshiro128p_states(256, seed=1).shape)
import cupy; print("cupy", cupy.__version__, "devices:", cupy.cuda.runtime.getDeviceCount())
PY
}

nd_larnd_conda() # GPU
{
  nd_conda || return 1
  simulate_pixels.py --input_filename "${1}" --output_filename "${2}" \
    --config "${3:-module0}" --light_simulated False --rand_seed "${4:-1}"
}

nd_study() # GPU
{
  nd_conda || return 1
  [ -f "${ND_SRC}/tests/threshold_induction_study.py" ] || {
    echo "missing tests/threshold_induction_study.py in ${ND_SRC} -- push branch or scp it"; return 1; }
  ( cd "${ND_SRC}" && python tests/threshold_induction_study.py "$@" )
}

nd_study_batch()  # nd_study_batch [-d] [-o out] [-e edep.h5] [-n n] [-i ind] [-t hrs] [-q qos] [extra py flags...]
{
  local out=$ND_WORK/tistudy edep=$ND_WORK/shower.EDEPSIM.h5
  local nev=1000 indev=200 hrs=12 qos=shared dry=0 extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -o|--outdir)            out=$2;   shift 2 ;;
      -e|--edep)              edep=$2;  shift 2 ;;
      -n|--n-events)          nev=$2;   shift 2 ;;
      -i|--induction-events)  indev=$2; shift 2 ;;
      -t|--hours)             hrs=$2;   shift 2 ;;
      -q|--qos)               qos=$2;   shift 2 ;;
      -d|--dry-run)           dry=1;    shift   ;;   # write the script, do NOT submit
      *)                      extra+=("$1"); shift ;;
    esac
  done
  local job=$ND_WORK/study_job.sh
  cat > "$job" <<EOF
#!/bin/bash -l
#SBATCH -A $ND_ACCT_GPU
#SBATCH -C gpu
#SBATCH -q $qos
#SBATCH --gpus 1 -c 128 -N 1
#SBATCH -t ${hrs}:00:00
#SBATCH -o $ND_WORK/study_%j.log
source ~/dune_sim.sh
nd_conda
cd "$ND_SRC"
python tests/threshold_induction_study.py --edep-h5 "$edep" --outdir "$out" --n-events $nev --induction-events $indev ${extra[*]}
EOF
  if [ "$dry" = 1 ]; then
    echo "== DRY RUN: wrote $job (not submitted) =="
    grep -E '^#SBATCH -t|^#SBATCH -q|^python' "$job"
    return 0
  fi
  sbatch "$job"
}
