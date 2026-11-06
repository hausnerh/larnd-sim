# tests/bi207/bi207_sim.sh  --  Bi-207 calibration-source sim helpers for the FSD Cube.
#
# Source this ALONGSIDE ~/dune_sim.sh on NERSC (reuses its env + `incont`, `nd_convert`,
# `nd_verify`):
#     source ~/dune_sim.sh
#     source $ND_SRC/tests/bi207/bi207_sim.sh
#
# ACCURATE DECAY -- the real thing, via Geant4's G4RadioactiveDecay:
# The sim2x2 edep-sim DEFAULT physics list (QGSP_BERT) has NO G4RadioactiveDecay, so
# `/gps/ion 83 207` never decays (1 trajectory, 0 deposits). Setting PHYSLIST=Shielding
# switches edep-sim to the Shielding reference list, which registers G4RadioactiveDecay
# (with internal conversion + atomic de-excitation + Auger). Then `/gps/ion 83 207` produces
# the full Bi-207 EC cascade -- conversion electrons (481.7/975.7/1047.8 keV calibration
# lines), Auger, and the 570/1064/1770 keV gammas with angular correlations -- all at t=0,
# transported in LAr. CONFIRMED working (20 decays -> ~40 trajectories/decay, deposits, t=0).
# No BxDecay0 needed (it's on conda-forge but adds a primaries-plumbing step we don't need).
#
# STAGE 1 (this file): generate + validate (check_bi207.py). We generate in the module0 GDML
# the showers use (a big LAr medium: a <=1.7 MeV e- ranges out in <1 cm, fully contained
# anywhere) at an interior point. STAGE 2 translates the source point onto the FSD-cube cathode
# and runs simulate_pixels(fsd_cube)+ndlar_flow; STAGE 3 is the per-pixel Q/ADC analysis.
#
# STATS: only ~11% of decays yield a conversion electron (the clean calibration handle; the rest
# emit gammas -> Compton continuum). The lines emerge in the per-pixel DEPOSITED-Q spectrum with
# enough decays -- plan for 1e4-1e5, not 1e3, especially per-pixel.

# Write the Bi-207 radioactive-ion macro. Args: [x y z cm] [macro]; defaults to an interior
# point of volTPCActive (ND_VERTEX) and ./bi207.mac. SD/segmentation match nd_gun.
bi207_gun()
{
  local pos="${ND_VERTEX}" mac=bi207.mac
  if [ $# -ge 3 ]; then pos="$1 $2 $3"; shift 3; fi
  [ -n "$1" ] && mac=$1
  cat > "${mac}" <<EOF
/edep/hitSeparation ${ND_SD} -1 mm
/edep/hitSagitta    ${ND_SD} 1.0 mm
/edep/hitLength     ${ND_SD} 1.0 mm
/edep/update
# Bi-207 ion at rest; G4RadioactiveDecay (requires PHYSLIST=Shielding) emits the full EC cascade
/gps/particle ion
/gps/ion 83 207 0 0
/gps/energy 0 MeV
/gps/position ${pos} cm
/gps/ang/type iso
EOF
  echo "wrote ${mac}  (Bi-207 ion @ (${pos}) cm, iso, SD=${ND_SD})"
}

# edep-sim with the Shielding physics list (so RadioactiveDecay is active). Args: <mac> <out.root> <N>.
# PHYSLIST is exported INSIDE the container so it reaches edep-sim regardless of shifter env passing.
bi207_edep()
{
  incont "export PHYSLIST=Shielding; edep-sim -C -g '${ND_GEOM}' -o '${2}' -e ${3:-1000} '${1}'"
}

# Generate + convert + verify. Args: [nEvents] [tag].  Each edep event = one Bi-207 decay.
bi207_gen()
{
  local n=${1:-1000} tag=${2:-bi207}
  cd "${ND_WORK}" || return 1
  bi207_gun "${tag}.mac"                                 || return 2
  bi207_edep "${tag}.mac" "${tag}.EDEPSIM.root" "${n}"   || return 3
  nd_convert "${tag}.EDEPSIM.root" "${tag}.EDEPSIM.h5"   || return 4
  nd_verify  "${tag}.EDEPSIM.h5"
  echo "DONE -> ${ND_WORK}/${tag}.EDEPSIM.h5"
  echo "VALIDATE (plot shows the e- lines emerging with stats):"
  echo "  python ${ND_SRC}/tests/bi207/check_bi207.py ${ND_WORK}/${tag}.EDEPSIM.h5 --outdir ${ND_WORK}/bi207_check_${tag}"
}

# STAGE 2+3 one-shot: translate the Bi-207 edep onto the fsd_cube cathode, run the FEE sim
# (simulate_pixels, needs a GPU) and ndlar_flow (h5flow, CPU), leaving a FLOW file ready for
# check_bi207_fee.py. The h5flow leg mirrors ndlar_flow's test/end2end/run_fsdcube_mc.sh (the
# fsdcube MC workflow; our LARND is the same simulate_pixels(fsd_cube) output mkramer's showers use).
# Needs: a GPU node for simulate_pixels, and ND_FLOW = your ndlar_flow repo root (the dir with yamls/).
# Args: [tag] [inset_cm].  Default tag=bi207, inset=0.5 cm off the cathode INTO the LAr (so nearly all
# of the isotropic decay cloud is collected while still at ~full drift).
bi207_fsd()
{
  local tag=${1:-bi207} inset=${2:-0.5}
  local edep="${ND_WORK}/${tag}.EDEPSIM.h5"
  local fsd="${ND_WORK}/${tag}_fsd.EDEPSIM.h5"
  local larnd="${ND_WORK}/${tag}_fsd.LARND.h5"
  local flow="${ND_WORK}/${tag}_fsd.FLOW.hdf5"
  : "${ND_FLOW:?set ND_FLOW to your ndlar_flow repo root (the dir containing yamls/)}"
  [ -f "${edep}" ] || { echo "missing ${edep} (run bi207_gen ${tag} first)"; return 1; }

  echo ">>> [1/3] translate onto the fsd_cube cathode (inset ${inset} cm) -> ${fsd}"
  rm -f "${fsd}"
  python "${ND_SRC}/tests/bi207/bi207_to_fsd.py" "${edep}" "${fsd}" --inset "${inset}" || return 2

  echo ">>> [2/3] simulate_pixels (FEE, GPU) -> ${larnd}"
  rm -f "${larnd}"
  python "${ND_SRC}/cli/simulate_pixels.py" \
    --input_filename="${fsd}" --output_filename="${larnd}" \
    --config=fsd_cube --rand_seed=1 || return 3

  echo ">>> [3/3] ndlar_flow (h5flow, CPU) -> ${flow}"
  rm -f "${flow}"
  ( cd "${ND_FLOW}" && h5flow -i "${larnd}" -o "${flow}" -c \
      yamls/fsdcube_flow/workflows/charge/charge_event_building_mc.yaml \
      yamls/fsdcube_flow/workflows/charge/charge_event_reconstruction_mc.yaml \
      yamls/fsdcube_flow/workflows/combined/combined_reconstruction_mc.yaml \
      yamls/fsdcube_flow/workflows/charge/prompt_calibration_mc.yaml \
      yamls/fsdcube_flow/workflows/charge/filtered_calibration_mc.yaml ) || return 4

  echo "DONE -> ${flow}"
  echo "ANALYZE (FEE Q spectra vs predicted Bi-207 lines):"
  echo "  python ${ND_SRC}/tests/bi207/check_bi207_fee.py ${flow} --outdir ${ND_WORK}/${tag}_fee"
}
