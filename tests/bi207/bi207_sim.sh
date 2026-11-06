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
