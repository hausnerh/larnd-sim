#!/bin/bash
# =============================================================================
# run_fsdcube_induction.sh  --  orchestrate the FSD Cube induction study
#
# Three stages, submitted as dependent Slurm jobs (source ~/dune_sim.sh first):
#   1. CPU : generate REAL edep-sim electron showers in the fsd_cube GDML
#   2. GPU : shower MULTIPLICITY GRID   (--mult-grid) on those showers   [afterok:1]
#   3. GPU : muon BURST + full 1-pitch induction waveforms (--burst-only --wf-txt)
#
# Usage:   source ~/dune_sim.sh    # or it is sourced below
#          ./run_fsdcube_induction.sh            # submit all three
#          ./run_fsdcube_induction.sh --dry-run  # write the job scripts, don't sbatch
#
# Built on the nd_* helpers in dune_sim.sh (ND_WORK, ND_SRC, ND_ACCT_*, nd_conda,
# nd_gun/nd_edep/nd_convert, incont). Edit the CONFIG block, verify the fsd_cube
# GEOMETRY block (the one thing dune_sim.sh hard-codes to module0), then run.
# =============================================================================
set -euo pipefail

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
[ -n "${ND_WORK:-}" ] || source ~/dune_sim.sh          # get the env + helpers

# ------------------------------- CONFIG --------------------------------------
TAG=fsdcube_induction
NSHOWER=1000                                            # showers for the multiplicity grid
ENERGY=300                                              # e- gun energy (MeV); mono, per dune_sim.sh
THRESHOLDS="2500 3750 5000 6250 7500"                   # e-   (2.5 -> 7.5 ke- in 1.25 steps)
RESETS="-1 1024 512 256 128 64"                         # PERIODIC_RESET_CYCLES (-1 = off)
MUON_EVENTS=250                                         # through-going muons for the burst run
BURST_EVENTS=80                                         # muons whose neighbour waveforms are captured
WFTXT=25                                                # full-resolution 1-pitch induction waveforms -> text
GRID_HRS=12; MUON_HRS=2; GEN_HRS=2                       # wall-clock per stage
QOS=shared                                              # -c 128 already => ~224 GB, fits 1000 showers
EDEP=$ND_WORK/${TAG}.EDEPSIM.h5                          # generated in stage 1, consumed in stage 2
OUT_GRID=$ND_WORK/tistudy_mult_grid
OUT_MUON=$ND_WORK/tistudy_muon_burst

# --------------------- fsd_cube GEOMETRY (VERIFY THESE) ----------------------
# dune_sim.sh points ND_GEOM/ND_VERTEX/ND_SD at module0. edep MUST run in the
# fsd_cube geometry or the showers land in the wrong volume. Set these to the
# fsd_cube values; leave VERTEX=AUTO to derive the active-volume centre with ROOT.
FSDCUBE_GDML="${FSDCUBE_GDML:-$(ls "$ND_WORK"/ND_Production/geometry/*fsd*cube*.gdml \
                                  "$ND_WORK"/ND_Production/geometry/*fsd*.gdml 2>/dev/null | head -1 || true)}"
FSDCUBE_VERTEX="${FSDCUBE_VERTEX:-AUTO}"                 # "x y z" in cm, or AUTO
FSDCUBE_SD="${FSDCUBE_SD:-TPCActive_shape}"             # sensitive-detector name in the GDML

# ------------------------------ PREFLIGHT ------------------------------------
echo "== preflight =="
if [ -z "$FSDCUBE_GDML" ] || [ ! -f "$FSDCUBE_GDML" ]; then
  echo "!! No fsd_cube GDML found under $ND_WORK/ND_Production/geometry/."
  echo "!! Set FSDCUBE_GDML=<path> and re-run. Look with:"
  echo "     ls \$ND_WORK/ND_Production/geometry/ | grep -i fsd"
  exit 1
fi
echo "   GDML : $FSDCUBE_GDML"
if [ "$FSDCUBE_VERTEX" = "AUTO" ]; then
  echo "   deriving active-volume centre with ROOT (in-container)..."
  FSDCUBE_VERTEX=$(incont "python3 - <<PY
import ROOT, array
g=ROOT.TGeoManager.Import('$FSDCUBE_GDML')
name=next(v.GetName() for v in g.GetListOfVolumes() if 'Active' in v.GetName())
def walk(node,hm):
    mm=ROOT.TGeoHMatrix(hm); mm.Multiply(node.GetMatrix())
    if node.GetVolume().GetName()==name: return mm
    for i in range(node.GetVolume().GetNdaughters()):
        r=walk(node.GetVolume().GetNode(i),mm)
        if r: return r
mm=walk(g.GetTopNode(),ROOT.TGeoHMatrix()); b=g.GetVolume(name).GetShape()
loc=array.array('d',[b.GetOrigin()[0],b.GetOrigin()[1],b.GetOrigin()[2]]); gl=array.array('d',[0,0,0]); mm.LocalToMaster(loc,gl)
print('%.4f %.4f %.4f'%(gl[0],gl[1],gl[2]))
PY" | tail -1)
fi
echo "   VERTEX (cm): $FSDCUBE_VERTEX     SD: $FSDCUBE_SD"
echo "   showers=$NSHOWER  thresholds=[$THRESHOLDS]  resets=[$RESETS]  wf-txt=$WFTXT"
echo "   >>> verify VERTEX sits inside the fsd_cube active volume before a real run <<<"

mkdir -p "$OUT_GRID" "$OUT_MUON"
J1=$ND_WORK/${TAG}_1_edep.sh
J2=$ND_WORK/${TAG}_2_grid.sh
J3=$ND_WORK/${TAG}_3_muon.sh

# ----------------------- stage 1: edep generation (CPU) ----------------------
cat > "$J1" <<EOF
#!/bin/bash -l
#SBATCH -A $ND_ACCT_CPU
#SBATCH -C cpu
#SBATCH -q $QOS
#SBATCH -N 1 -t ${GEN_HRS}:00:00
#SBATCH -o $ND_WORK/${TAG}_1_edep_%j.log
source ~/dune_sim.sh
export ND_GEOM="$FSDCUBE_GDML"          # <- fsd_cube geometry (overrides module0)
export ND_VERTEX="$FSDCUBE_VERTEX"
export ND_SD="$FSDCUBE_SD"
cd "\$ND_WORK"
nd_gun     e- $ENERGY ${TAG}.mac
nd_edep    ${TAG}.mac ${TAG}.EDEPSIM.root $NSHOWER
nd_convert ${TAG}.EDEPSIM.root ${EDEP}
nd_verify  ${EDEP}
echo "STAGE1 DONE -> ${EDEP}"
EOF

# --------------------- stage 2: shower multiplicity grid (GPU) ---------------
cat > "$J2" <<EOF
#!/bin/bash -l
#SBATCH -A $ND_ACCT_GPU
#SBATCH -C gpu
#SBATCH -q $QOS
#SBATCH --gpus 1 -c 128 -N 1 -t ${GRID_HRS}:00:00
#SBATCH -o $ND_WORK/${TAG}_2_grid_%j.log
source ~/dune_sim.sh
nd_conda
cd "\$ND_SRC"
python tests/threshold_induction_study.py --config fsd_cube --mult-grid \\
  --edep-h5 "${EDEP}" --n-events $NSHOWER --outdir "$OUT_GRID" \\
  --thresholds $THRESHOLDS --resets $RESETS
echo "STAGE2 DONE -> $OUT_GRID/ti_mult_grid.npz"
EOF

# ------------- stage 3: muon burst + full induction waveforms (GPU) ----------
cat > "$J3" <<EOF
#!/bin/bash -l
#SBATCH -A $ND_ACCT_GPU
#SBATCH -C gpu
#SBATCH -q $QOS
#SBATCH --gpus 1 -c 128 -N 1 -t ${MUON_HRS}:00:00
#SBATCH -o $ND_WORK/${TAG}_3_muon_%j.log
source ~/dune_sim.sh
nd_conda
cd "\$ND_SRC"
python tests/threshold_induction_study.py --config fsd_cube --burst-only \\
  --burst-thetas 0 --muon-length 200 --muon-events $MUON_EVENTS \\
  --burst-events $BURST_EVENTS --burst-neighbors 3 --burst-inductions 0.5 1.0 1.5 \\
  --muon-scan-events 100 --muon-thresholds 2000 2500 3000 4000 5000 \\
  --wf-txt $WFTXT --outdir "$OUT_MUON"
echo "STAGE3 DONE -> $OUT_MUON (+ induction_waveforms/*.txt)"
EOF

if [ "$DRY" = 1 ]; then
  echo "== DRY RUN: wrote $J1, $J2, $J3 (not submitted) =="
  for j in "$J1" "$J2" "$J3"; do echo "--- $j ---"; grep -E '^#SBATCH -[Cqt]|^python|^nd_edep|^nd_convert' "$j"; done
  exit 0
fi

# ------------------------------- submit --------------------------------------
echo "== submitting =="
ID1=$(sbatch --parsable "$J1");                          echo "  stage1 edep (CPU)  : $ID1"
ID2=$(sbatch --parsable --dependency=afterok:$ID1 "$J2"); echo "  stage2 grid (GPU)  : $ID2  [after $ID1]"
ID3=$(sbatch --parsable "$J3");                          echo "  stage3 muon (GPU)  : $ID3  [independent]"
echo "done. watch:  squeue --me ; tail -f $ND_WORK/${TAG}_*_*.log"
