#!/usr/bin/env bash
#
# run_tricell_scans.sh
# ====================
# Run the tricell ds-accuracy study in a staged sequence, each stage
# writing to its own output directory so the plots/npz/log can be
# reviewed side by side. Every stage adds ONE more witness-quality cut
# than the previous, so each cut's effect on bias, RMS, and yield is
# individually attributable.
#
# Run from the repo ROOT on the GPU node (jupyter-hhausner):
#
#     bash tests/run_tricell_scans.sh
#
# Outputs land under  tricell_runs/<timestamp>/<stage_name>/  by
# default. Each stage dir holds the PNGs, tricell_ds_accuracy_results.npz,
# and run.log (full console output, including the per-cut rejection
# counts printed at the end of each run).
#
# Knobs (environment variables):
#   PYTHON            python executable           (default: python)
#   TRICELL_OUT_ROOT  base output directory       (default: timestamped)
#   TRICELL_COMMON    extra args appended to EVERY stage, e.g.
#                     TRICELL_COMMON="--n-tracks 600" for more yield,
#                     or "--n-tracks 10 --track-lengths-cm 5" for a
#                     quick smoke test.
#
# A stage that fails does NOT abort the rest — its failure is recorded
# and the script moves on, so an overnight sweep always completes.

set -uo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="tests/test_tricell_ds_accuracy.py"
OUT_ROOT="${TRICELL_OUT_ROOT:-tricell_runs/$(date +%Y%m%d_%H%M%S)}"
# shellcheck disable=SC2206
COMMON_ARGS=(${TRICELL_COMMON:-})

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: ${SCRIPT} not found. Run from the repo root." >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "Python     : ${PYTHON}"
echo "Common args: ${COMMON_ARGS[*]:-(none)}"

SUMMARY=()

run_stage () {
    local name="$1"; shift
    local outdir="${OUT_ROOT}/${name}"
    mkdir -p "${outdir}"
    echo
    echo "=============================================================="
    echo "STAGE ${name}"
    echo "  outdir: ${outdir}"
    echo "  args  : $* ${COMMON_ARGS[*]:-}"
    echo "=============================================================="
    "${PYTHON}" "${SCRIPT}" --outdir "${outdir}" \
        "$@" "${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}" 2>&1 \
        | tee "${outdir}/run.log"
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -eq 0 ]]; then
        SUMMARY+=("OK    ${name}")
    else
        SUMMARY+=("FAIL  ${name} (exit ${status})")
        echo "STAGE ${name} FAILED (exit ${status}) — continuing." >&2
    fi
}

# 1. Baseline: Δt units fix only, centroid timing, all witness cuts OFF.
#    This is the reference the staged cuts are measured against.
run_stage 00_baseline_centroid

# 2. Same baseline with peak_rate timing — the estimator comparison.
run_stage 01_peak_rate \
    --primary-timing-method peak_rate

# 3. CUT 1 — halo-vs-track: require >= 2 FEE packets per witness.
run_stage 02_cut1_packets \
    --minimum-witness-packets 2

# 4. CUT 1 + CUT 2 — add the witness-vs-w1 direction cross-check (12 deg).
run_stage 03_cut12_direction \
    --minimum-witness-packets 2 \
    --maximum-direction-disagreement-deg 12

# 5. CUT 1 + 2 + 3 — add the physical-plausibility bound on the
#    reconstructed direction (drift window + zenith band).
run_stage 04_cut123_plausibility \
    --minimum-witness-packets 2 \
    --maximum-direction-disagreement-deg 12 \
    --maximum-implied-drift-cm 30 \
    --implied-zenith-min-deg 5 \
    --implied-zenith-max-deg 85

echo
echo "=============================================================="
echo "ALL STAGES COMPLETE — ${OUT_ROOT}"
for line in "${SUMMARY[@]}"; do
    echo "  ${line}"
done
echo "=============================================================="
ls -1 "${OUT_ROOT}"
