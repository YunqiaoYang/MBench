#!/usr/bin/env bash
# MBench end-to-end demo
#
# Runs every registered MBench metric on 4 demo samples per side (8 total),
# covering all four subsets (causal / environment / human / object) on both
# MBench-A and MBench-T.
#
# Each metric is invoked independently so that a failure in one (e.g. missing
# DA3 artifact, missing GPU model weight, no VLM judge configured) does not
# block the rest. Per-metric stdout is teed into demo/output/logs/<side>__<metric>.log
# and a final pass/fail/contract-skip summary is printed.
#
# Run from the repository root (the directory above demo/):
#   bash demo/run_demo.sh
#
# Optional: to also exercise the trigger-conditioned T-side metrics with a
# real VLM judge, export these BEFORE invoking the script:
#   export MBENCH_VLM_BASE_URL=...
#   export MBENCH_VLM_API_KEY=...
#   export MBENCH_VLM_MODEL=...
# (Otherwise T-side trigger-gated metrics report raw-only scores; see the
# "Trigger gating" section in the top-level README.)

set -uo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Sanity check: heavy demo files (videos / DA3 npz) are gitignored, so a
# fresh public clone will only have JSON metadata. Bail out with a friendly
# message instead of producing 23 cryptic "missing media.videos" errors. ---
_first_a_video="demo/dataset/a/models/matrix_game_2/outputs/causal/a00110_01560/left_then_right_25s/video.mp4"
if [ ! -f "$_first_a_video" ]; then
  cat >&2 <<EOF
[demo] Heavy demo files are missing: $_first_a_video not found.
[demo] On a fresh git clone, demo/dataset/ ships only metadata (sample.json,
[demo] samples.jsonl, dataset.yaml) -- videos, ground-truth media, and DA3
[demo] artifacts are gitignored because they total >1 GB.
[demo]
[demo] Download the demo bundle from:
[demo]     https://drive.google.com/drive/folders/1Fpa21xpbfC8MN-GO86mMvJegGda8UDMK?usp=sharing
[demo] and unpack it on top of demo/dataset/ so each path lines up with the
[demo] tree shown in demo/README.md, then re-run this script.
EOF
  exit 2
fi

# Force in-tree mbench (don't pick up a globally installed older copy)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
MBENCH=(python3 -m mbench.cli)

OUT_ROOT="demo/output"
LOG_DIR="$OUT_ROOT/logs"
rm -rf "$OUT_ROOT"
mkdir -p "$LOG_DIR"

# (side, dataset_dir, metric)
A_METRICS=(
  "mbencha.entity.human_identity_consistency"
  "mbencha.entity.human_appearance_consistency"
  "mbencha.entity.object_texture_consistency"
  "mbencha.entity.object_geometry_consistency"
  "mbencha.environment.spatial_epipolar"
  "mbencha.environment.spatial_reprojection"
  "mbencha.environment.rendering_style"
  "mbencha.environment.rendering_lighting"
  "mbencha.causal.camera_interaction"
  "mbencha.causal.prompt_interaction"
  "mbencha.causal.state_progress"
  "mbencha.causal.progress_correctness"
)
T_METRICS=(
  "mbencht.entity.human_identity_consistency"
  "mbencht.entity.human_appearance_consistency"
  "mbencht.entity.object_texture_consistency"
  "mbencht.entity.object_geometry_consistency"
  "mbencht.environment.spatial_epipolar"
  "mbencht.environment.spatial_reprojection"
  "mbencht.environment.rendering_style"
  "mbencht.environment.rendering_lighting"
  "mbencht.causal.prompt_interaction"
  "mbencht.causal.state_progress"
  "mbencht.causal.progress_correctness"
)

PASS=()
FAIL=()
SKIP=()

run_metric() {
  local side="$1" data_dir="$2" metric="$3"
  local out_dir="$OUT_ROOT/$side/$metric"
  local log_file="$LOG_DIR/${side}__${metric}.log"

  printf "[demo] %s / %-50s " "$side" "$metric"
  if "${MBENCH[@]}" eval "$data_dir" \
       --metrics "$metric" \
       --output "$out_dir" \
       --quiet --no-progress-bar > "$log_file" 2>&1; then
    # Count valid items in items.jsonl. valid=False items also land there
    # (with valid:false), so an empty score does not mean a real result.
    local items_file="$out_dir/metrics/$metric/items.jsonl"
    local n_valid=0 n_total=0
    if [ -f "$items_file" ]; then
      n_total=$(wc -l < "$items_file" | tr -d ' \n')
      # grep -c returns 1 when there are zero matches, which triggers our
      # default; suppress that with `|| true` and pipe through tr to strip
      # any trailing whitespace.
      n_valid=$( { grep -c '"valid": true' "$items_file" || true; } | tr -d ' \n')
      [ -z "$n_valid" ] && n_valid=0
    fi
    if [ "$n_valid" -gt 0 ]; then
      printf "✓ pass  (%d/%d items valid)\n" "$n_valid" "$n_total"
      PASS+=("$side/$metric ($n_valid/$n_total)")
    else
      # Sniff for the most likely cause from the log
      local cause="no valid items"
      if grep -q "missing=artifacts.da3" "$log_file"; then cause="missing DA3 artifact"; fi
      if grep -q "no valid frame-pair scores" "$log_file"; then cause="SIFT/DA3: no valid frame-pair scores"; fi
      if grep -q "missing required keyword argument: judge" "$log_file"; then cause="VLM judge not configured (--vlm-judge)"; fi
      if grep -q "NVIDIA driver" "$log_file"; then cause="no GPU / CUDA driver"; fi
      if grep -q "No module named" "$log_file"; then cause="missing Python dep ($(grep -m1 "No module named" "$log_file" | sed -E "s/.*No module named '([^']+)'.*/\1/"))"; fi
      printf "— skip  (%s)\n" "$cause"
      SKIP+=("$side/$metric — $cause")
    fi
  else
    printf "✗ fail  (see %s)\n" "$log_file"
    FAIL+=("$side/$metric")
  fi
}

echo
echo "========================================================================"
echo " MBench-A (4 samples × ${#A_METRICS[@]} metrics)"
echo "========================================================================"
for m in "${A_METRICS[@]}"; do
  run_metric a demo/dataset/a "$m"
done

echo
echo "========================================================================"
echo " MBench-T (4 samples × ${#T_METRICS[@]} metrics)"
echo "========================================================================"
for m in "${T_METRICS[@]}"; do
  run_metric t demo/dataset/t "$m"
done

echo
echo "========================================================================"
echo " Summary"
echo "========================================================================"
echo "  Passed (≥1 valid item with real score): ${#PASS[@]}"
for x in "${PASS[@]}"; do echo "    ✓ $x"; done
echo "  Skipped (env/dep/data missing):         ${#SKIP[@]}"
for x in "${SKIP[@]}"; do echo "    — $x"; done
echo "  Failed (script/import error):           ${#FAIL[@]}"
for x in "${FAIL[@]}"; do echo "    ✗ $x"; done

echo
echo "  Per-metric outputs:    $OUT_ROOT/<side>/<metric>/"
echo "  Per-metric logs:       $LOG_DIR/<side>__<metric>.log"

if [ "${#FAIL[@]}" -gt 0 ]; then
  exit 1
fi
