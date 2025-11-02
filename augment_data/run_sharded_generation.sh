#!/usr/bin/env bash
# Orchestrate sharded TTS generation using generate_addess_minimax_selenium.py
# Steps:
#  1) Dry-run to compute target IDs and write ids_all.txt under parent voice_dir
#  2) Split ids_all.txt into N contiguous shards
#  3) Launch N parallel generation processes (one per shard) writing to voice_dir/shards/<shard_i>
#  4) Merge shard metadata and wavs back into parent voice_dir
#
# Requirements:
#  - Python scripts in the same repo:
#      * asr_ft/augment_data/generate_addess_minimax_selenium.py
#      * asr_ft/augment_data/split_ids_contiguous.py
#      * asr_ft/augment_data/merge_shards.py
#
# Usage example:
#  bash /home/nampv1/projects/asr/asr_ft/augment_data/run_sharded_generation.sh \
#    --tts_type clone \
#    --text_path /home/nampv1/projects/asr/asr_ft/augment_data/tests/texts_example.txt \
#    --output_dir /home/nampv1/projects/asr/asr_ft/augment_data/tests/test_output \
#    --provider minimax_selenium \
#    --delay 3.0 \
#    --batch_size 10 \
#    --num_shards 3

#
# Optional:
#  --voice_dir /path/to/output/<provider>/<model>/<voice>
#    If not provided, the script will try to auto-discover it after dry-run.
#
set -euo pipefail

# Defaults
TTS_TYPE="clone"
DELAY="3.0"
BATCH_SIZE="10"
NUM_SHARDS="2"
VOICE_DIR=""
PROVIDER="minimax_selenium"

# Paths (relative to repo root; adjust if needed)
GEN_SCRIPT="/home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py"
SPLIT_SCRIPT="/home/nampv1/projects/asr/asr_ft/augment_data/split_ids_contiguous.py"
MERGE_SCRIPT="/home/nampv1/projects/asr/asr_ft/augment_data/merge_shards.py"

print_usage() {
  cat <<EOF
Usage: $0 --text_path FILE --output_dir DIR [options]

Required:
  --text_path FILE         Input text file
  --output_dir DIR         Output directory root

Options:
  --tts_type {synthesize|clone}   TTS type (default: ${TTS_TYPE})
  --provider NAME                 Provider (default: ${PROVIDER})
  --delay FLOAT                   Delay between requests (default: ${DELAY})
  --batch_size INT                Batch size (default: ${BATCH_SIZE})
  --num_shards INT                Number of shards (default: ${NUM_SHARDS})
  --voice_dir DIR                 Parent voice_dir (optional; auto-detect if omitted)

Example:
  $0 --tts_type clone \
     --text_path /data/texts.txt \
     --output_dir /data/out \
     --provider minimax_selenium \
     --delay 3.0 \
     --batch_size 10 \
     --num_shards 3
EOF
}

# Parse args
TEXT_PATH=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tts_type) TTS_TYPE="$2"; shift 2;;
    --text_path) TEXT_PATH="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --provider) PROVIDER="$2"; shift 2;;
    --delay) DELAY="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --num_shards) NUM_SHARDS="$2"; shift 2;;
    --voice_dir) VOICE_DIR="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown option: $1"; print_usage; exit 1;;
  esac
done

if [[ -z "${TEXT_PATH}" || -z "${OUTPUT_DIR}" ]]; then
  echo "❌ Missing required --text_path or --output_dir"
  print_usage
  exit 1
fi

if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "❌ Generation script not found at ${GEN_SCRIPT}"; exit 1
fi
if [[ ! -f "${SPLIT_SCRIPT}" ]]; then
  echo "❌ Split script not found at ${SPLIT_SCRIPT}"; exit 1
fi
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "❌ Merge script not found at ${MERGE_SCRIPT}"; exit 1
fi

# 1) Dry-run to produce ids_all.txt under parent voice_dir
echo "🔎 Running dry-run to compute target IDs..."
python "${GEN_SCRIPT}" \
  --tts_type "${TTS_TYPE}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --provider "${PROVIDER}" \
  --delay "${DELAY}" \
  --batch_size "${BATCH_SIZE}" \
  --continue \
  --dry-run

# Auto-discover voice_dir if not provided
if [[ -z "${VOICE_DIR}" ]]; then
  echo "🧭 Auto-detecting voice_dir..."
  mapfile -t CANDIDATES < <(find "${OUTPUT_DIR}/${PROVIDER}" -type f -name ids_all.txt 2>/dev/null | sort -r)
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "❌ Could not find ids_all.txt under ${OUTPUT_DIR}/${PROVIDER}"; exit 1
  fi
  IDS_PATH="${CANDIDATES[0]}"
  VOICE_DIR="$(dirname "${IDS_PATH}")"
  echo "✅ Detected voice_dir: ${VOICE_DIR}"
else
  IDS_PATH="${VOICE_DIR}/ids_all.txt"
  if [[ ! -f "${IDS_PATH}" ]]; then
    echo "❌ ids_all.txt not found at ${IDS_PATH}"; exit 1
  fi
fi

# 2) Split into contiguous shards
echo "✂️  Splitting ids_all.txt into ${NUM_SHARDS} contiguous shard(s)..."
python "${SPLIT_SCRIPT}" \
  --ids_path "${IDS_PATH}" \
  --num_shards "${NUM_SHARDS}" \
  --output_dir "${VOICE_DIR}" \
  --output_prefix "ids_shard_"

# 3) Launch shards in parallel
pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  SHARD_FILE="${VOICE_DIR}/ids_shard_${i}.txt"
  if [[ ! -f "${SHARD_FILE}" ]]; then
    echo "⚠️ Shard file missing: ${SHARD_FILE}, skipping"
    continue
  fi
  echo "🚀 Launching shard ${i}..."
  python "${GEN_SCRIPT}" \
    --tts_type "${TTS_TYPE}" \
    --text_path "${TEXT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --provider "${PROVIDER}" \
    --delay "${DELAY}" \
    --batch_size "${BATCH_SIZE}" \
    --no-continue \
    --text_id_list_path "${SHARD_FILE}" \
    --shard_name "shard_${i}" &
  pids+=("$!")
  # Small stagger to avoid starting all browsers at once
  sleep 60
done

# Wait for all shards
echo "⏳ Waiting for ${#pids[@]} shard(s) to finish..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

# 4) Merge back
echo "🔗 Merging shards back into parent voice_dir..."
python "${MERGE_SCRIPT}" \
  --voice_dir "${VOICE_DIR}" \
  --move_wav \
  --backup_parent_metadata \
  --remove_shard_dirs

echo "🎉 Done. Merged metadata and WAVs are under: ${VOICE_DIR}"



#  bash /home/nampv1/projects/asr/asr_ft/augment_data/run_sharded_generation.sh \
#    --tts_type clone \
#    --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/district_list_with_no_prefix.txt \
#    --output_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/district_list_with_no_prefix/ \
#    --provider minimax_selenium \
#    --delay 3.0 \
#    --batch_size 10 \
#    --num_shards 3

#  bash /home/nampv1/projects/asr/asr_ft/augment_data/run_sharded_generation.sh \
#    --tts_type clone \
#    --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/cdp_list_with_prefix.txt \
#    --output_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/cdp_list_with_prefix/ \
#    --provider minimax_selenium \
#    --delay 3.0 \
#    --batch_size 10 \
#    --num_shards 4

#  bash /home/nampv1/projects/asr/asr_ft/augment_data/run_sharded_generation.sh \
#    --tts_type clone \
#    --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/district_list_with_prefix.txt \
#    --output_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/district_list_with_prefix/ \
#    --provider minimax_selenium \
#    --delay 3.0 \
#    --batch_size 10 \
#    --num_shards 1

#  bash /home/nampv1/projects/asr/asr_ft/augment_data/run_sharded_generation.sh \
#    --tts_type clone \
#    --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/commune_list_with_prefix.txt \
#    --output_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/commune_list_with_prefix/ \
#    --provider minimax_selenium \
#    --delay 1.0 \
#    --batch_size 10 \
#    --num_shards 1

#  bash /home/nampv1/projects/asr/asr_ft/augment_data/run_sharded_generation.sh \
#    --tts_type clone \
#    --text_path /home/nampv1/projects/asr/asr_ft/augment_data/tests/texts_example.txt \
#    --output_dir /home/nampv1/projects/asr/asr_ft/augment_data/tests/test_output \
#    --provider minimax_selenium \
#    --delay 3.0 \
#    --batch_size 10 \
#    --num_shards 4
