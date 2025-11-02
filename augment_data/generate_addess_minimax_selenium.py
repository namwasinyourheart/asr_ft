#!/usr/bin/env python3
# ============================================================
# Generate Address Audio Dataset with ID System
# ============================================================

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict

# Add speech-synth-engine to path
sys.path.insert(0, "/home/nampv1/projects/tts/speech-synth-engine")

from speech_synth_engine.dataset.dataset_generator import DatasetGenerator

from dotenv import load_dotenv
load_dotenv("/home/nampv1/projects/asr/asr_ft/augment_data/.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetGenerator")
logger.setLevel(logging.INFO)

from speech_synth_engine.dataset.text_loaders import TextFileLoader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate audio dataset from text file using TTS providers'
    )

    parser.add_argument(
        '--text_path',
        type=str,
        required=True,
        help='Path to text file to generate audio from'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for generated audio files'
    )

    parser.add_argument(
        '--provider',
        type=str,
        default='gtts',
        choices=['gtts', 'gemini', 'vnpost', 'minimax_selenium'],
        help='TTS provider to use (default: gtts)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=3.0,
        help='Delay between requests in seconds (default: 3.0)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )

    parser.add_argument(
        '--start_text_id',
        type=int,
        default=None,
        help='Start text_id (inclusive) to generate a specific range'
    )

    parser.add_argument(
        '--end_text_id',
        type=int,
        default=None,
        help='End text_id (inclusive) to generate a specific range'
    )

    parser.add_argument(
        '--text_id_list_path',
        type=str,
        default=None,
        help='Path to text_id list file to generate specific text_ids'
    )

    # Auto-continue flags (default True). Use dest to avoid using reserved word 'continue' as attribute.
    parser.add_argument(
        '--continue',
        dest='auto_continue',
        action='store_true',
        default=True,
        help='Automatically continue: detect and regenerate missing items (default: True)'
    )
    parser.add_argument(
        '--no-continue',
        dest='auto_continue',
        action='store_false',
        help='Disable auto-continue behavior (process as specified without missing check)'
    )

    # Dry-run mode: only compute final text_id list and save to ids_all.txt, no generation
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Dry run: compute target text_ids after filtering and write to ids_all.txt; do not generate audio'
    )

    # Shard name: write outputs under voice_dir/shards/<shard_name>
    parser.add_argument(
        '--shard_name',
        type=str,
        default=None,
        help='Optional shard name; outputs will be placed under voice_dir/shards/<shard_name>'
    )

    parser.add_argument(
        '--tts_type',
        type=str,
        default='synthesize',
        choices=['synthesize', 'clone'],
        help='TTS type to use (default: synthesize)'
    )


    return parser.parse_args()


import pandas as pd

def read_metadata_file(metadata_path: Path) -> Dict[int, str]:
    """Read metadata file and return a dict of processed text_ids -> text"""
    processed: Dict[int, str] = {}
    if metadata_path.exists():
        try:
            df = pd.read_csv(
                metadata_path,
                sep="\t",
                header=0,
                names=["utt_id", "text_id", "text", "audio_path", "provider", "model", "voice", "tts_type", "sample_rate", "lang", "duration", "gen_date"],
                index_col=False,
            )
            for _, row in df.iterrows():
                text_id = int(row['text_id'])
                processed[text_id] = row['text']
        except Exception as e:
            logger.warning(f"Could not read metadata file: {e}")
    return processed


def analyze_metadata_wav_consistency(metadata_path: Path, text_path: Path, wav_dir: Path, loader: TextFileLoader):
    """Compare text_items, metadata.tsv, and actual .wav files.

    Returns:
        results: dict containing sets of IDs (as ints) and their lengths
        report: pandas DataFrame describing the detailed state of each text_id
    """
    # Read metadata if exists
    try:
        if metadata_path.exists():
            df = pd.read_csv(
                metadata_path,
                sep="\t",
                header=0,
                names=["utt_id", "text_id", "text", "audio_path", "provider", "model", "voice", "tts_type", "sample_rate", "lang", "duration", "gen_date"],
                index_col=False,
            )
        else:
            df = pd.DataFrame(columns=["utt_id", "text_id", "text", "audio_path", "provider", "model", "voice", "tts_type", "sample_rate", "lang", "duration", "gen_date"])
    except Exception as e:
        logger.warning(f"Could not read metadata for consistency check: {e}")
        df = pd.DataFrame(columns=["utt_id", "text_id", "text", "audio_path", "provider", "model", "voice", "tts_type", "sample_rate", "lang", "duration", "gen_date"])

    # Load text items
    text_items = loader.load(str(text_path))

    # List wav files (if dir exists)
    wav_filenames = []
    if wav_dir.exists():
        try:
            wav_filenames = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        except Exception as e:
            logger.warning(f"Could not list wav dir '{wav_dir}': {e}")

    # Sets of IDs
    text_ids = {int(tid) for tid, *_ in text_items}
    metadata_ids = set()
    try:
        if 'text_id' in df.columns and not df.empty:
            metadata_ids = set(df['text_id'].astype(int).tolist())
    except Exception as e:
        logger.warning(f"Could not parse text_id from metadata: {e}")

    def extract_id_from_filename(filename: str) -> int:
        base = os.path.splitext(os.path.basename(filename))[0]
        return int(base.split('_')[0])

    wav_ids = set()
    for fn in wav_filenames:
        try:
            wav_ids.add(extract_id_from_filename(fn))
        except Exception:
            continue

    # Comparisons
    in_both = text_ids & metadata_ids & wav_ids
    in_metadata_not_wav = (text_ids & metadata_ids) - wav_ids
    in_wav_not_metadata = (text_ids & wav_ids) - metadata_ids
    in_neither = text_ids - (metadata_ids | wav_ids)

    # Build report
    import pandas as _pd
    ids_sorted = sorted(text_ids)
    report = _pd.DataFrame({
        'text_id': ids_sorted,
        'in_metadata': [i in metadata_ids for i in ids_sorted],
        'in_wav': [i in wav_ids for i in ids_sorted],
    })
    def _status(row):
        if row['in_metadata'] and row['in_wav']:
            return 'both'
        if row['in_metadata']:
            return 'metadata_only'
        if row['in_wav']:
            return 'wav_only'
        return 'neither'
    report['status'] = report.apply(_status, axis=1)

    results = {
        'in_both': sorted(in_both),
        'missing_in_wav': sorted(in_metadata_not_wav),
        'missing_in_metadata': sorted(in_wav_not_metadata),
        'missing_in_both': sorted(in_neither),
        'len': {
            'in_both': len(in_both),
            'missing_in_wav': len(in_metadata_not_wav),
            'missing_in_metadata': len(in_wav_not_metadata),
            'missing_in_both': len(in_neither),
        }
    }

    logger.info(f"✅ Có trong cả metadata & wav_dir: {results['len']['in_both']} item(s)")
    logger.info(f"⚠️ Có trong metadata nhưng KHÔNG có file wav: {results['len']['missing_in_wav']} item(s)")
    logger.info(f"⚠️ Có file wav nhưng KHÔNG có trong metadata: {results['len']['missing_in_metadata']} item(s)")
    logger.info(f"🚫 Không có trong metadata và cũng KHÔNG có file wav: {results['len']['missing_in_both']} item(s)")

    return results, report



def get_providers_config(provider: str) -> dict:
    """Get providers configuration based on provider type"""
    configs = {
        'gtts': {
            "gtts": {"language": "vi"}
        },
        'gemini': {
            "gemini": {
                "sample_rate": 24000,
                "model": "gemini-2.5-flash-preview-tts",
                "language": "vi"
            }
        },
        'vnpost': {
            "vnpost": {
                "sample_rate": 22050,
                "language": "vi",
                "voice": "Hà My"
            }
        },
        'minimax_selenium': {
            "minimax_selenium": {
                "sample_rate": 24000,
                "google_email": os.getenv("MINIMAX_GOOGLE_EMAIL"),
                "google_password": os.getenv("MINIMAX_GOOGLE_PASSWORD"),    
                # "headless": False,
                "browser": "chrome"  
            }
        },

    }

    return configs.get(provider, configs['gtts'])


def get_provider_model_voice(provider: str) -> list:
    """Get provider model voice based on provider type"""
    configs = {
        'gtts': ("gtts", "default", "vi"),
        'gemini': ("gemini", "gemini-2.5-flash-preview-tts", "vi"),
        'vnpost': ("vnpost", "default", "Hà My"),
        'minimax_selenium': ("minimax_selenium", "default", "bdtl_spk2")
    }

    return configs.get(provider, configs['gtts'])


def main():
    """Main function"""
    args = parse_arguments()

    # Validate input file exists
    text_path = Path(args.text_path)
    if not text_path.exists():
        logger.error(f"❌ Text file not found: {text_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Starting generation...")
    logger.info(f"   📄 Text file: {text_path}")
    logger.info(f"   📁 Output dir: {output_dir}")
    logger.info(f"   🎵 Provider: {args.provider}")
    logger.info(f"   ⏱️ Delay: {args.delay}s")
    logger.info(f"   📦 Batch size: {args.batch_size}")
    if args.start_text_id is not None or args.end_text_id is not None:
        logger.info(f"   🔢 Range args: start_text_id={args.start_text_id}, end_text_id={args.end_text_id}")

    try:

        # Get provider configuration
        providers_config = get_providers_config(args.provider)
        provider_model_voice = get_provider_model_voice(args.provider)


        # Load text items
        logger.info("📖 Loading text items...")
        loader = TextFileLoader()
        text_items = loader.load(args.text_path)

        if not text_items:
            logger.error("❌ No text items found in file")
            return 1

        # Create generator and generate
        generator = DatasetGenerator(output_dir, providers_config)

        # Save text items
        try:
            voice_dir = output_dir / args.provider / provider_model_voice[1] / provider_model_voice[2]

            output_path = voice_dir / "text_items.txt"
            generator._save_text_items(text_items, output_path)
        except Exception as e:
            generator.logger.warning(f"⚠️ Could not save text items: {e}")


        # Paths and existing metadata (parent voice_dir without shard)
        voice_dir = output_dir / args.provider / provider_model_voice[1] / provider_model_voice[2]
        metadata_path = voice_dir / "metadata.tsv"
        wav_dir = voice_dir / "wav"

        # Validate exclusivity of modes and apply filtering based on precedence
        if (args.start_text_id is None) ^ (args.end_text_id is None):
            logger.error("❌ You must provide both --start_text_id and --end_text_id, or neither.")
            return 1

        has_list = args.text_id_list_path is not None
        has_range = args.start_text_id is not None and args.end_text_id is not None

        if sum([has_list, has_range, True if args.auto_continue else False]) > 1 and has_list:
            logger.info("ℹ️ Using --text_id_list_path; ignoring range and auto-continue if provided.")
        elif sum([has_list, has_range, True if args.auto_continue else False]) > 1 and has_range:
            logger.info("ℹ️ Using range [start_text_id, end_text_id]; ignoring auto-continue.")

        original_count = len(text_items)

        # Mode 1: explicit list of IDs
        if has_list:
            # Read IDs from file (supports comma/space/newline separated digits)
            import re
            target_ids = set()
            try:
                with open(args.text_id_list_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        for token in re.findall(r"\d+", line):
                            target_ids.add(int(token))
            except Exception as e:
                logger.error(f"❌ Could not read text_id_list_path: {e}")
                return 1
            before = len(text_items)
            text_items = [item for item in text_items if int(item[0]) in target_ids]
            logger.info(f"📝 List filter applied: {len(text_items)} items (from {before})")

        # Mode 2: range of IDs
        elif has_range:
            start_id = args.start_text_id
            end_id = args.end_text_id
            if start_id > end_id:
                logger.error("❌ --start_text_id must be less than or equal to --end_text_id")
                return 1
            text_items = [item for item in text_items if start_id <= int(item[0]) <= end_id]
            logger.info(f"📐 Range filter applied: [{start_id}, {end_id}] -> {len(text_items)} items (from {original_count})")

            # Within range, avoid regenerating items already fully present (both metadata and wav)
            results, _ = analyze_metadata_wav_consistency(metadata_path, text_path, wav_dir, loader)
            allowed_ids = {int(item[0]) for item in text_items}
            missing_ids = set(results['missing_in_both']) | set(results['missing_in_wav'])
            target_ids = allowed_ids & missing_ids
            before = len(text_items)
            text_items = [item for item in text_items if int(item[0]) in target_ids]
            skipped = before - len(text_items)
            if skipped > 0:
                logger.info(f"⏭️  Skipping {skipped} fully-present items within range")

        # Mode 3: auto-continue (default True)
        elif args.auto_continue:
            results, report = analyze_metadata_wav_consistency(metadata_path, text_path, wav_dir, loader)
            missing_ids = set(results['missing_in_both']) | set(results['missing_in_wav'])
            before = len(text_items)
            text_items = [item for item in text_items if int(item[0]) in missing_ids]
            logger.info(f"🔄 Auto-continue selected missing IDs: {len(text_items)} items (from {before})")
            # Save report for reference
            try:
                voice_dir.mkdir(parents=True, exist_ok=True)
                report.to_csv(voice_dir / 'consistency_report.csv', index=False)
            except Exception as e:
                logger.warning(f"Could not save consistency report: {e}")

        # Mode 4: no-continue, process as-is
        else:
            logger.info("⚠️ Auto-continue disabled; proceeding without missing check (may regenerate existing items).")

        # If dry-run: write the computed ids and exit
        if args.dry_run:
            try:
                voice_dir.mkdir(parents=True, exist_ok=True)
                ids_out = voice_dir / 'ids_all.txt'
                with open(ids_out, 'w', encoding='utf-8') as f:
                    for item in text_items:
                        f.write(f"{int(item[0])}\n")
                logger.info(f"📝 Dry-run: wrote {len(text_items)} text_id(s) to {ids_out}")
            except Exception as e:
                logger.error(f"❌ Dry-run failed to write ids_all.txt: {e}")
                return 1
            return 0

        if len(text_items) == 0:
            logger.info("✅ Nothing to process after filtering (range/default + metadata).")
            return 0

        logger.info(f"✅ Loaded {len(text_items)} text items for generation")


        reference_audio = Path("/media/nampv1/hdd/data/mẫu-giọng-nhân-viên-nhập-liệu-bưu-cục-thăng-long-24-10-20251024T103708Z-1-001/mẫu-giọng-nhân-viên-nhập-liệu-bưu-cục-thăng-long-24-10/spk2_1.m4a")
        voice = "bdtl_spk2"

        logger.info("🎵 Generating audio files...")

        # Apply shard subdirectory at generation time (keep parent voice_dir for planning files)
        gen_provider_model_voice = provider_model_voice
        if args.shard_name:
            gen_provider_model_voice = (
                provider_model_voice[0],
                provider_model_voice[1],
                f"{provider_model_voice[2]}/shards/{args.shard_name}"
            )

        summary = generator.generate_from_text_list(
            tts_type=args.tts_type,
            text_items=text_items,
            provider_model_voice=gen_provider_model_voice,
            reference_audio=reference_audio,
            delay_between_requests=args.delay,
            batch_size=args.batch_size
        )

        # Report results
        logger.info("📊 Generation Summary:")
        logger.info(f"   ✅ Successful: {summary.successful_generations}")
        logger.info(f"   ⏭️ Skipped duplicates: {len([r for r in summary.results if r.skipped_duplicate])}")
        logger.info(f"   ❌ Failed: {summary.failed_generations}")
        logger.info(f"   📁 Output directory: {output_dir}")

        if summary.successful_generations > 0:
            # Show sample result
            first_result = summary.results[0]
            logger.info(f"🎵 Sample audio file: {first_result.audio_path}")
            logger.info(f"📋 Sample metadata file: {first_result.metadata_path}")

        logger.info("🎉 Generation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Error during generation: {e}")
        return 1



if __name__ == "__main__":
    main()


# Usage Example

# python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py \
# --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/commune_list_with_no_prefix.txt \
# --output_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/commune_list_with_no_prefix/ \
# --provider minimax_selenium \
# --delay 3.0 \
# --batch_size 10 \
# --start_text_id 4800 \
# --end_text_id 4899

# python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py \
# --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/district_list_with_no_prefix.txt \
# --output_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/district_list_with_no_prefix/ \
# --provider minimax_selenium \
# --delay 3.0 \
# --batch_size 10 \
# --start_text_id 150 \
# --end_text_id 185


# python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py \
# --text_path /home/nampv1/projects/asr/asr_ft/augment_data/tests/texts_example.txt \
# --output_dir /home/nampv1/projects/asr/asr_ft/augment_data/tests/test_output \
# --provider minimax_selenium \
# --delay 3.0 \
# --batch_size 10 \
# --continue

# python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py \
# --tts_type clone \
# --text_path /home/nampv1/projects/asr/asr_ft/augment_data/tests/texts_example.txt \
# --output_dir /home/nampv1/projects/asr/asr_ft/augment_data/tests/test_output \
# --provider minimax_selenium \
# --delay 3.0 \
# --batch_size 10 \
# --continue


# python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py \
# --tts_type clone \
# --text_path /home/nampv1/projects/asr/asr_ft/augment_data/tests/texts_example.txt \
# --output_dir /home/nampv1/projects/asr/asr_ft/augment_data/tests/test_output \
# --provider minimax_selenium \
# --delay 3.0 \
# --batch_size 10 \
# --continue \
# --dry-run