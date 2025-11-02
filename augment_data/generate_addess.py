#!/usr/bin/env python3
# ============================================================
# Generate Address Audio Dataset with ID System
# ============================================================

import os
import sys
import logging
import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Add speech-synth-engine to path
sys.path.insert(0, "/home/nampv1/projects/tts/speech-synth-engine")

from speech_synth_engine.dataset.dataset_generator import DatasetGenerator

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
        choices=['gtts', 'gemini', 'vnpost'],
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
        '--continue_from_text_id',
        type=int,
        default=None,
        help='Continue from specific text_id'
    )

    parser.add_argument("--text_ids", type=str, default=None,
                    help="Danh sách text_id (phân tách bằng dấu phẩy) để sinh cụ thể.")
    parser.add_argument("--text_ids_file", type=str, default=None,
                    help="Đường dẫn đến file chứa danh sách text_id cần sinh (mỗi dòng một id).")

    return parser.parse_args()


import pandas as pd

def read_metadata_file(metadata_path: Path) -> Tuple[Dict[int, str], int]:
    """Read metadata file and return a dict of processed text_ids and the last text_id"""
    processed = {}
    last_text_id = 0
    
    if metadata_path.exists():
        try:
            df = pd.read_csv(
                metadata_path, 
                sep="\t", 
                header=0, 
                names=
                [
                    "utt_id", "text_id", "text", "audio_path", "provider", "model", 
                    "voice", "operation_type", "sample_rate", "lang", "duration", "gen_date"
                ]
            )
            for _, row in df.iterrows():
                text_id = int(row['text_id'])
                processed[text_id] = row['text']
                last_text_id = max(last_text_id, text_id)
        except Exception as e:
            logger.warning(f"Could not read metadata file: {e}")
    
    return processed, last_text_id


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
                "headless": False,
                "browser": "chrome"  
            }
        },

    }

    return configs.get(provider, configs['gtts'])


def get_provider_model_voice_list(provider: str) -> list:
    """Get provider model voice list based on provider type"""
    configs = {
        'gtts': [("gtts", "default", "vi")],
        'gemini': [("gemini", "gemini-2.5-flash-preview-tts", "vi")],
        'vnpost': [("vnpost", "default", "Hà My")],
        'minimax_selenium': [("minimax_selenium", "default", "")]
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

    try:

        # Get provider configuration
        providers_config = get_providers_config(args.provider)
        provider_model_voice_list = get_provider_model_voice_list(args.provider)

        
        # Load text items
        logger.info("📖 Loading text items...")
        loader = TextFileLoader()
        text_items = loader.load(args.text_path)

        # print("text_items", text_items)

        if not text_items:
            logger.error("❌ No text items found in file")
            return 1

            
        # # Check for existing metadata
        metadata_path = output_dir / args.provider / provider_model_voice_list[0][1] / provider_model_voice_list[0][2] / "metadata.tsv"
        # processed_texts, last_text_id = read_metadata_file(metadata_path)

        # print("len(processed_texts)", len(processed_texts))
        # print("last_text_id", last_text_id)
        
        # # Filter out already processed texts
        # original_count = len(text_items)
        # if processed_texts:
        #     text_items = [item for item in text_items 
        #                  if int(item[0]) > last_text_id]
        #     skipped = original_count - len(text_items)
        #     if skipped > 0:
        #         logger.info(f"⏭️  Skipping {skipped} already processed items")
        #         if len(text_items) == 0:
        #             logger.info("✅ All items already processed")
        #             return 0
        # logger.info(f"✅ Loaded {len(text_items)} text items")


        # 🧩 Lọc theo danh sách text_id cụ thể nếu có
        if args.text_ids or args.text_ids_file:
            if args.text_ids_file:
                with open(args.text_ids_file, "r", encoding="utf-8") as f:
                    target_ids = {int(line.strip()) for line in f if line.strip().isdigit()}
            else:
                target_ids = {int(x.strip()) for x in args.text_ids.split(",") if x.strip().isdigit()}
            
            logger.info(f"🎯 Chỉ sinh cho {len(target_ids)} text_id cụ thể.")

            # text_items = [(id, text), ...]
            text_items = [(tid, txt) for tid, txt in text_items if int(tid) in target_ids]

            logger.info(f"📦 Sau khi lọc, còn {len(text_items)} bản ghi cần sinh.")



        # Create generator and generate
        logger.info(f"🎵 Generating {len(text_items)} audio files...")
        generator = DatasetGenerator(output_dir, providers_config)

        # Set metadata path for the generator
        generator.metadata_path = metadata_path

        # If metadata file exists, append to it, otherwise create new
        mode = 'a' if metadata_path.exists() else 'w'
        
        summary = generator.generate_from_text_list(
            text_items=text_items,
            provider_model_voice_list=provider_model_voice_list,
            delay_between_requests=args.delay,
            batch_size=args.batch_size,
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
    exit_code = main()
    sys.exit(exit_code)


# Usage Example
# python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess.py \
# --text_path /media/nampv1/hdd/data/vn_commune_district_province/raw/text/commune_list_with_no_prefix.txt \
# --output_dir /media/nampv1/hdd/data/Voice-of-Address/commune_list_with_no_prefix/ \
# --provider gtts \
# --delay 3.0 \
# --batch_size 10 \
# --continue_from_text_id 1000 \
# --text_ids "97,98,99,100,101,102,103,104,4609,4610,4611,4612"

