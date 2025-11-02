#!/usr/bin/env python3
"""
Merge shard outputs (metadata.tsv and wav files) back into the parent voice_dir.

Usage example:
  python merge_shards.py \
    --voice_dir /path/to/output_dir/provider/model/voice \
    --shards_subdir shards \
    --output_metadata /path/to/output_dir/provider/model/voice/metadata.tsv

Notes:
- It scans voice_dir/<shards_subdir>/* for shard directories.
- For each shard, it reads metadata.tsv if available and copies wav/* into voice_dir/wav/.
- It concatenates shard metadata and (optionally) existing parent metadata, de-duplicates by text_id, and writes merged metadata.tsv.
- By default, WAV files are copied if they don't exist in parent; set --overwrite_wav to overwrite.
"""

import argparse
import os
from pathlib import Path
import shutil
import pandas as pd

META_COLS = [
    "utt_id", "text_id", "text", "audio_path", "provider", "model",
    "voice", "tts_type", "sample_rate", "lang", "duration", "gen_date"
]


def parse_args():
    p = argparse.ArgumentParser(description="Merge shard metadata and wav files into parent voice_dir")
    p.add_argument("--voice_dir", type=str, required=True, help="Parent voice_dir (e.g., .../provider/model/voice)")
    p.add_argument("--shards_subdir", type=str, default="shards", help="Subdirectory name containing shards (default: shards)")
    p.add_argument("--output_metadata", type=str, default=None, help="Output metadata.tsv path (default: voice_dir/metadata.tsv)")
    p.add_argument("--merge_wav", action="store_true", default=True, help="Copy/move shard wavs into parent voice_dir/wav (default: True)")
    p.add_argument("--no-merge_wav", dest="merge_wav", action="store_false", help="Disable merging wav files")
    p.add_argument("--overwrite_wav", action="store_true", default=False, help="Overwrite existing wavs in parent (default: False)")
    p.add_argument("--move_wav", action="store_true", default=False, help="Move wavs instead of copy (saves disk, removes from shard)")
    p.add_argument("--include_parent_metadata", action="store_true", default=True, help="Include existing parent metadata.tsv in merge (default: True)")
    p.add_argument("--no-include_parent_metadata", dest="include_parent_metadata", action="store_false", help="Ignore existing parent metadata")
    p.add_argument("--backup_parent_metadata", action="store_true", default=True, help="Backup existing parent metadata.tsv before overwriting (default: True)")
    p.add_argument("--no-backup_parent_metadata", dest="backup_parent_metadata", action="store_false", help="Do not create backup of parent metadata")
    p.add_argument("--cleanup_shards", action="store_true", default=False, help="After merge, delete shard wav/metadata (keep directories)")
    p.add_argument("--remove_shard_dirs", action="store_true", default=False, help="After merge, remove entire shard directories (implies cleanup)")
    return p.parse_args()


def read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=META_COLS)
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=0,
            names=META_COLS,
            index_col=False,
        )
        # normalize dtypes we rely on
        if "text_id" in df.columns:
            df["text_id"] = pd.to_numeric(df["text_id"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        print(f"⚠️ Could not read metadata at {path}: {e}")
        return pd.DataFrame(columns=META_COLS)


def copy_wavs(shard_wav_dir: Path, parent_wav_dir: Path, overwrite: bool, move: bool):
    if not shard_wav_dir.exists():
        return 0, 0
    parent_wav_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for name in os.listdir(shard_wav_dir):
        src = shard_wav_dir / name
        if not src.is_file():
            continue
        dst = parent_wav_dir / name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        try:
            if move:
                # Use move (will overwrite if overwrite=True, else skip handled above)
                if dst.exists() and overwrite:
                    dst.unlink(missing_ok=True)
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"⚠️ Failed to copy {src} -> {dst}: {e}")
    return copied, skipped


def main():
    args = parse_args()
    voice_dir = Path(args.voice_dir)
    if not voice_dir.exists():
        raise FileNotFoundError(f"voice_dir not found: {voice_dir}")
    shards_root = voice_dir / args.shards_subdir
    if not shards_root.exists():
        raise FileNotFoundError(f"shards directory not found: {shards_root}")

    out_meta = Path(args.output_metadata) if args.output_metadata else (voice_dir / "metadata.tsv")

    # Collect shard metadata
    shard_dirs = [d for d in shards_root.iterdir() if d.is_dir()]
    print(f"🔎 Found {len(shard_dirs)} shards under {shards_root}")

    merged = []
    if args.include_parent_metadata:
        parent_meta = read_metadata(out_meta)
        if not parent_meta.empty:
            print(f"📥 Loaded parent metadata: {len(parent_meta)} rows from {out_meta}")
            merged.append(parent_meta)

    total_copied = 0
    total_skipped = 0

    for shard_dir in sorted(shard_dirs):
        meta_path = shard_dir / "metadata.tsv"
        wav_dir = shard_dir / "wav"

        df = read_metadata(meta_path)
        if not df.empty:
            print(f"📥 Loaded shard metadata: {len(df)} rows from {meta_path}")
            merged.append(df)
        else:
            print(f"ℹ️ No metadata found at {meta_path} (or empty)")

        if args.merge_wav:
            copied, skipped = copy_wavs(wav_dir, voice_dir / "wav", overwrite=args.overwrite_wav, move=args.move_wav)
            total_copied += copied
            total_skipped += skipped
            print(f"🎧 WAV merge from {wav_dir}: copied={copied}, skipped={skipped}")
        # Optional cleanup shard files after merging
        if args.cleanup_shards or args.remove_shard_dirs:
            try:
                # Remove shard metadata file
                meta_path.exists() and meta_path.unlink()
                # Remove empty wav files directory if move was used and dir now empty
                if wav_dir.exists() and len(os.listdir(wav_dir)) == 0:
                    wav_dir.rmdir()
                # Remove shard directory entirely if requested
                if args.remove_shard_dirs:
                    import shutil as _shutil
                    _shutil.rmtree(shard_dir, ignore_errors=True)
                    print(f"🗑️  Removed shard directory {shard_dir}")
                else:
                    print(f"🧹 Cleaned shard artifacts in {shard_dir}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup shard {shard_dir}: {e}")

    # Merge metadata frames
    if merged:
        meta = pd.concat(merged, ignore_index=True)
        # Drop duplicate text_id keeping the last occurrence (newest shards override)
        if "text_id" in meta.columns:
            meta.sort_values(by=["text_id", "gen_date"], inplace=True, na_position="last")
            meta.drop_duplicates(subset=["text_id"], keep="last", inplace=True)
            # Final order by text_id ascending
            meta.sort_values(by=["text_id"], inplace=True)
        # Reassign utt_id sequentially 1..N to keep ascending order in merged metadata
        if "utt_id" in meta.columns:
            meta["utt_id"] = list(range(1, len(meta) + 1))
        # Ensure column order
        meta = meta[[c for c in META_COLS if c in meta.columns]]
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        # Optional: backup parent metadata
        if args.backup_parent_metadata and out_meta.exists():
            try:
                backup_path = out_meta.with_suffix(out_meta.suffix + ".bak")
                shutil.copy2(out_meta, backup_path)
                print(f"🗄️  Backed up existing parent metadata to {backup_path}")
            except Exception as e:
                print(f"⚠️ Failed to backup parent metadata: {e}")
        meta.to_csv(out_meta, sep="\t", index=False, header=True)
        print(f"✅ Wrote merged metadata: {len(meta)} rows -> {out_meta}")
    else:
        print("ℹ️ No metadata to merge (no shard metadata and parent excluded)")

    if args.merge_wav:
        print(f"🎉 WAV merge summary: copied={total_copied}, skipped={total_skipped}; parent dir={voice_dir / 'wav'}")


if __name__ == "__main__":
    main()


# python /home/nampv1/projects/asr/asr_ft/augment_data/merge_shards.py \
#   --voice_dir /home/nampv1/projects/asr/asr_ft/augment_data/tests/test_output/minimax_selenium/default/bdtl_spk2 \
#   --merge_wav \
#   --move_wav \
#   --overwrite_wav \
#   --cleanup_shards \
#   --backup_parent_metadata \
#   --remove_shard_dirs

# python /home/nampv1/projects/asr/asr_ft/augment_data/merge_shards.py \
#   --voice_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated/cdp_list_with_prefix/minimax_selenium/default/bdtl_spk2 \
#   --merge_wav \
#   --move_wav \
#   --overwrite_wav \
#   --cleanup_shards \
#   --backup_parent_metadata \
#   --remove_shard_dirs