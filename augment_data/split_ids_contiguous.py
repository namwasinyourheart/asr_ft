#!/usr/bin/env python3
"""
Split a list of text_id (one per line) into N contiguous shards.

Usage:
  python split_ids_contiguous.py \
    --ids_path /path/to/ids_all.txt \
    --num_shards 3 \
    --output_dir /path/to/output_dir \
    --output_prefix ids_shard_

This will create files:
  /path/to/output_dir/ids_shard_0.txt
  /path/to/output_dir/ids_shard_1.txt
  /path/to/output_dir/ids_shard_2.txt

Notes:
- Input file must contain one integer text_id per line.
- Shards are contiguous by order in the input file.
- When total count is not divisible by N, the first (total % N) shards get one extra item.
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split ids file into contiguous shards")
    p.add_argument("--ids_path", type=str, required=True, help="Path to ids_all.txt")
    p.add_argument("--num_shards", type=int, required=True, help="Number of shards to create")
    p.add_argument("--output_dir", type=str, default=None, help="Directory to write shards. Default: same dir as ids_path")
    p.add_argument("--output_prefix", type=str, default="ids_shard_", help="Prefix for shard files")
    return p.parse_args()


def split_contiguous(ids, num_shards):
    n = len(ids)
    base = n // num_shards
    rem = n % num_shards
    shards = []
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        end = start + size
        shards.append(ids[start:end])
        start = end
    return shards


def main():
    args = parse_args()
    ids_path = Path(args.ids_path)
    if not ids_path.exists():
        raise FileNotFoundError(f"ids_path not found: {ids_path}")
    out_dir = Path(args.output_dir) if args.output_dir else ids_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with ids_path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    shards = split_contiguous(ids, args.num_shards)

    for i, shard in enumerate(shards):
        out_file = out_dir / f"{args.output_prefix}{i}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(shard) + ("\n" if shard else ""))
        print(f"✅ Wrote shard {i}: {len(shard)} ids -> {out_file}")

    print(f"Done. Total ids: {len(ids)}; shards: {len(shards)}")


if __name__ == "__main__":
    main()
