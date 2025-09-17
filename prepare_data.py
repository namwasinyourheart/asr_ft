#!/usr/bin/env python3
"""
Data preparation utilities for speech datasets (Hugging Face `datasets`).
Refactored / cleaned version of original script.
"""
import os
import re
import string
import unicodedata
import shutil
import warnings

from typing import Callable, Dict, List, Tuple

from datasets import (
    Audio,
    DatasetDict,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from pyarrow.lib import ArrowInvalid

from hydra import initialize, compose
from omegaconf import OmegaConf
from transformers import set_seed
from tqdm.auto import tqdm

from src.utils.audio_utils import (
    # add_column_datasetname,
    # add_column_filename,
    add_sample_id,
    get_filename2sid,
    get_sid2meta,
    unify_colnames,
    unify_splitnames,
)
from src.utils.exp_utils import create_exp_dir, setup_environment
from src.utils.model_utils import load_processor
from src.utils.utils import load_dict_from_json, save_dict_to_json

warnings.filterwarnings("ignore")


# ---------------------------
# Config utilities
# ---------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config file for generating.",
    )
    args, override_args = parser.parse_known_args()
    return args, override_args


def load_cfg(config_path: str, override_args: List[str] = None):
    """
    Load a configuration file using Hydra and OmegaConf.

    Returns:
        (cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args)
    """
    override_args = override_args or []
    config_path = os.path.normpath(config_path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config_dir = os.path.dirname(config_path)
    config_fn = os.path.splitext(os.path.basename(config_path))[0]

    try:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=config_fn, overrides=override_args)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

    exp_args = cfg.exp_manager
    data_args = cfg.data
    model_args = cfg.model
    train_args = cfg.train
    eval_args = cfg.evaluate
    device_args = cfg.device
    gen_args = cfg.generate

    return cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args


def save_cfg(cfg, config_path: str):
    OmegaConf.save(cfg, config_path)
    print(f"Configuration saved to {config_path}")


# ---------------------------
# Text normalization utilities
# ---------------------------
def preprocess_text(text: str) -> str:
    """Normalize a single text string (NFKC, lowercase, remove punctuation/hyphens)."""
    if text is None:
        return text
    txt = unicodedata.normalize("NFKC", text)
    txt = txt.lower()
    txt = txt.replace("-", " ")
    txt = re.sub(f"[{re.escape(string.punctuation)}]", "", txt)
    return txt


def normalize_text(example: Dict) -> Dict:
    """Map-style normalization for a single example."""
    example["text"] = preprocess_text(example.get("text"))
    return example


def batch_normalize_text(batch: Dict) -> Dict:
    """Batched map function to normalize a list of texts."""
    texts = []
    for t in batch.get("text", []):
        texts.append(preprocess_text(t))
    return {"text": texts}


# ---------------------------
# Feature / label computation
# ---------------------------
def filter_inputs(input_length: int) -> bool:
    """Filter audio inputs: keep only if 0 < length < 48e4 (30s @ 16kHz)."""
    return 0 < input_length < 48e4


def filter_labels(labels_length: int) -> bool:
    """Filter label sequences longer than 448 tokens."""
    return labels_length < 448


def batch_compute_features_and_labels_wrapper(processor):
    """
    Returns a batched processing function for datasets.map:
      - computes input_features via processor.feature_extractor
      - tokenizes text via processor.tokenizer
    Expected batch fields: "audio", "text", "filename", "sample_id"
    """
    def batch_compute_features_and_labels(batch: Dict) -> Dict:
        input_lengths = []
        input_features = []
        labels = []
        labels_length = []
        filenames = []
        sample_ids = []

        for audio, text, fname, sid in zip(
            batch["audio"], batch["text"], batch["filename"], batch["sample_id"]
        ):
            arr = audio["array"]
            sr = audio["sampling_rate"]

            input_lengths.append(len(arr))
            input_features.append(
                processor.feature_extractor(arr, sampling_rate=sr).input_features[0]
            )

            lbl = processor.tokenizer(text).input_ids if text is not None else []
            labels.append(lbl)
            labels_length.append(len(lbl))

            filenames.append(fname)
            sample_ids.append(sid)

        return {
            "sample_id": sample_ids,
            "filename": filenames,
            "input_length": input_lengths,
            "input_features": input_features,
            "labels_length": labels_length,
            "labels": labels,
        }

    return batch_compute_features_and_labels


def compute_features_and_labels_wrapper(processor):
    """
    Single-example compute function for non-batched map.
    Expected example fields: "audio", "text", "filename", "sample_id"
    """
    def compute_features_and_labels(example: Dict) -> Dict:
        audio = example["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]

        example["input_length"] = len(arr)
        example["input_features"] = processor.feature_extractor(
            arr, sampling_rate=sr
        ).input_features[0]

        example["labels"] = processor.tokenizer(example.get("text", "")).input_ids
        example["labels_length"] = len(example["labels"])

        return example

    return compute_features_and_labels


# ---------------------------
# Sharded processing helpers
# ---------------------------
def process_sharded_dataset_dict(
    dataset: DatasetDict,
    func: Callable,
    save_dir: str,
    num_shards: int = 100,
    batch_size: int = 1000,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
    columns_to_remove: List[str] = None,
    desc: str = "Processing",
    force_clear: bool = False,
):
    """
    Process a DatasetDict in shards and save each shard to disk.
    """
    if os.path.exists(save_dir):
        if force_clear:
            shutil.rmtree(save_dir)
        else:
            print(f"save_dir {save_dir} already exists, will resume shards")
    os.makedirs(save_dir, exist_ok=True)

    for split, ds in dataset.items():
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in tqdm(range(num_shards), desc=f"{desc} {split}", unit="shard"):
            shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
            if len(shard) == 0:
                continue

            shard_dir = os.path.join(split_dir, f"shard_{i}")
            if os.path.exists(shard_dir):
                print(f"Skipping existing {split}/{os.path.basename(shard_dir)}")
                continue

            shard = shard.map(
                func,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                writer_batch_size=writer_batch_size,
                remove_columns=columns_to_remove,
                desc=f"{desc} {split} shard {i}/{num_shards}",
            )
            shard.save_to_disk(shard_dir)


def load_sharded_dataset(prepared_data_dir: str) -> DatasetDict:
    """
    Load dataset saved in shard directories and concatenate shards per split.
    """
    splits = {}
    for split in os.listdir(prepared_data_dir):
        split_dir = os.path.join(prepared_data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        shard_paths = sorted(
            [
                os.path.join(split_dir, d)
                for d in os.listdir(split_dir)
                if d.startswith("shard_")
            ]
        )
        if not shard_paths:
            continue
        loaded_shards = [load_from_disk(p) for p in shard_paths]
        splits[split] = concatenate_datasets(loaded_shards)
    return DatasetDict(splits)


# ---------------------------
# Dataset processing pipeline
# ---------------------------
def prepare_metadata(dataset: DatasetDict, common_processed_data_dir: str):
    """
    Generate or load sid2meta and filename2sid mappings.
    """
    os.makedirs(common_processed_data_dir, exist_ok=True)
    all_sid2meta_path = os.path.join(common_processed_data_dir, "all_sid2meta.json")
    all_filename2sid_path = os.path.join(common_processed_data_dir, "all_filename2sid.json")

    if os.path.exists(all_sid2meta_path):
        all_sid2meta = load_dict_from_json(all_sid2meta_path)
    else:
        all_sid2meta = get_sid2meta(
            dataset, fields=("filename", "gender", "dialect", "province_name", "dataset_name")
        )
        save_dict_to_json(all_sid2meta, all_sid2meta_path)

    if os.path.exists(all_filename2sid_path):
        all_filename2sid = load_dict_from_json(all_filename2sid_path)
    else:
        all_filename2sid = get_filename2sid(dataset)
        save_dict_to_json(all_filename2sid, all_filename2sid_path)

    return all_sid2meta, all_filename2sid



def safe_map(dataset, func, initial_batch_size=10000, min_batch_size=1, **kwargs):
    batch_size = initial_batch_size
    while batch_size >= min_batch_size:
        try:
            return dataset.map(
                func,
                batched=True,
                batch_size=batch_size,
                **kwargs,
            )
        except Exception as e:
            # Bắt luôn ArrowInvalid hoặc RuntimeError
            msg = str(e).lower()
            if "out of memory" in msg or "overflow" in msg:
                batch_size = batch_size // 2
                print(f"Batch size too large, reducing to {batch_size}...")
            else:
                raise
    raise RuntimeError(f"Cannot process dataset: batch size < {min_batch_size} still fails.")
import os
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
from datasets import DatasetDict
from tqdm import tqdm

def add_column_filename(dataset, col_audio="audio", col_name="filename", prefix=None, initial_batch_size=10000):
    """
    Add a 'filename' column to each split in a DatasetDict.
    - If audio has valid paths, use basenames.
    - If no paths exist at all, generate synthetic IDs
      (00001.wav, sample_00001.wav, etc.).
    - Tries fast add_column with pyarrow.string.
    - Falls back to safe_map if ArrowInvalid offset overflow or OOM occurs.
    """
    new_splits = {}
    for split in tqdm(dataset, desc="Adding filename"):
        dset = dataset[split]

        if col_name in dset.column_names:
            new_splits[split] = dset
            continue

        # Peek first example → check if audio paths exist
        first_ex = dset[0][col_audio]
        if isinstance(first_ex, dict):
            has_path = bool(first_ex.get("path"))
        else:
            has_path = bool(dset.features[col_audio].decode_example(first_ex).get("path"))

        if not has_path:
            # generate synthetic IDs
            width = len(str(len(dset)))
            if prefix is None:
                filenames = [f"{i:0{width}d}.wav" for i in range(len(dset))]
            else:
                filenames = [f"{prefix}_{i:0{width}d}.wav" for i in range(len(dset))]
        else:
            # extract basenames
            filenames = []
            for ex in dset:
                audio_val = ex[col_audio]
                if isinstance(audio_val, dict):
                    path = audio_val.get("path", None)
                else:
                    path = dset.features[col_audio].decode_example(audio_val).get("path", None)
                filenames.append(os.path.basename(path) if path else "")

        # # --- Fast path
        # try:
        #     values = pa.array(filenames, type=pa.string())
        #     new_splits[split] = dset.add_column(col_name, values)
        #     continue
        # except ArrowInvalid:
        #     print(f"[WARN] Fallback to safe_map for split '{split}' (ArrowInvalid offset overflow).")
        # except Exception as e:
        #     print(f"[WARN] Fallback to safe_map for split '{split}' due to error: {e}")

        # --- Fallback: safe_map
        def _add_col(batch, indices):
            return {col_name: [filenames[i] for i in indices]}

        new_splits[split] = safe_map(
            dset,
            lambda batch, indices: _add_col(batch, indices),
            initial_batch_size=initial_batch_size,
            with_indices=True,
            desc=f"Adding {col_name} (safe_map) to {split}"
        )

    return DatasetDict(new_splits)



from datasets import DatasetDict
import pyarrow as pa
from tqdm import tqdm
from pyarrow.lib import ArrowInvalid

def add_column_datasetname(dataset, ds_name, initial_batch_size=10000, col_name="dataset_name"):
    """
    Add a dataset_name column to each split in a DatasetDict.
    - Tries fast pyarrow.add_column (large_string).
    - Falls back to safe_map if ArrowInvalid offset overflow or OOM occurs.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        ds_name (str|None): dataset name to fill
        initial_batch_size (int): initial batch size for safe_map fallback
        col_name (str): column name to add (default "dataset_name")

    Returns:
        DatasetDict: with dataset_name column added
    """
    if ds_name is None:
        return dataset

    new_splits = {}
    for split in tqdm(dataset, desc="Adding dataset_name"):
        dset = dataset[split]

        if col_name in dset.column_names:
            new_splits[split] = dset
            continue

        # n = len(dset)

        # # --- Fast path: pyarrow add_column
        # try:
        #     values = pa.array([ds_name] * n, type=pa.string())
        #     new_splits[split] = dset.add_column(col_name, values)
        #     continue
        # except ArrowInvalid:
        #     print(f"[WARN] Fallback to safe_map for split '{split}' (ArrowInvalid offset overflow).")
        # except Exception as e:
        #     print(f"[WARN] Fallback to safe_map for split '{split}' due to error: {e}")

        # --- Fallback: safe_map
        def _add_col(batch):
            length = len(next(iter(batch.values())))
            return {col_name: [ds_name] * length}

        new_splits[split] = safe_map(
            dset,
            _add_col,
            initial_batch_size=initial_batch_size,
            desc=f"Adding {col_name} (safe_map) to {split}"
        )

    return DatasetDict(new_splits)



def process_dataset(
    dataset: DatasetDict,
    processor,
    prepared_data_dir: str,
    data_args,
    exp_args,
) -> DatasetDict:
    """
    Normalize text, compute features/labels, and filter dataset.
    Returns dataset ready for training/evaluation.
    """
    dataset = unify_colnames(dataset)
    dataset = unify_splitnames(dataset)
    dataset = add_sample_id(dataset)
    dataset = add_column_filename(dataset)

    root_data_dir = data_args.root_data_dir
    common_processed_data_dir = os.path.join(root_data_dir, "processed")

    # Prepare metadata mapping files for raw dataset
    _ = prepare_metadata(dataset, common_processed_data_dir)

    # Normalize text
    # dataset = dataset.map(
    #     batch_normalize_text,
    #     batched=True,
    #     batch_size=getattr(data_args, "batch_size", 1024),
    #     num_proc=getattr(data_args, "num_proc", 1),
    #     desc="Normalizing text...",
    # )

    dataset = safe_map(
        dataset,
        batch_normalize_text,
        initial_batch_size=10000,
        min_batch_size=1,
        num_proc=getattr(data_args, "num_proc", 1),
        desc="Normalizing text...",
    )

    # Compute features and labels
    batch_fn = batch_compute_features_and_labels_wrapper(processor)

    if getattr(data_args, "do_shard_for_feature_computation", False):
        process_sharded_dataset_dict(
            dataset,
            func=batch_fn,
            save_dir=prepared_data_dir,
            num_shards=getattr(data_args, "num_shards", 50),
            batch_size=getattr(data_args, "batch_size", 1000),
            num_proc=getattr(data_args, "num_proc", 1),
            writer_batch_size=getattr(data_args, "writer_batch_size", 4000),
            columns_to_remove=None,
            desc="Computing features and labels",
            force_clear=False,
        )
        dataset = load_sharded_dataset(prepared_data_dir)
    else:
        if not os.path.exists(prepared_data_dir):
            new_splits = {}
            for split, ds in dataset.items():
                # compute and remove extra columns while writing out
                new_splits[split] = ds.map(
                    batch_fn,
                    batched=True,
                    batch_size=getattr(data_args, "batch_size", 1000),
                    num_proc=getattr(data_args, "num_proc", 1),
                    writer_batch_size=getattr(data_args, "writer_batch_size", 4000),
                    remove_columns=[c for c in ds.column_names if c not in ["sample_id", "filename", "input_features", "labels"]],
                    desc=f"Computing features and labels ({split})",
                )
            dataset = DatasetDict(new_splits)
        else:
            dataset = load_from_disk(prepared_data_dir)
            return dataset

    # Filter and cleanup
    dataset = (
        dataset.filter(filter_inputs, input_columns=["input_length"])
        .filter(filter_labels, input_columns=["labels_length"])
        .remove_columns(["input_length", "labels_length"])
    )

    return dataset


# ---------------------------
# HF dataset creation and splitting helpers
# ---------------------------
def create_hf_ds(
    dataset_script_path: str,
    data_dir: str,
    save_dir: str = None,
    streaming: bool = False,
) -> DatasetDict:
    """
    Create HF dataset from a local dataset script and optionally save to disk.
    """
    ds = load_dataset(
        path=dataset_script_path,
        data_dir=data_dir,
        trust_remote_code=True,
        streaming=streaming,
    )

    if not streaming and save_dir:
        ds.save_to_disk(save_dir)
    return ds


def make_splits(dataset: DatasetDict, test_size=0.1, dev_size=0.1, seed=42) -> DatasetDict:
    """
    Ensure train/dev/test splits exist. If no 'test' split, create one from train.
    """
    if "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        train_data = split["train"]
        test_data = split["test"]
    else:
        train_data = dataset["train"]
        test_data = dataset["test"]

    train_dev = train_data.train_test_split(test_size=dev_size, seed=seed)
    return DatasetDict({"train": train_dev["train"], "dev": train_dev["test"], "test": test_data})


# ---------------------------
# Schema normalization utilities
# ---------------------------
def ensure_gender_is_string(ds, map_batch_size: int = 1024):
    """Ensure 'gender' column exists and is string dtype."""
    if "gender" not in ds.column_names:
        ds = ds.add_column("gender", ["na"] * len(ds))
        # return ds.cast_column({"gender": Value("string")})
        # return ds.cast_column("gender", Value("string"))
        return ds

    def _to_str(batch):
        col = []
        for x in batch["gender"]:
            if x is None:
                col.append("na")
            elif isinstance(x, str):
                col.append(x)
            else:
                try:
                    col.append(str(x))
                except Exception:
                    col.append("na")
        return {"gender": col}

    ds = ds.map(_to_str, batched=True, batch_size=map_batch_size, desc="normalize gender to str")
    # ds = ds.cast_column("gender", Value("string"))
    return ds

from datasets import Value, ClassLabel

def ensure_gender_is_string(ds, map_batch_size: int = 1024):
    if "gender" not in ds.column_names:
        ds = ds.add_column("gender", ["na"] * len(ds))

    # Nếu đang là ClassLabel -> convert thủ công
    if isinstance(ds.features["gender"], ClassLabel):
        names = ds.features["gender"].names
        def to_str(batch):
            return {"gender": [names[x] if x is not None else "na" for x in batch["gender"]]}
        ds = ds.map(to_str, batched=True, batch_size=map_batch_size, desc="map gender->string")

    # Nếu không phải string -> ép sang str
    def force_str(batch):
        return {"gender": [str(x) if x is not None else "na" for x in batch["gender"]]}
    ds = ds.map(force_str, batched=True, batch_size=map_batch_size, desc="force gender->string")

    return ds




def force_all_strings(ds, target: str = "string", map_batch_size: int = 1024):
    """
    Convert all string-like columns to `Value("string")` safely without triggering
    large memory combine operations.
    """
    target_value = Value(target)
    new_features = ds.features.copy()

    for col, feature in ds.features.items():
        if isinstance(feature, Value) and feature.dtype in ("string", "large_string"):
            new_features[col] = target_value

    # map to enforce objects as strings
    to_map = [col for col, feat in new_features.items() if isinstance(feat, Value) and feat.dtype == target]
    if to_map:
        def _batch_convert(batch):
            return {col: [str(x) if x is not None else "" for x in batch[col]] for col in to_map}
        ds = ds.map(_batch_convert, batched=True, batch_size=map_batch_size, desc="Force cast all string cols (safe)")

    ds = ds.cast(new_features, batch_size=map_batch_size)
    return ds

from datasets import Value, ClassLabel

def force_all_strings(ds, target: str = "string", map_batch_size: int = 1024):
    """
    Convert all non-audio columns to `Value("string")`, including ClassLabel, int, float,
    string, large_string.
    """
    target_value = Value(target)
    new_features = ds.features.copy()

    # B1. ép schema: tất cả non-audio -> string
    for col, feature in ds.features.items():
        if col == "audio":
            continue
        if isinstance(feature, (Value, ClassLabel)):  # Value(string/large_string/int/...) or ClassLabel
            new_features[col] = target_value
        else:
            # fallback: ép luôn về string
            new_features[col] = target_value

    # B2. convert dữ liệu -> string
    def _batch_convert(batch):
        out = {}
        for col in ds.column_names:
            if col == "audio":
                out[col] = batch[col]
            else:
                out[col] = [str(x) if x is not None else "" for x in batch[col]]
        return out

    ds = ds.map(_batch_convert, batched=True, batch_size=map_batch_size, desc="force all to string")

    # B3. cast schema để features đồng bộ
    ds = ds.cast(new_features, batch_size=map_batch_size)

    return ds


def normalize_schema(ds, map_batch_size: int = 1024):
    """
    Normalize schema:
      - cast audio to mono 16k
      - ensure gender exists and is string
    """
    if "audio" in ds.column_names:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))
        except ArrowInvalid:
            print("[WARN] Offset overflow casting 'audio', fallback skip cast.")
        except Exception as e:
            print(f"[WARN] Error casting 'audio': {e}")

    ds = ensure_gender_is_string(ds, map_batch_size=map_batch_size)
    ds = force_all_strings(ds, map_batch_size=map_batch_size)
    return ds

def normalize_schema(ds, map_batch_size: int = 1024):
    # 1. Chuẩn hóa audio
    if "audio" in ds.column_names:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))
        except Exception as e:
            print(f"[WARN] Skipping audio cast: {e}")

    # 2. Ensure gender exists (optional)
    if "gender" not in ds.column_names:
        ds = ds.add_column("gender", ["na"] * len(ds))

    # 3. Force tất cả metadata -> string
    ds = force_all_strings(ds, map_batch_size=map_batch_size)
    return ds



# ---------------------------
# Multi-dataset merging pipeline
# ---------------------------
def prepare_multi_data(exp_args, data_args, model_args, device_args):
    """
    Load, (optionally) split/subset many datasets, normalize, merge them,
    compute features/labels and save prepared dataset.
    """
    root_data_dir = data_args.root_data_dir
    common_processed_data_dir = getattr(data_args, "common_processed_data_dir", os.path.join(root_data_dir, "processed"))
    exps_data_dir = getattr(data_args, "exps_data_dir", os.path.join(root_data_dir, "exps"))

    prepared_data_dir = (
        data_args.prepared_data_dir
        or os.path.join(exps_data_dir, f"{exp_args.exp_name}__{exp_args.exp_variant}")
    )
    print("prepared_data_dir:", prepared_data_dir)

    from datasets import concatenate_datasets, Features, Value, Audio

    # Định nghĩa schema chuẩn cho toàn bộ pipeline
    FINAL_FEATURES = Features({
        "sample_id": Value("string"),
        "filename": Value("string"),
        "dataset_name": Value("string"),
        "audio": Audio(sampling_rate=16000, mono=True),
        "text": Value("string"),
        # "gender": Value("string"),
        # "dialect": Value("string"),
        # "age": Value("string"),
        # "province_name": Value("string"),
    })

    def normalize_and_cast(ds, features=FINAL_FEATURES):
        # 1. Drop các cột thừa
        extra_cols = [c for c in ds.column_names if c not in features]
        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # 2. Thêm các cột thiếu
        for col in features.keys():
            if col not in ds.column_names:
                ds = ds.add_column(col, ["na"] * len(ds))

        # 3. Cast sang schema chuẩn
        ds = ds.cast(features)
        return ds


    from datasets import concatenate_datasets

    def normalize_and_cast_sharded(ds, features=FINAL_FEATURES, num_shards=8):
        """
        Chuẩn hóa schema theo FINAL_FEATURES nhưng tránh overflow bằng cách
        chia dataset thành nhiều shard nhỏ rồi cast từng shard.
        """
        shards = []
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)

            # 1. Drop cột thừa
            extra_cols = [c for c in shard.column_names if c not in features]
            if extra_cols:
                shard = shard.remove_columns(extra_cols)

            # 2. Thêm cột thiếu
            for col in features.keys():
                if col not in shard.column_names:
                    shard = shard.add_column(col, ["na"] * len(shard))

            # Cast cột gender trước
            if "gender" in shard.column_names:
                shard = shard.map(lambda x: {"gender": str(x["gender"])})

            # # 3. Cast từng cột theo features
            # for col, feat in features.items():
            #     if col in shard.column_names:
            #         try:
            #             shard = shard.cast_column(col, feat)
            #         except Exception as e:
            #             print(f"[WARN] cast_column failed for {col}: {e}")

            shards.append(shard)

        # Hợp nhất lại
        return concatenate_datasets(shards)


        from datasets import concatenate_datasets

    def normalize_and_cast_auto_shard(ds, features=FINAL_FEATURES, shard_threshold=20000, max_shards=8):
        """
        Chuẩn hóa schema theo FINAL_FEATURES.
        Nếu dataset quá lớn (rows > shard_threshold), sẽ chia shard để tránh overflow.
        """
        num_rows = len(ds)
        if num_rows > shard_threshold:
            num_shards = min(max_shards, max(1, num_rows // shard_threshold))
            print(f"[INFO] Using {num_shards} shards for dataset with {num_rows} rows")
            shards = []
            for i in range(num_shards):
                shard = ds.shard(num_shards=num_shards, index=i)
                shard = _normalize_and_cast_single(shard, features)
                shards.append(shard)
            return concatenate_datasets(shards)
        else:
            return _normalize_and_cast_single(ds, features)


    def _normalize_and_cast_single(ds, features):
        # 1. Drop cột thừa
        extra_cols = [c for c in ds.column_names if c not in features]
        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # 2. Thêm cột thiếu
        for col in features.keys():
            if col not in ds.column_names:
                ds = ds.add_column(col, ["na"] * len(ds))

        # Cast cột gender trước
        if "gender" in ds.column_names:
            ds = ds.map(lambda x: {"gender": str(x["gender"])})

        # # 3. Cast từng cột
        # for col, feat in features.items():
        #     if col in ds.column_names:
        #         try:
        #             ds = ds.cast_column(col, feat)
        #         except Exception as e:
        #             print(f"[WARN] cast_column failed for {col}: {e}")

        ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))

        # if "sample_id" in ds.column_names:
        #     ds = ds.cast_column("sample_id", Value("string"))

        return ds



    # from datasets import concatenate_datasets, Audio, Value
    # from pyarrow.lib import ArrowInvalid

    # def normalize_and_cast_auto_shard(ds, features=FINAL_FEATURES, shard_threshold=20000, max_shards=8, fallback_batch_size=1024):
    #     """
    #     Chuẩn hóa schema theo FINAL_FEATURES.
    #     Nếu dataset quá lớn (rows > shard_threshold), sẽ chia shard để tránh overflow.
    #     Fallback: nếu pyarrow offset overflow → dùng batched map với casting đúng.
    #     """
    #     num_rows = len(ds)

    #     def _process_shard(shard):
    #         try:
    #             return _normalize_and_cast_single(shard, features)
    #         except ArrowInvalid as e:
    #             print(f"[WARN] ArrowInvalid offset overflow, falling back to batched map: {e}")

    #             def _normalize_batch(batch):
    #                 # 1. Drop cột thừa
    #                 batch = {k: v for k, v in batch.items() if k in features}

    #                 # 2. Thêm cột thiếu
    #                 for col in features:
    #                     if col not in batch:
    #                         batch[col] = ["na"] * len(next(iter(batch.values())))

    #                 # 3. Cast gender
    #                 if "gender" in batch:
    #                     batch["gender"] = [str(g) for g in batch["gender"]]

    #                 # 4. Cast audio & sample_id
    #                 if "audio" in batch:
    #                     # Không dùng pyarrow trực tiếp, để dạng dict bình thường, convert sau map
    #                     batch["audio"] = [
    #                         {"path": a["path"], "array": a.get("array", None), "sampling_rate": 16000}
    #                         for a in batch["audio"]
    #                     ]
    #                 if "sample_id" in batch:
    #                     batch["sample_id"] = [str(s) for s in batch["sample_id"]]

    #                 return batch

    #             shard = shard.map(
    #                 _normalize_batch,
    #                 batched=True,
    #                 batch_size=fallback_batch_size,
    #                 desc="Fallback normalization"
    #             )

    #             # Cast audio & sample_id PyArrow sau map
    #             if "audio" in shard.column_names:
    #                 shard = shard.cast_column("audio", Audio(sampling_rate=16000, mono=True))
    #             if "sample_id" in shard.column_names:
    #                 shard = shard.cast_column("sample_id", Value("string"))

    #             return shard

    #         except Exception as e:
    #             print(f"[WARN] Fallback map due to error: {e}")
    #             return shard.map(
    #                 lambda batch: batch,
    #                 batched=True,
    #                 batch_size=fallback_batch_size,
    #                 desc="Fallback normalization"
    #             )

    #     if num_rows > shard_threshold:
    #         num_shards = min(max_shards, max(1, num_rows // shard_threshold))
    #         print(f"[INFO] Using {num_shards} shards for dataset with {num_rows} rows")
    #         shards = []
    #         for i in range(num_shards):
    #             shard = ds.shard(num_shards=num_shards, index=i)
    #             shard = _process_shard(shard)
    #             shards.append(shard)
    #         return concatenate_datasets(shards)
    #     else:
    #         return _process_shard(ds)


    # def _normalize_and_cast_single(ds, features):
    #     # 1. Drop cột thừa
    #     extra_cols = [c for c in ds.column_names if c not in features]
    #     if extra_cols:
    #         ds = ds.remove_columns(extra_cols)

    #     # 2. Thêm cột thiếu
    #     for col in features.keys():
    #         if col not in ds.column_names:
    #             ds = ds.add_column(col, ["na"] * len(ds))

    #     # 3. Cast gender
    #     if "gender" in ds.column_names:
    #         ds = ds.map(lambda x: {"gender": str(x["gender"])})

    #     # 4. Cast audio & sample_id
    #     ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))
    #     if "sample_id" in ds.column_names:
    #         ds = ds.cast_column("sample_id", Value("string"))

    #     return ds





    if not os.path.exists(prepared_data_dir) or data_args.continue_prep:
        merged_dataset = {}

        for i, ds_cfg in enumerate(data_args.datasets):
            # if i == 2:
            #     break
            ds_name = ds_cfg.get("name", "unknown")
            hf_raw_data_dir = ds_cfg.get("hf_raw_data_dir", os.path.join(root_data_dir, "raw", "hf", ds_name))
            dataset_script_path = ds_cfg.get("dataset_script_path")
            dataset_source_dir = ds_cfg.get("dataset_source_dir")
            use_existing_hfds = ds_cfg.get("use_existing_hfds", True)

            print(f"\n=== Loading dataset: {ds_name} ===")
            if os.path.exists(hf_raw_data_dir) and use_existing_hfds:
                dataset = load_from_disk(hf_raw_data_dir)
            else:
                dataset = create_hf_ds(
                    dataset_script_path=dataset_script_path,
                    data_dir=dataset_source_dir,
                    save_dir=hf_raw_data_dir if ds_cfg.get("save_hfds", False) else None,
                    streaming=ds_cfg.get("streaming", False),
                )

            # split
            do_split = ds_cfg.get("do_split", data_args.do_split)
            if do_split:
                dataset = make_splits(dataset, data_args.test_ratio, data_args.val_ratio, exp_args.seed)

            # subset
            subset_ratio = ds_cfg.get("subset_ratio", data_args.subset_ratio)
            if subset_ratio and 0 < subset_ratio < 1:
                dataset = DatasetDict({
                    split: dataset[split].shuffle(seed=exp_args.seed).select(range(int(subset_ratio * len(dataset[split]))))
                    for split in dataset.keys()
                })

            # add dataset_name, unify, add ids/filenames, dtype fixes
            # dataset = add_column_datasetname(dataset, ds_name, data_args.add_col_dsname_batch_size)
            dataset = add_column_datasetname(dataset,ds_name, initial_batch_size=10000)
            dataset = unify_colnames(dataset)
            dataset = unify_splitnames(dataset)
            dataset = add_sample_id(dataset)

            # dataset = add_column_filename(dataset, col_audio="audio", col_name="filename", prefix=None, map_batch_size=data_args.add_col_dsname_batch_size)
            dataset = add_column_filename(dataset, col_audio="audio", col_name="filename", prefix=None, initial_batch_size=10000)

            dataset = unify_sample_id_dtype(dataset, dtype="string", map_batch_size=data_args.add_col_dsname_batch_size)

            for i, split in enumerate(dataset.keys()):
                if split not in merged_dataset:
                    merged_dataset[split] = []
                # ds = normalize_and_cast(dataset[split], FINAL_FEATURES)
                # ds = normalize_and_cast_sharded(dataset[split], FINAL_FEATURES, num_shards=10)
                # ds = normalize_and_cast_auto_shard(dataset[split], FINAL_FEATURES, shard_threshold=20000, max_shards=8, fallback_batch_size=data_args.fallback_batch_size)
                ds = normalize_and_cast_auto_shard(dataset[split], FINAL_FEATURES, shard_threshold=10000, max_shards=8)
                merged_dataset[split].append(ds)
                

        for split, ds_list in merged_dataset.items():
            print("split:", split)
            print("ds_list:", ds_list)

        # # Normalize, cast, and concatenate per split
        # for split, ds_list in merged_dataset.items():
        #     ds_list = [normalize_schema(d, map_batch_size=data_args.add_col_dsname_batch_size) for d in ds_list]
        #     # ds_list = [ensure_gender_is_string(d, map_batch_size=data_args.add_col_dsname_batch_size) for d in ds_list]
        #     # ds_list = [force_all_strings(d, map_batch_size=data_args.add_col_dsname_batch_size) for d in ds_list]
        #     merged_dataset[split] = concatenate_datasets(ds_list)

        

        for split, ds_list in merged_dataset.items():
            merged_dataset[split] = concatenate_datasets(ds_list)

        merged_dataset = DatasetDict(merged_dataset)

        # Load processor + process merged dataset
        processor = load_processor(model_args)
        prepared_dataset = process_dataset(merged_dataset, processor, prepared_data_dir, data_args, exp_args)
        all_sid2meta, all_filename2sid = prepare_metadata(merged_dataset, os.path.join(root_data_dir, "processed"))

        if not os.path.exists(prepared_data_dir):
            prepared_dataset.save_to_disk(prepared_data_dir)
    else:
        print("Loading prepared dataset from disk ...")
        if data_args.do_shard_for_feature_computation:
            prepared_dataset = load_sharded_dataset(prepared_data_dir)
        else:
            prepared_dataset = load_from_disk(prepared_data_dir)
        all_sid2meta, all_filename2sid = prepare_metadata(prepared_dataset, common_processed_data_dir)

    if data_args.do_show:
        show_ds_examples(prepared_dataset)

    return prepared_dataset, all_sid2meta


# ---------------------------
# Utilities for schema / dtype fixes used above
# ---------------------------
def unify_sample_id_dtype(dataset_dict: Dict[str, object], dtype: str = "string", map_batch_size: int = 1024) -> DatasetDict:
    """
    Ensure 'sample_id' column has consistent dtype across splits.
    Tries cast_column then falls back to map() when ArrowInvalid occurs.
    """
    hf_dtype = Value(dtype)
    new_splits = {}

    for split, dset in dataset_dict.items():
        if "sample_id" not in dset.column_names:
            new_splits[split] = dset
            continue

        try:
            new_splits[split] = dset.cast_column("sample_id", hf_dtype)
            continue
        except ArrowInvalid:
            print(f"[WARN] Offset overflow in split '{split}', fallback to map().")
        except Exception as e:
            print(f"[WARN] Error in cast_column on split '{split}', fallback to map(): {e}")

        def _cast_ids(batch):
            if dtype == "string":
                return {"sample_id": [str(x) for x in batch["sample_id"]]}
            elif dtype.startswith("int"):
                return {"sample_id": [int(x) for x in batch["sample_id"]]}
            elif dtype.startswith("float"):
                return {"sample_id": [float(x) for x in batch["sample_id"]]}
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        dset = dset.map(
            _cast_ids,
            batched=True,
            batch_size=map_batch_size,
            desc=f"Casting sample_id in {split} (fallback)",
        )
        new_splits[split] = dset

    return DatasetDict(new_splits)


# ---------------------------
# Small helper to display examples
# ---------------------------
def show_ds_examples(ds_dict: DatasetDict, num_examples: int = 3, show_audio_array: bool = False, audio_preview_len: int = 10):
    """
    Print a few examples from each split for quick inspection.
    """
    for split_name, ds in ds_dict.items():
        print(f"\n=== {split_name.upper()} ===")
        for i, example in enumerate(ds):
            if i >= num_examples:
                break
            print(f"\nExample {i + 1}:")
            for key, value in example.items():
                display_value = value
                if isinstance(value, dict) and "array" in value:
                    arr = value["array"]
                    sr = value.get("sampling_rate")
                    path = value.get("path")
                    if show_audio_array:
                        arr_preview = arr[:audio_preview_len]
                        display_value = f"<Audio shape={arr.shape}, sr={sr}, path={path}, preview={arr_preview}>"
                    else:
                        display_value = f"<Audio shape={arr.shape}, sr={sr}, path={path}>"
                elif hasattr(value, "shape"):
                    display_value = f"<{type(value).__name__} shape={value.shape}>"
                elif isinstance(value, str) and len(value) > 80:
                    display_value = value[:77] + "..."
                print(f"  {key}: {display_value}")


# ---------------------------
# Single-dataset prepare flow
# ---------------------------
def prepare_data(exp_args, data_args, model_args, device_args):
    """
    Main flow to prepare a single dataset and metadata.
    Returns: prepared_dataset, all_sid2meta
    """
    root_data_dir = data_args.root_data_dir
    hf_raw_data_dir = getattr(data_args, "hf_raw_data_dir", os.path.join(root_data_dir, "raw", "hf"))
    common_processed_data_dir = getattr(data_args, "common_processed_data_dir", os.path.join(root_data_dir, "processed"))
    exps_data_dir = getattr(data_args, "exps_data_dir", os.path.join(root_data_dir, "exps"))

    prepared_data_dir = (
        data_args.prepared_data_dir
        or os.path.join(exps_data_dir, f"{exp_args.exp_name}__{exp_args.exp_variant}")
    )

    print("prepared_data_dir:", prepared_data_dir)

    if not os.path.exists(prepared_data_dir) or data_args.continue_prep:
        if os.path.exists(hf_raw_data_dir) and data_args.use_existing_hfds:
            dataset = load_from_disk(hf_raw_data_dir)
        else:
            dataset = create_hf_ds(
                dataset_script_path=data_args.dataset_script_path,
                data_dir=data_args.dataset_source_dir,
                save_dir=hf_raw_data_dir if data_args.save_hfds else None,
            )

        if data_args.subset_ratio and 0 < data_args.subset_ratio < 1:
            dataset = DatasetDict({
                split: dataset[split].shuffle(seed=exp_args.seed).select(range(int(data_args.subset_ratio * len(dataset[split]))))
                for split in dataset.keys()
            })

        if data_args.do_split:
            dataset = make_splits(dataset, data_args.test_ratio, data_args.val_ratio, exp_args.seed)

        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        processor = load_processor(model_args)

        prepared_dataset = process_dataset(dataset, processor, prepared_data_dir, data_args, exp_args)
        all_sid2meta, all_filename2sid = prepare_metadata(dataset, common_processed_data_dir)

        if not os.path.exists(prepared_data_dir):
            prepared_dataset.save_to_disk(prepared_data_dir)
    else:
        print("Loading prepared dataset from disk ...")
        if data_args.do_shard_for_feature_computation:
            prepared_dataset = load_sharded_dataset(prepared_data_dir)
        else:
            prepared_dataset = load_from_disk(prepared_data_dir)
        all_sid2meta, all_filename2sid = prepare_metadata(prepared_dataset, common_processed_data_dir)

    if data_args.do_show:
        show_ds_examples(prepared_dataset)

    return prepared_dataset, all_sid2meta


# ---------------------------
# Entrypoint
# ---------------------------
def main():
    setup_environment()
    args, override_args = parse_args()
    cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args = load_cfg(
        args.config_path, override_args
    )

    if cfg.exp_manager.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    exp_name = cfg.exp_manager.exp_name
    exps_dir = cfg.exp_manager.exps_dir
    exp_variant = cfg.exp_manager.exp_variant
    (exp_dir, exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir) = create_exp_dir(
        exp_name, exp_variant, exps_dir
    )

    config_path = os.path.join(exp_variant_dir, f"{exp_name}__{exp_variant}.yaml")
    save_cfg(cfg, config_path)

    set_seed(exp_args.seed)

    if data_args.do_merge:
        prepared_dataset, all_sid2meta = prepare_multi_data(exp_args, data_args, model_args, device_args)
    else:
        prepared_dataset, all_sid2meta = prepare_data(exp_args, data_args, model_args, device_args)

    print(prepared_dataset)
    print(all_sid2meta.keys())
    for split in all_sid2meta:
        print(f"{split}: {len(all_sid2meta[split])} sid2meta")


if __name__ == "__main__":
    main()
