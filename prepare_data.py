import os

import warnings

from datasets import load_from_disk, Audio

from hydra import initialize, compose
from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils import load_whisper_model, load_processor

from src.utils.exp_utils import setup_environment, create_exp_dir

from tqdm.auto import tqdm


warnings.filterwarnings("ignore")


from src.utils.utils import save_dict_to_json, load_dict_from_json

import string
import unicodedata
import re
import time

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for generating.")

    args, override_args = parser.parse_known_args()
    return args, override_args



def load_cfg(config_path, override_args=None):

    """
    Load a configuration file using Hydra and OmegaConf.

    Args:
        config_path (str): Path to the configuration file.
        override_args (list, optional): List of arguments to override configuration values.

    Returns:
        cfg: Loaded configuration object.
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

def save_cfg(cfg, config_path):
    """
    Save the configuration to a YAML file.

    Args:
        cfg (OmegaConf): The configuration object to save.
        config_path (str): The path where the configuration file will be saved.

    Returns:
        None
    """
    OmegaConf.save(cfg, config_path)
    print(f"Configuration saved to {config_path}")


def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)  # apply NFKC
    text = text.lower()  # convert to uppercase
    text = text.replace("-", " ")  # remove hyphen
    text = re.sub("[{}]".format(string.punctuation), "", text)  # remove punctuation
    # text = re.sub(r"\s+", "", text).strip()  # remove all whitespace ONLY for Thai
    return text


def normalize_text(example):
    text = example["text"]  
    text = unicodedata.normalize("NFKC", text)  # apply NFKC
    text = text.lower()  # convert to lowercase
    text = text.replace("-", " ")  # remove hyphen
    text = re.sub("[{}]".format(string.punctuation), "", text)  # remove punctuation

    # example["text"] = text
    return {"text": text}

import unicodedata, re, string

def batch_normalize_text(batch):
    texts = []
    for t in batch["text"]:
        if t is None:
            texts.append(t)
            continue
        t = unicodedata.normalize("NFKC", t)  
        t = t.lower()                         
        t = t.replace("-", " ")             
        t = re.sub(f"[{re.escape(string.punctuation)}]", "", t) 
        texts.append(t)
    return {"text": texts}
    # return batch


def filter_inputs(input_length):
	"""Filter inputs with zero input length or longer than 30s"""
	return 0 < input_length < 48e4  # 30s × 16kHz

def filter_labels(labels_length):
	"""Filter label sequences longer than max length 448 tokens"""
	return labels_length < 448  # MODEL.config.max_length


def batch_compute_features_and_labels_wrapper(processor):
    def batch_compute_features_and_labels(batch):
        input_lengths = []
        input_features = []
        labels = []
        labels_length = []
        filenames = []
        sample_ids = []

        for audio, text, fname, sid in zip(batch["audio"], batch["text"], batch["filename"], batch["sample_id"]):
            arr = audio["array"]
            sr = audio["sampling_rate"]

            input_lengths.append(len(arr))
            input_features.append(
                processor.feature_extractor(arr, sampling_rate=sr).input_features[0]
            )

            lbl = processor.tokenizer(text).input_ids
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
    def compute_features_and_labels(example):
        audio = example['audio']
        batch["input_length"] = len(audio["array"])
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]).input_features[0]

        example["labels"] = processor.tokenizer(example["text"]).input_ids
        example["labels_length"] = len(example["labels"]) 

        example["filename"] = example["filename"]
        example["sample_id"] = example["sample_id"]

        return example

    return compute_features_and_labels


import shutil
import os
from datasets import DatasetDict, concatenate_datasets, load_from_disk

def process_sharded_dataset_dict(dataset, func, save_dir,
                                 num_shards=100, batch_size=1000, num_proc=1,
                                 writer_batch_size=1000, columns_to_remove=None,
                                 desc="Processing",
                                 force_clear=False):
    """
    Process a DatasetDict in shards, applying `func` to each shard and saving to disk.

    Args:
        dataset (DatasetDict): input dataset
        func (callable): function to apply via map
        save_dir (str): path to save shards
        num_shards (int): number of shards per split
        batch_size (int)
        num_proc (int)
        writer_batch_size (int)
        columns_to_remove(list): columns to remove after processing
        desc (str)
        force_clear (bool): if True, delete save_dir if it exists
    """
    if os.path.exists(save_dir):
        if force_clear:
            print(f"Clearing existing save_dir: {save_dir}")
            shutil.rmtree(save_dir)
        else:
            print(f"save_dir {save_dir} already exists, will resume shards")
            # raise RuntimeError(f"save_dir already exists and is not empty: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    new_splits = {}

    for split, ds in dataset.items():
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        shard_paths = []
        cursor = 0
        for i in tqdm(range(num_shards), desc=f"{desc} {split}", unit="shard"):

            shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
            if len(shard) == 0:
                continue

            start, end = cursor, cursor + len(shard) - 1
            cursor = end + 1

            shard_dir = os.path.join(split_dir, f"shard_{i}_{start}-{end}")
            shard_paths.append(shard_dir)

            if os.path.exists(shard_dir):
                print(f"Skipping existing {split}/{os.path.basename(shard_dir)}")
                continue

            print(f"len(shard): {len(shard)}")
            shard = shard.map(
                func,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                writer_batch_size=writer_batch_size,
                remove_columns=columns_to_remove,
                desc=f"{desc} {split} shard {i}/{num_shards}"
            )



            print(shard.column_names)
            shard.save_to_disk(shard_dir)



        # loaded_shards = [load_from_disk(p) for p in shard_paths if os.path.exists(p)]
        # new_splits[split] = concatenate_datasets(loaded_shards)

    # return DatasetDict(new_splits)



import os
from datasets import Audio, DatasetDict
from src.utils.audio_utils import (
    get_sid2meta, 
    get_filename2sid, 
    unify_colnames,
    unify_splitnames,
    add_sample_id,
    add_column_filename
)

def process_dataset(dataset, processor, prepared_data_dir, data_args, exp_args):
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

    all_sid2meta, all_filename2sid = prepare_metadata(dataset, common_processed_data_dir)


    # Normalize text
    dataset = dataset.map(
        batch_normalize_text,
        batched=True,
        batch_size=data_args.batch_size,
        num_proc=data_args.num_proc,
        desc="Normalizing text..."
    )

    batch_compute_features_and_labels = batch_compute_features_and_labels_wrapper(processor)

    columns_to_retain = ["sample_id", "filename", "input_features", "labels"]
    # columns_to_remove = [
    #     col for col in list(next(iter(dataset['test'])).keys())
    #     if col not in columns_to_retain
    # ]

    # def get_columns_to_remove(ds):
    #     return [col for col in ds.column_names if col not in columns_to_retain]

    def get_cols_to_remove(ds):
        keep_cols = ["sample_id", "filename", "input_features", "labels"]
        return [c for c in ds.column_names if c not in keep_cols]

    # columns_to_remove = get_columns_to_remove(dataset)
    # print("columns_to_remove:", columns_to_remove)

    if data_args.do_shard_for_feature_computation:
        process_sharded_dataset_dict(
            dataset,
            func=batch_compute_features_and_labels,
            save_dir=prepared_data_dir,
            num_shards=data_args.num_shards,  # e.g., 50
            batch_size=data_args.batch_size,
            num_proc=data_args.num_proc, 
            writer_batch_size=data_args.writer_batch_size, # 4000
            columns_to_remove=None,
            desc="Computing features and labels",
            force_clear=False,
        )

        dataset = load_sharded_dataset(prepared_data_dir)

    else:
        if not os.path.exists(prepared_data_dir):
            new_splits = {}
            for split, ds in dataset.items():
                cols_to_remove = get_cols_to_remove(ds)
                new_splits[split] = ds.map(
                    batch_compute_features_and_labels,
                    batched=True,
                    batch_size=data_args.batch_size,
                    num_proc=data_args.num_proc,
                    writer_batch_size=data_args.writer_batch_size,
                    remove_columns=cols_to_remove,
                    desc=f"Computing features and labels ({split})"
                )
            dataset = DatasetDict(new_splits)

    # else:
    #     if not os.path.exists(prepared_data_dir):
    #         # Normal .map() without sharding
    #         dataset = dataset.map(
    #             batch_compute_features_and_labels,
    #             batched=True,
    #             batch_size=data_args.batch_size,  # 4000
    #             num_proc=data_args.num_proc, # 4
    #             writer_batch_size=data_args.writer_batch_size,
    #             remove_columns=columns_to_remove,
    #             desc="Computing features and labels"
    #         )
        else: 
            dataset = load_from_disk(prepared_data_dir)
            return dataset

    # Filter dataset by input and label lengths
    dataset = (
        dataset
        .filter(filter_inputs, 
                input_columns=["input_length"], 
                # batched=True,
                # num_proc=data_args.num_proc
        )
        .filter(filter_labels, 
                input_columns=["labels_length"], 
                # batched=True,
                # num_proc=data_args.num_proc
        )
        .remove_columns(["input_length", "labels_length"])
    )

    return dataset


def prepare_metadata(dataset, common_processed_data_dir):
    """
    Generate or load sid2meta and filename2sid mappings.
    Returns: all_sid2meta, all_filename2sid
    """
    os.makedirs(common_processed_data_dir, exist_ok=True)

    all_sid2meta_path = os.path.join(common_processed_data_dir, "all_sid2meta.json")
    all_filename2sid_path = os.path.join(common_processed_data_dir, "all_filename2sid.json")

    if os.path.exists(all_sid2meta_path):
        print(f"Loading all_sid2meta from {all_sid2meta_path}")
        all_sid2meta = load_dict_from_json(all_sid2meta_path)
    else:
        print(f"Generating all_sid2meta and saving to {all_sid2meta_path}")
        all_sid2meta = get_sid2meta(dataset, fields=("filename", "gender", "dialect", "province_name", "dataset_name"))
        save_dict_to_json(all_sid2meta, all_sid2meta_path)

    if os.path.exists(all_filename2sid_path):
        print(f"Loading all_filename2sid from {all_filename2sid_path}")
        all_filename2sid = load_dict_from_json(all_filename2sid_path)
    else:
        print(f"Generating all_filename2sid and saving to {all_filename2sid_path}")
        all_filename2sid = get_filename2sid(dataset)
        save_dict_to_json(all_filename2sid, all_filename2sid_path)

    return all_sid2meta, all_filename2sid

def load_sharded_dataset(prepared_data_dir):
    """
    Load dataset from sharded directories.
    Returns DatasetDict with all splits concatenated from shards.
    """
    splits = {}
    for split in os.listdir(prepared_data_dir):
        split_dir = os.path.join(prepared_data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        shard_paths = [
            os.path.join(split_dir, d)
            for d in os.listdir(split_dir)
            if d.startswith("shard_")
        ]
        shard_paths = sorted(shard_paths)  # đảm bảo đúng thứ tự
        loaded_shards = [load_from_disk(p) for p in shard_paths]
        splits[split] = concatenate_datasets(loaded_shards)

        # break
    return DatasetDict(splits)


from datasets import load_dataset, DatasetDict

def create_hf_ds(dataset_script_path: str, 
    data_dir: str, 
    save_dir: str = None, 
    streaming: bool = False) -> DatasetDict:
    """
    Create Hugging Face dataset from a local dataset loading script.

    Args:
        dataset_script_path (str): path to dataset.py (e.g., "prepare_data/vivos_dataset/vivos.py")
        data_dir (str): path to raw dataset directory (e.g., local_raw_data_dir_vivos)
        save_dir (str, optional): if given, will save the dataset to this directory
        streaming (bool): whether to use streaming mode (avoid loading full dataset into RAM)

    Returns:
        DatasetDict
    """
    print(f"Loading dataset using script={dataset_script_path}, data_dir={data_dir} ...")

    ds = load_dataset(
        path=dataset_script_path,
        data_dir=data_dir,
        trust_remote_code=True,  # needed if dataset.py uses custom code
        streaming=streaming,
    )

    if not streaming:
        # print(ds)
        # print("Example sample:", ds["train"][0])

        if save_dir:
            print(f"Saving dataset to disk at {save_dir} ...")
            ds.save_to_disk(save_dir)

    return ds

def make_splits(dataset, test_size=0.1, dev_size=0.1, seed=42):
    """
    Generate train/dev/test splits from a Hugging Face dataset.
    If dataset already has test split, reuse it; otherwise create it.

    Args:
        dataset (DatasetDict): Hugging Face dataset with at least 'train' split.
        test_size (int|float): Number of samples or ratio for test split.
        dev_size (int|float): Number of samples or ratio for dev split (from train).
        seed (int): Random seed for reproducibility.

    Returns:
        DatasetDict with train, dev, test splits.
    """
    # ensure test split exists
    if "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        train_data = split["train"]
        test_data = split["test"]
    else:
        train_data = dataset["train"]
        test_data = dataset["test"]

    # create dev split from train
    train_dev = train_data.train_test_split(test_size=dev_size, seed=seed)

    return DatasetDict({
        "train": train_dev["train"],
        "dev": train_dev["test"],
        "test": test_data
    })


def prepare_data(exp_args, data_args, model_args, device_args):
    """
    Main flow to prepare dataset and metadata.
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
        # Load raw dataset
        if os.path.exists(hf_raw_data_dir) and data_args.use_existing_hfds:
            print(f"Loading existing HF dataset from {hf_raw_data_dir}")
            dataset = load_from_disk(hf_raw_data_dir)

        else:
            print("Creating new HF dataset ...")
            dataset = create_hf_ds(  
                dataset_script_path=data_args.dataset_script_path, 
                data_dir=data_args.dataset_source_dir,   # raw dataset folder (e.g. VIVOS raw)
                save_dir=hf_raw_data_dir if data_args.save_hfds else None
            )

        # Optionally subset
        if data_args.subset_ratio and 0 < data_args.subset_ratio < 1:
            print("Subsetting dataset ...")
            dataset = DatasetDict({
                split: dataset[split].shuffle(seed=exp_args.seed)
                            .select(range(int(data_args.subset_ratio * len(dataset[split]))))
                for split in dataset.keys()
            })


        # Split into train, dev, test splits
        if data_args.do_split:
            print("Splitting dataset into train, dev, test splits ...")
            dataset = make_splits(dataset, data_args.test_ratio, data_args.val_ratio, exp_args.seed)

        # Cast audio column
        print("Casting audio column ...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Load processor
        print("Loading processor ...")
        processor = load_processor(model_args)

        # Process dataset and metadata
        print("Processing dataset and metadata ...")
        prepared_dataset = process_dataset(dataset, processor, prepared_data_dir, data_args, exp_args)
        all_sid2meta, all_filename2sid = prepare_metadata(dataset, common_processed_data_dir)

        # Save prepared dataset
        print("Saving prepared dataset ... to", prepared_data_dir)
        if not os.path.exists(prepared_data_dir): 
            prepared_dataset.save_to_disk(prepared_data_dir)

    else:
        # Load prepared dataset from disk
        print("Loading prepared dataset from disk ...")
        if data_args.do_shard_for_feature_computation:
            prepared_dataset = load_sharded_dataset(prepared_data_dir)
        else:
            prepared_dataset = load_from_disk(prepared_data_dir)

        # Load or create metadata if missing
        print("Loading or creating metadata ...")
        all_sid2meta, all_filename2sid = prepare_metadata(prepared_dataset, common_processed_data_dir)

    # Optionally show dataset examples
    if data_args.do_show:
        show_ds_examples(prepared_dataset)

    return prepared_dataset, all_sid2meta


from datasets import DatasetDict, concatenate_datasets, load_from_disk
import os
from datasets import Audio, Value

def normalize_schema(ds):
    # ép audio về schema chuẩn
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # ép gender về string nếu tồn tại
    if "gender" in ds.column_names:
        ds = ds.cast_column("gender", Value("string"))
    return ds


from datasets import Audio, Value
from pyarrow.lib import ArrowInvalid

def normalize_schema(ds, map_batch_size=1024):
    # ép audio về schema chuẩn
    if "audio" in ds.column_names:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        except ArrowInvalid:
            print("[WARN] Offset overflow casting 'audio', fallback skip cast.")
        except Exception as e:
            print(f"[WARN] Error casting 'audio': {e}")

    # ép gender về string nếu tồn tại
    if "gender" in ds.column_names:
        try:
            ds = ds.cast_column("gender", Value("string"))
        except ArrowInvalid:
            print("[WARN] Offset overflow casting 'gender', fallback to map().")

            def _cast_gender(batch):
                return {"gender": [str(x) for x in batch["gender"]]}

            ds = ds.map(
                _cast_gender,
                batched=True,
                batch_size=map_batch_size,
                desc="Casting gender (fallback)"
            )
            # không cần cast lại nữa vì đã str()

        except Exception as e:
            print(f"[WARN] Error casting 'gender': {e}")

    return ds

from datasets import Audio, Value
from pyarrow.lib import ArrowInvalid

def normalize_schema(ds, map_batch_size=1024):
    # Ép audio về schema chuẩn
    if "audio" in ds.column_names:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000, mono=True))
        except ArrowInvalid:
            print("[WARN] Offset overflow casting 'audio', fallback skip cast.")
        except Exception as e:
            print(f"[WARN] Error casting 'audio': {e}")

    # Ép gender về string nếu tồn tại, bất kể dtype ban đầu là gì
    if "gender" in ds.column_names:
        # Sử dụng map để đảm bảo chuyển đổi thành string an toàn
        def _cast_gender_to_string(batch):
            return {"gender": [str(x) for x in batch["gender"]]}
        
        # Áp dụng map chỉ khi dtype không phải là string
        if ds.features["gender"].dtype != "string":
            print("[INFO] Casting 'gender' to string to unify schema.")
            ds = ds.map(
                _cast_gender_to_string,
                batched=True,
                batch_size=map_batch_size,
                desc="Casting gender to string (safe)"
            )
            # Sau khi map, gán lại schema để đảm bảo đúng
            ds = ds.cast_column("gender", Value("string"))

    return ds


from datasets import DatasetDict

from datasets import Value
from src.utils.audio_utils import add_column_datasetname

def unify_sample_id_dtype(dataset_dict, dtype="string"):
    hf_dtype = Value(dtype)  # convert string -> datasets type
    for split in dataset_dict:
        if "sample_id" in dataset_dict[split].column_names:
            dataset_dict[split] = dataset_dict[split].cast_column("sample_id", hf_dtype)
    return dataset_dict

from datasets import Value
from pyarrow.lib import ArrowInvalid

def unify_sample_id_dtype(dataset_dict, dtype="string", map_batch_size=1024):
    """
    Ensure 'sample_id' column has consistent dtype across splits.
    - Try fast cast_column first.
    - If ArrowInvalid offset overflow occurs, fallback to map with batch conversion.
    """
    hf_dtype = Value(dtype)
    new_splits = {}

    for split, dset in dataset_dict.items():
        if "sample_id" not in dset.column_names:
            new_splits[split] = dset
            continue

        # --- Fast path
        try:
            new_splits[split] = dset.cast_column("sample_id", hf_dtype)
            continue
        except ArrowInvalid:
            print(f"[WARN] Offset overflow in split '{split}', fallback to map().")
        except Exception as e:
            print(f"[WARN] Error in cast_column on split '{split}', fallback to map(): {e}")

        # --- Fallback: enforce type with map
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
            desc=f"Casting sample_id in {split} (fallback)"
        )

        new_splits[split] = dset

    return DatasetDict(new_splits)


from datasets import Dataset, DatasetDict, Value

def force_all_strings(ds, map_batch_size=1024):
    """
    Convert all string-like columns (string, large_string) to Value("string").
    """
    for col, feature in ds.features.items():
        if isinstance(feature, Value) and feature.dtype in ("string", "large_string"):
            try:
                ds = ds.cast_column(col, Value("string"))
            except Exception:
                # fallback khi overflow
                ds = ds.map(
                    lambda batch: {col: [str(x) for x in batch[col]]},
                    batched=True,
                    batch_size=map_batch_size,
                    desc=f"Force cast {col} to string (fallback)"
                )
    return ds


def force_all_strings(ds, target="string", map_batch_size=1024):
    """
    Convert all string-like columns to uniform dtype without triggering offset overflow.
    - Avoids pyarrow combine_chunks().
    """
    from datasets import Features, Value

    target_value = Value(target)
    new_features = ds.features.copy()

    for col, feature in ds.features.items():
        if isinstance(feature, Value) and feature.dtype in ("string", "large_string"):
            new_features[col] = target_value

    # map để ép dữ liệu thành string, batch nhỏ để không overflow
    ds = ds.map(
        lambda batch: {col: [str(x) if x is not None else "" for x in batch[col]] 
                      for col, feat in new_features.items() if isinstance(feat, Value) and feat.dtype == target},
        batched=True,
        batch_size=map_batch_size,
        desc="Force cast all string cols (safe mode)"
    )

    # gán lại features schema, nhưng KHÔNG ép combine_chunks
    ds = ds.cast(new_features, batch_size=map_batch_size)

    return ds



def prepare_multi_data(exp_args, data_args, model_args, device_args):
    """
    Load, split (if needed), optionally subset, normalize, and merge multiple datasets,
    then preprocess into a unified prepared dataset.
    Returns: prepared_dataset, all_sid2meta
    """

    root_data_dir = data_args.root_data_dir
    common_processed_data_dir = getattr(data_args, "common_processed_data_dir", os.path.join(root_data_dir, "processed"))
    exps_data_dir = getattr(data_args, "exps_data_dir", os.path.join(root_data_dir, "exps"))
    
    prepared_data_dir = (
        data_args.prepared_data_dir
        or os.path.join(exps_data_dir, f"{exp_args.exp_name}__{exp_args.exp_variant}")
    )
    print("prepared_data_dir:", prepared_data_dir)

    if not os.path.exists(prepared_data_dir) or data_args.continue_prep:

        merged_dataset = {}

        # 1. Iterate over datasets declared in config
        for ds_cfg in data_args.datasets:
            ds_name = ds_cfg.get("name", "unknown")
            hf_raw_data_dir = ds_cfg.get(
                "hf_raw_data_dir",
                os.path.join(root_data_dir, "raw", "hf", ds_name),
            )
            dataset_source_dir = ds_cfg.get("dataset_source_dir", None)
            dataset_script_path = ds_cfg.get("dataset_script_path", None)
            use_existing_hfds = ds_cfg.get("use_existing_hfds", True)

            print(f"\n=== Loading dataset: {ds_name} ===")
            if os.path.exists(hf_raw_data_dir) and use_existing_hfds:
                print(f"Loading existing HF dataset from {hf_raw_data_dir}")
                dataset = load_from_disk(hf_raw_data_dir)
                # print("dataset after load from disk:", dataset.column_names)
            else:
                print(f"Creating HF dataset from {dataset_script_path}, {dataset_source_dir}")
                dataset = create_hf_ds(
                    dataset_script_path=dataset_script_path,
                    data_dir=dataset_source_dir,
                    save_dir=hf_raw_data_dir if ds_cfg.get("save_hfds", False) else None,
                    streaming=ds_cfg.get("streaming", False),
                )
                # print("dataset after create hfd:", dataset.column_names)

            # 2. Split dataset into train/dev/test if requested (per dataset)
            do_split = ds_cfg.get("do_split", data_args.do_split)
            if do_split:
                print(f"Splitting dataset {ds_name} into train/dev/test ...")
                dataset = make_splits(dataset, data_args.test_ratio, data_args.val_ratio, exp_args.seed)

            # 3. Optionally subset this dataset before merging
            subset_ratio = ds_cfg.get("subset_ratio", data_args.subset_ratio)
            if subset_ratio and 0 < subset_ratio < 1:
                print(f"Subsetting dataset {ds_name} with ratio={subset_ratio} ...")
                dataset = DatasetDict({
                    split: dataset[split]
                        .shuffle(seed=exp_args.seed)
                        .select(range(int(subset_ratio * len(dataset[split]))))
                    for split in dataset.keys()
                })

                # print("dataset after subset:", dataset.column_names)

            # 4. Add dataset_name column for traceability
            # for split in dataset.keys():
            #     dataset[split] = dataset[split].add_column("dataset_name", [ds_name] * len(dataset[split]))

            # 4. Add dataset_name column for traceability
            # for split in dataset.keys():
            #     if "dataset_name" not in dataset[split].column_names:
            #         dataset[split] = dataset[split].map(
            #             lambda _: {"dataset_name": ds_name},
            #             desc=f"Adding dataset_name to {split}"
            #         )

            dataset = add_column_datasetname(dataset, ds_name, data_args.add_col_dsname_batch_size)

            # dataset = add_column_datasetname(dataset, ds_name)

            # print("dataset after add dataset_name:", dataset.column_names)

            dataset = unify_colnames(dataset)
            # print("dataset after unify colnames:", dataset.column_names)

            dataset = unify_splitnames(dataset)
            # print("dataset after unify splitnames:", dataset.column_names)

            dataset = add_sample_id(dataset)
            # print("dataset after add sample id:", dataset.column_names)

            # dataset = add_column_filename(dataset)
            dataset = add_column_filename(dataset, col_audio="audio", col_name="filename", prefix=None, map_batch_size=data_args.add_col_dsname_batch_size)
            # print("dataset after add filename:", dataset.column_names)


            # dataset = unify_sample_id_dtype(dataset, dtype="string")
            dataset = unify_sample_id_dtype(dataset, dtype="string", map_batch_size=data_args.add_col_dsname_batch_size)

            for split in dataset.keys():
                # print("len split", split, ":", len(dataset[split]))
                print("len split", split, ":", len(dataset[split]))

            # print("len split train:", len(dataset["train"]))
            # print("len split val:", len(dataset["val"]))
            # print("len split test:", len(dataset["test"]))

            # 5. Accumulate splits
            for split in dataset.keys():
                if split not in merged_dataset:
                    merged_dataset[split] = []
                merged_dataset[split].append(dataset[split])


        print("=== Processing merged dataset ===")

        # print("merged_dataset:", merged_dataset.keys())

        # for split, ds_list in merged_dataset.items():
        #     # Chuẩn hóa schema audio cho từng dataset trong list
        #     # new_list = [d.cast_column("audio", Audio(sampling_rate=None)) for d in ds_list]
        #     # new_list = [normalize_schema(d) for d in ds_list]

        #     # new_list = [normalize_schema(d, map_batch_size=data_args.add_col_dsname_batch_size) for d in ds_list]

        #     # merged_dataset[split] = concatenate_datasets(new_list)

        #     # new_list = [
        #     #     force_all_strings(normalize_schema(d, map_batch_size=128))
        #     #     for d in ds_list
        #     # ]
        #     # merged_dataset[split] = concatenate_datasets(new_list)

        #     merged_dataset[split] = concatenate_datasets(ds_list)

        for split, ds_list in merged_dataset.items():
            # Chuẩn hóa schema audio cho từng dataset trong list
            # Normalize schema for each dataset in the list before concatenation.
            new_list = [normalize_schema(d, map_batch_size=data_args.add_col_dsname_batch_size) for d in ds_list]

            # Force all string-like columns to a unified "string" dtype to prevent "large_string" issues.
            new_list = [force_all_strings(d, map_batch_size=data_args.add_col_dsname_batch_size) for d in new_list]

            # Now, concatenate the datasets with unified schemas.
            merged_dataset[split] = concatenate_datasets(new_list)




        # print("merged_dataset:", merged_dataset)

        # 6. Concatenate across datasets
        # merged_dataset = DatasetDict({
        #     split: concatenate_datasets(ds_list) for split, ds_list in merged_dataset.items()
        # })
        merged_dataset = DatasetDict(merged_dataset)
        # print("merged_dataset:", merged_dataset)

        # # 7. Cast audio column
        # print("Casting audio column to 16kHz ...")
        # merged_dataset = merged_dataset.cast_column("audio", Audio(sampling_rate=16000))

        # 8. Load processor
        print("Loading processor ...")
        processor = load_processor(model_args)

        # 9. Process dataset + metadata
        print("Processing merged dataset ...")
        prepared_dataset = process_dataset(merged_dataset, processor, prepared_data_dir, data_args, exp_args)
        all_sid2meta, all_filename2sid = prepare_metadata(merged_dataset, os.path.join(root_data_dir, "processed"))

        print("prepared_dataset:", prepared_dataset)

        # 10. Save prepared dataset
        if not os.path.exists(prepared_data_dir):
            print("Saving prepared dataset ... to", prepared_data_dir)
            prepared_dataset.save_to_disk(prepared_data_dir)
    else:
        # Load prepared dataset from disk
        print("Loading prepared dataset from disk ...")
        if data_args.do_shard_for_feature_computation:
            prepared_dataset = load_sharded_dataset(prepared_data_dir)
        else:
            prepared_dataset = load_from_disk(prepared_data_dir)

        # Load or create metadata if missing
        print("Loading or creating metadata ...")
        all_sid2meta, all_filename2sid = prepare_metadata(prepared_dataset, common_processed_data_dir)

    # 11. Optionally show dataset examples
    if data_args.do_show:
        show_ds_examples(prepared_dataset)

    return prepared_dataset, all_sid2meta



def show_ds_examples(ds_dict, num_examples=3, show_audio_array=False, audio_preview_len=10):
    """
    Print examples from each split of an IterableDatasetDict.

    Args:
        ds_dict: IterableDatasetDict
        num_examples: number of examples per split
        show_audio_array: if True, show array values (truncated if long)
        audio_preview_len: number of elements of array to show if show_audio_array=True
    """
    for split_name, ds in ds_dict.items():
        print(f"\n=== {split_name.upper()} ===")
        for i, example in enumerate(ds):
            if i >= num_examples:
                break
            print(f"\nExample {i + 1}:")
            for key, value in example.items():
                display_value = value
                # handle audio dict
                if isinstance(value, dict) and 'array' in value:
                    arr = value['array']
                    sr = value.get('sampling_rate')
                    path = value.get('path')
                    if show_audio_array:
                        # show only a preview to avoid huge output
                        arr_preview = arr[:audio_preview_len]
                        display_value = f"<Audio shape={arr.shape}, sampling_rate={sr}, path={path}, array_preview={arr_preview}>"
                    else:
                        display_value = f"<Audio shape={arr.shape}, sampling_rate={sr}, path={path}>"
                elif hasattr(value, "shape"):
                    display_value = f"<{type(value).__name__} shape={value.shape}>"
                elif isinstance(value, str) and len(value) > 80:
                    display_value = value[:77] + "..."
                print(f"  {key}: {display_value}")


def main():
    setup_environment()

    # Parse arguments
    args, override_args = parse_args()

    # Load configuration
    cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args = load_cfg(args.config_path, override_args)


    if cfg.exp_manager.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    # Create experiment directories
    exp_name = cfg.exp_manager.exp_name
    exps_dir = cfg.exp_manager.exps_dir
    exp_variant = cfg.exp_manager.exp_variant

    (exp_dir, exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir) = create_exp_dir(exp_name, exp_variant, exps_dir)


    # Save configuration if have any changes from the overrides
    config_path = os.path.join(exp_variant_dir, f'{exp_name}__{exp_variant}.yaml')
    save_cfg(cfg, config_path)

    # Set seed
    set_seed(exp_args.seed)

    if data_args.do_merge:
        prepared_dataset, all_sid2meta = prepare_multi_data(exp_args, data_args, model_args, device_args)
    else:
        prepared_dataset, all_sid2meta = prepare_data(exp_args, data_args, model_args, device_args)

    # dataset, all_sid2meta = prepare_data(exp_args, data_args, model_args, device_args)

    print(prepared_dataset)
    print(all_sid2meta.keys())
    for split in all_sid2meta:
        print(f"{split}: {len(all_sid2meta[split])} sid2meta")


if __name__ == "__main__":
    main()
