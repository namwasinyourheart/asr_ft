import os
import shutil
import joblib
import argparse
from functools import partial

import warnings

import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Audio
from torch.utils.data import DataLoader

from hydra import initialize, compose
from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils import load_whisper_model, load_processor

from src.utils.exp_utils import setup_environment, create_exp_dir

from tqdm.auto import tqdm


warnings.filterwarnings("ignore")


import json

def save_dict_to_json(d: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_dict_from_json(filepath: str) -> dict:
    """Load dictionary from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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
    
    # assert os.path.basename(config_path).replace('.yaml', '') == cfg.exp_manager.exp_name, \
    # assert cfg.exp_manager.phase_name + '__' + 
    # assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    # f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."

    # cfg.train.lora.task_type = cfg.train.progress_callback.model_type = cfg.model.model_type
    
    exp_args = cfg.exp_manager
    data_args = cfg.data
    # tokenizer_args = cfg.tokenizer
    # prompt_args = cfg.prompt
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



import string
import unicodedata
import re

def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)  # apply NFKC
    text = text.lower()  # convert to uppercase
    text = text.replace("-", " ")  # remove hyphen
    text = re.sub("[{}]".format(string.punctuation), "", text)  # remove punctuation
    # text = re.sub(r"\s+", "", text).strip()  # remove all whitespace ONLY for Thai
    return text


def filter_inputs(input_length):
	"""Filter inputs with zero input length or longer than 30s"""
	return 0 < input_length < 48e4  # 30s × 16kHz

def filter_labels(labels_length):
	"""Filter label sequences longer than max length 448 tokens"""
	return labels_length < 448  # MODEL.config.max_length

def compute_features_and_labels_wrapper(processor):
    def compute_features_and_labels(batch):
        audio = batch['audio']
        batch["input_length"] = len(audio["array"])
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        batch["labels_length"] = len(batch["labels"]) 

        batch["filename"] = batch["filename"]
        batch["sample_id"] = batch["sample_id"]
        
        return batch

    return compute_features_and_labels


def add_sample_id(ex, idx):
    ex["sample_id"] = idx
    return ex


def get_sid2meta(dataset, 
                fields=("filename", "region","province_name","gender"), 
                splits=None):
    """
    Build id2meta dictionary from dataset.
    
    Args:
        dataset: DatasetDict
        fields: tuple of field names to include in metadata
        splits: list of splits to process (None = all splits)
    
    Returns:
        dict: {sample_id: {field: value, ...}, ...}
    """
    splits = splits or dataset.keys()
    return {
        ex["sample_id"]: {field: ex.get(field, "") for field in fields} 
        for split in tqdm(splits)
        for ex in tqdm(dataset[split])
    }


def get_filename2sid(id2meta: dict) -> dict:
    """
    Tạo dict mapping: filename -> sample_id từ id2meta dict, có tqdm hiển thị tiến trình
    
    Args:
        id2meta (dict): dict như test_id2meta.json đã load
    
    Returns:
        dict: mapping filename -> sample_id
    """
    filename2sid = {}
    for k, v in tqdm(id2meta.items(), desc="Building filename -> sample_id map"):
        filename2sid[v['filename']] = k
    return filename2sid


    
def prepare_data(exp_args, data_args, model_args, device_args):
    
    processor = load_processor(model_args)
    # dataset = load_dataset(data_args.raw_data_dir, 
    #                        streaming=data_args.streaming)

    dataset = load_from_disk(data_args.raw_data_dir)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) 


    print("Add sample_id")
    if data_args.streaming:
        dataset = dataset.map(
            add_sample_id_streaming,
            with_indices=True,
            batched=False  # Xử lý từng mẫu một
        )
    else:
        for split in tqdm(dataset):
            sample_ids = list(range(len(dataset[split])))
            dataset[split] = dataset[split].add_column("sample_id", sample_ids)
    
    # Get id2meta
    print("Getting sid2meta...")
    
    # test_id2meta_path = os.path.join(exp_variant_data_dir, "test_id2meta.json")

    common_processed_data_dir = data_args.common_processed_data_dir
    os.makedirs(common_processed_data_dir, exist_ok=True)
    
    test_id2meta_path = os.path.join(common_processed_data_dir, "test_id2meta.json")

    if not os.path.exists(test_id2meta_path):
        test_id2meta = get_sid2meta(dataset, splits=["test"])
        print(f"Saving test_id2meta to {test_id2meta_path}")
        save_dict_to_json(test_id2meta, test_id2meta_path)
    else:
        print(f"{test_id2meta_path} already exists, skipping creation.")
        test_id2meta = load_dict_from_json(test_id2meta_path)

    # Get filename2sid
    print("Getting filename2sid...")
    
    test_filename2sid_path = os.path.join(common_processed_data_dir, "test_filename2sid.json")
    if not os.path.exists(test_filename2sid_path):
        test_filename2sid = get_filename2sid(test_id2meta)
        print(f"Saving test_id2meta to {test_filename2sid_path}")
        save_dict_to_json(test_filename2sid, test_filename2sid_path)
    
    else:
        print(f"{test_id2meta_path} already exists, skipping creation.")
        test_filename2sid = load_dict_from_json(test_filename2sid_path)
        
    exp_variant_data_dir = os.path.join(exp_args.exps_dir, 
                                        exp_args.exp_name, 
                                        exp_args.exp_variant, 
                                        "data")    
    prepared_data_dir = os.path.join(exp_variant_data_dir, 
                                     data_args.prepared_data_dirname)
    if not os.path.exists(prepared_data_dir):
    
        columns_to_retain = ["sample_id", "filename", "input_features", "labels"]
        columns_to_remove = [col for col in list(next(iter(dataset['test'])).keys()) if col not in columns_to_retain]
        
        compute_features_and_labels = compute_features_and_labels_wrapper(processor)
        
        dataset = dataset.map(compute_features_and_labels, 
                              remove_columns=columns_to_remove,
                              batched=False
                              )

        dataset = (dataset
            .filter(filter_inputs, input_columns= ["input_length"])  # no `remove_columns` coz streaming
        	.filter(filter_labels, input_columns=["labels_length"])
        )


        dataset.save_to_disk(prepared_data_dir)
        
    else:
        dataset = load_from_disk(prepared_data_dir)
    
    if data_args.do_show:
        # Show dataset examples
        show_ds_examples(dataset)

    # if data_args.do_save:
    #     # Lưu dataset
    #     dataset.save_to_disk(prepared_data_dir)

    return dataset,test_id2meta

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
    
    dataset, test_id2meta = prepare_data(exp_args, data_args, model_args, device_args)
    print(dataset)
    print(test_id2meta.keys())


if __name__ == "__main__":
    main()