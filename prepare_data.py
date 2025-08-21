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

def compute_features_and_labels_wrapper(processor):
    def compute_features_and_labels(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
    
        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(audio["array"], 
                                                    sampling_rate=16000,).input_features[0]
                                                    # sampling_rate=audio["sampling_rate"]).input_features[0]
    
        # encode target text to label ids 
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
        batch["filename"] = batch["filename"]
        batch["sample_id"] = batch["sample_id"]
        return batch

    return compute_features_and_labels


# from functools import partial


def add_sample_id(ex, idx):
    ex["sample_id"] = idx
    return ex

# ds['test'] = ds['test'].map(add_sample_id, with_indices=True)

def prepare_data(exp_args, data_args, model_args, device_args):
    
    processor = load_processor(model_args)

    # from transformers import WhisperProcessor
    # processor = WhisperProcessor.from_pretrained(model_args.pretrained_model_name_or_path, 
    #                                              lang="vi", 
    #                                              task="transcribe")
    
    columns_to_retain = ["sample_id", "audio", "text", "filename", "input_features", "labels"]

    # dataset = load_dataset(data_args.raw_data_dir, 
    #                        streaming=data_args.streaming)

    dataset = load_from_disk(data_args.raw_data_dir)
    # print(dataset)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    

    # dataset = dataset.map(add_sample_id, with_indices=True)

    # dataset = dataset.map(
    #     add_sample_id,
    #     with_indices=True,
    #     batched=True,        # map theo batch thay vì từng sample
    #     batch_size=100,     # tuỳ chỉnh, giảm nếu vẫn lỗi
    #     # keep_in_memory=False  # ghi tạm ra đĩa thay vì RAM
    # )


    def add_sample_id_streaming(example, idx):
        example["sample_id"] = idx
        return example

    from tqdm.auto import tqdm


    columns_to_remove = [col for col in list(next(iter(dataset['test'])).keys()) if col not in columns_to_retain]

    print("Add sample_id")
    # Sử dụng khi streaming=True
    if data_args.streaming:
        # Thêm sample_id bằng cách sử dụng map với with_indices
        dataset = dataset.map(
            add_sample_id_streaming,
            with_indices=True,
            batched=False  # Xử lý từng mẫu một
        )
    else:
        # Cách xử lý thông thường khi không streaming
        for split in tqdm(dataset):
            sample_ids = list(range(len(dataset[split])))
            dataset[split] = dataset[split].add_column("sample_id", sample_ids)


    print("Get id2meta")
    test_id2meta = {}
    for split in tqdm(dataset):  # hoặc ['test'] nếu chỉ cần test
        for ex in tqdm(dataset[split]):
            test_id2meta[ex["sample_id"]] = {
                "region": ex.get("region", ""),
                "province": ex.get("province_name", ""),
                "gender": ex.get("gender", ""),
            }

    
    test_id2meta_path = os.path.join(exp_args.exps_dir, exp_args.exp_name, exp_args.exp_variant, "data", data_args.id2meta_filename)
    
    print(f"Saving test_id2meta to {test_id2meta_path}")
    save_dict_to_json(test_id2meta, test_id2meta_path)

    compute_features_and_labels = compute_features_and_labels_wrapper(processor)
    # _compute_features_and_labels = partial(compute_features_and_labels)
    
    dataset = dataset.map(compute_features_and_labels, 
                          remove_columns=columns_to_remove)

    if data_args.do_show:
        # Show dataset examples
        show_ds_examples(dataset)

    if data_args.do_save:
        # Lưu dataset
        prepared_data_path = os.path.join(exp_args.exps_dir, exp_args.exp_name, exp_args.exp_variant, "data", data_args.prepared_data_dirname)
        dataset.save_to_disk(prepared_data_path)

    return dataset, test_id2meta

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

    dataset = prepare_data(exp_args, data_args, model_args, device_args)
    print(dataset)

if __name__ == "__main__":
    main()