import os
import joblib
import argparse
from functools import partial

import warnings

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader

from hydra import initialize, compose
from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils import load_whisper_model, load_processor

from src.utils.exp_utils import setup_environment, create_exp_dir

from prepare_data import prepare_data, preprocess_text

from tqdm.auto import tqdm
from evaluate import load
from collections import defaultdict


warnings.filterwarnings("ignore")

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        batch["filename"] = [f["filename"] for f in features]
        batch["sample_id"] = [f["sample_id"] for f in features]

        return batch


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


import json
import csv
import os


def write_to_txt(file_path, predictions_list):
    """Writes prediction results to a TXT file."""
    with open(file_path, "w", encoding="utf-8") as f:

        for prediction in predictions_list:
            for key, value in prediction.items():
                f.write(f"{key}: {value}\n")
                f.write("-" * 24 + "\n")
            # f.write("-" * 48 + "\n\n")
            f.write("\n\n")


def save_predictions(predictions_list, directory, filename):
    """
    Saves predictions in a format determined by the file extension.

    Args:
        predictions_list (list): List of prediction results.
        directory (str): Directory path to save files.
        filename (str): Filename with extension (e.g., 'results.txt', 'results.json', 'results.csv').
    """

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Extract file extension
    file_extension = filename.split('.')[-1].lower()
    file_path = os.path.join(directory, filename)

    # Choose appropriate write function
    if file_extension == "txt":
        write_to_txt(file_path, predictions_list)
    # elif file_extension == "json":
    #     write_to_json(file_path, predictions_list)
    # elif file_extension == "csv":
    #     write_to_csv(file_path, predictions_list)
    else:
        raise ValueError("Unsupported file extension. Use '.txt', '.json', or '.csv'.")

def save_metrics(metrics, directory, filename):
    """
    Saves evaluation metrics (e.g., accuracy) to a TXT file.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
        directory (str): Directory path to save the file.
        filename (str):  Filename with extension .txt
    """
    file_path = os.path.join(directory, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Name: {metrics.get('exp_name', 'N/A')}\n")
        f.write(f"Experiment Variant: {metrics.get('exp_variant', 'N/A')}\n")
        f.write("-" * 48 + "\n\n")
        
        for key, value in metrics.items():
            if key != "exp_name" and key != "exp_variant":  # Avoid duplicating experiment name
                f.write(f"{key}: {value}\n")
        
        f.write("-" * 48 + "\n")

import pandas as pd 
def summarize_metric(metric_by_group: dict, 
                     model_name: str = "model_1", 
                     top_n_province: int = 5,
                     filename: str = "summary_metrics.csv"
                     ):
    row = {"Model": model_name}
    
    # Region
    for k, v in metric_by_group["region"].items():
        row[f"Region_{k}"] = v
    
    # Gender
    gender_map = {1: "Male", 0: "Female"}
    for k, v in metric_by_group["gender"].items():
        row[f"Gender_{gender_map[k]}"] = v
    
    # Province: Get top-n that having highest WER
    top_provinces = sorted(metric_by_group["province_name"].items(), key=lambda x: -x[1])[:top_n_province]
    for prov, val in top_provinces:
        row[f"Province_{prov}"] = val


    df = pd.DataFrame([row])
    df.to_csv(filename, index=False)
    
    return df


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

    # Get dataset
    dataset, id2meta = prepare_data(exp_args, data_args, model_args, device_args)
    
    # Load model and processor
    from transcribe import load_model_for_transcribe
    
    model = load_model_for_transcribe(model_args, device_args)
    # model.generation_config.language = "vi"
    # model.generation_config.task = "transcribe"
    # model.generation_config.forced_decoder_ids = None

    model.eval()
    model = model.to('cuda')
    
    processor = load_processor(model_args)
    tokenizer = processor.tokenizer
    
    test_ds = dataset['test']

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    test_dataloader = DataLoader(test_ds, 
                                 batch_size=eval_args.batch_size, 
                                 collate_fn=data_collator
                                )
    wer_metric = load("wer")
    predictions_list = []
    
    
    grouped_preds = {
        "region": defaultdict(list),
        "province_name": defaultdict(list),
        "gender": defaultdict(list),
    }
    grouped_labels = {
        "region": defaultdict(list),
        "province_name": defaultdict(list),
        "gender": defaultdict(list),
    }
    
    
    for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating...")):

        if step == eval_args.break_step:
                break
        
        with torch.no_grad():
            input_features = batch["input_features"].to(model.device, dtype=model.dtype)
            
            generated_tokens = model.generate(
                input_features=input_features,
                return_dict_in_generate=True,
                max_new_tokens=gen_args.max_new_tokens,
            ).sequences.cpu().numpy()

            # print("[DEBUG] generated_tokens.shape:", generated_tokens.shape)
            
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions = [preprocess_text(decoded_pred) for decoded_pred in decoded_preds]
            ground_truth = [preprocess_text(gt) for gt in decoded_labels]

            wer_metric.add_batch(predictions=decoded_preds, 
                                 references=decoded_labels)

            for sid, fn, pred, label in zip(batch["sample_id"], batch["filename"], predictions, ground_truth):
                predictions_list.append({
                    "sid": sid,
                    "filename": fn,
                    "prediction": pred,
                    "label": label,
                })
    
            for i, sid in enumerate(batch["sample_id"]):
                meta = id2meta[str(sid)]
                for key in grouped_preds.keys():
                    grouped_preds[key][meta[key]].append(decoded_preds[i])
                    grouped_labels[key][meta[key]].append(decoded_labels[i])
    
    # Save predictions
    save_predictions(predictions_list, 
                     exp_variant_results_dir, 
                     eval_args.prediction_filename,
                    )
    # Compute WER
    wer= 100 * wer_metric.compute()
    print("Overall WER:", wer)

    
    # Compute WER by group: region, province_name, gender 
    wer_by_group = {}
    
    for meta_key in grouped_preds.keys():
        wer_by_group[meta_key] = {}
        for group_value in grouped_preds[meta_key]:

            metric_tmp = load("wer")
            metric_tmp.add_batch(
                predictions=grouped_preds[meta_key][group_value],
                references=grouped_labels[meta_key][group_value],
            )
            wer_by_group[meta_key][group_value] = 100 * metric_tmp.compute()
    print("WER by Group:", wer_by_group)

    # Save metrics
    metrics = {
        "exp_name": exp_args.exp_name,
        "exp_variant": exp_args.exp_variant,
        "wer": wer,
        "wer_by_group": wer_by_group
    }

    save_metrics(metrics, 
                 exp_variant_results_dir, 
                 eval_args.metric_filename)

    summarize_metric(wer_by_group,
                     model_name=exp_args.exp_name + '_' + exp_args.exp_variant,
                     top_n_province=10,
                     filename=os.path.join(exp_variant_results_dir, f"summary_metrics_{exp_args.exp_name}_{exp_args.exp_variant}.csv")
                     )
    
if __name__ == "__main__":
    main()