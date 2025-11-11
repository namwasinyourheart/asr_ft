import os
import warnings

import numpy as np
from datasets import concatenate_datasets
from torch.utils.data import DataLoader

from hydra import initialize, compose
from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils import load_whisper_model, load_processor

from src.utils.exp_utils import setup_environment, create_exp_dir
from src.utils.log_utils import setup_logger

from prepare_data_no_metadata import prepare_data, preprocess_text

from tqdm.auto import tqdm
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

logger = setup_logger(__name__)

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


def write_to_txt(file_path, predictions_list, append=False):
    """Write or append prediction results to a TXT file."""
    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for prediction in predictions_list:
            for key, value in prediction.items():
                f.write(f"{key}: {value}\n")
                f.write("-" * 24 + "\n")
            f.write("\n\n")


def save_predictions(predictions_list, directory, filename, append=False):
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
        write_to_txt(file_path, predictions_list, append=append)
    # elif file_extension == "json":
    #     write_to_json(file_path, predictions_list)
    # elif file_extension == "csv":
    #     write_to_csv(file_path, predictions_list)
    else:
        raise ValueError("Unsupported file extension. Use '.txt', '.json', or '.csv'.")

def save_batch_predictions(directory, filename, predictions_list, batch_step: int):
    """Save a batch of predictions using save_predictions, appending from step>0."""
    append = batch_step != 0
    save_predictions(predictions_list, directory, filename, append=append)

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

    logger.info(f"Metrics saved to {file_path}")

import pandas as pd 
def summarize_metric(metric_by_group: dict, 
                     model_name: str = "model_1", 
                     top_n_province: int = 5,
                     filename: str = "summary_metrics.csv"
                     ):
    row = {"Model": model_name}
    
    # Dialect
    for k, v in metric_by_group["dialect"].items():
        row[f"Dialect_{k}"] = v
    
    # Gender
    gender_map = {1: "Male", 0: "Female", "male": "Male", "female": "Female"}
    for k, v in metric_by_group.get("gender", {}).items():
        row[f"Gender_{gender_map.get(k, str(k))}"] = v

    
    # Province: Get top-n that having highest WER
    for prov, val in sorted(
        metric_by_group.get("province_name", {}).items(),
        key=lambda x: -x[1]
    )[:top_n_province]:
        row[f"Province_{prov}"] = val

    df = pd.DataFrame([row])
    df.to_csv(filename, index=False)
    
    return df
import numpy as np


def calculate_wer_per_sample(ref, hyp):
        r = ref.split()
        h = hyp.split()
        d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint16)
        # d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)

        for i in range(len(r)+1):
            d[i][0] = i
        for j in range(len(h)+1):
            d[0][j] = j

        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitute = d[i-1][j-1] + 1
                    insert    = d[i][j-1] + 1
                    delete    = d[i-1][j] + 1
                    d[i][j] = min(substitute, insert, delete)

        i, j = len(r), len(h)
        S = D = I = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] and r[i-1] == h[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
                S += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i][j] == d[i][j-1] + 1:
                I += 1
                j -= 1
            else:
                D += 1
                i -= 1

        N = max(1, len(r))
        wer_value = (S + D + I) / N
        return wer_value, S, D, I, N

def calculate_wer(refs, hyps, return_details=False):

    total_S = total_D = total_I = total_N = 0
    wer_list = []
    details = []

    for ref, hyp in zip(refs, hyps):
        wer_val, S, D, I, N = calculate_wer_per_sample(ref, hyp)
        total_S += S
        total_D += D
        total_I += I
        total_N += N
        wer_list.append(wer_val)

        if return_details:
            details.append({
                "ref": ref,
                "hyp": hyp,
                "wer": wer_val,
                "S": S,
                "D": D,
                "I": I,
                "N": N,
            })

    micro_wer = (total_S + total_D + total_I) / total_N
    macro_wer = float(np.mean(wer_list)) if wer_list else 0.0

    summary = {
        "micro_wer": micro_wer,
        "macro_wer": macro_wer,
        "S": total_S,
        "D": total_D,
        "I": total_I,
        "N": total_N,
        "n_samples": len(refs),
    }
    if return_details:
        return summary, details
    else:
        return summary
import requests
from consts import TEXT_POSTPROCESSING_URL_ENDPOINT
def postprocess_text_via_api(text):
    url = TEXT_POSTPROCESSING_URL_ENDPOINT
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"text": text}
    
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # Raises an error if the request failed
    postprocessed_text = response.json()['text']
    return postprocessed_text


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

    logger.info("exp_variant_results_dir: {}".format(exp_variant_results_dir))

    # Save configuration if have any changes from the overrides
    config_path = os.path.join(exp_variant_dir, f'{exp_name}__{exp_variant}.yaml')
    save_cfg(cfg, config_path)

    # Set seed
    set_seed(exp_args.seed)

    # Get dataset
    dataset = prepare_data(exp_args, data_args, model_args, device_args)

    # Load model and processor
    from transcribe import load_model_for_transcribe
    
    model = load_model_for_transcribe(model_args, device_args)
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    # model.generation_config.forced_decoder_ids = None

    from accelerate import Accelerator
    accelerator = Accelerator(cpu=device_args.use_cpu)

    model.eval()
    model = model.to(accelerator.device)
    
    processor = load_processor(model_args)
    tokenizer = processor.tokenizer
    

    if eval_args.include_train_split and 'train' in dataset:
        logger.info("Include train split for evaluation")

        if 'test' in dataset:
            test_ds = concatenate_datasets([dataset['test'], dataset['train']])
        else:
            logger.warning("No test split found — using only train split for evaluation")
            test_ds = dataset['train']

    else:
        if 'test' in dataset:
            test_ds = dataset['test']
        else:
            logger.warning("No test split found — using train split as fallback")
            test_ds = dataset['train']

        
    print("test_ds:", test_ds)



    # if eval_args.include_train_split and 'train' in dataset:
    #     logger.info("Include train split for evaluation")
    #     test_ds = concatenate_datasets([dataset['test'], dataset['train']])
    # else:
    #     test_ds = dataset['test']
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    test_dataloader = DataLoader(test_ds, 
                                 batch_size=eval_args.batch_size, 
                                 collate_fn=data_collator
                                )
    predictions_list = []
    all_preds, all_refs = [], []
    
    
    # Pre-compute output path for incremental saving
    incremental_out_path = os.path.join(exp_variant_results_dir, eval_args.prediction_filename)

    for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating...")):

        if step == eval_args.break_step:
                break
        
        with torch.no_grad():
            input_features = batch["input_features"].to(model.device, dtype=model.dtype)
            
            generated_tokens = model.generate(
                input_features=input_features,
                return_dict_in_generate=True,
                # max_new_tokens=gen_args.max_new_tokens,
            ).sequences.cpu().numpy()

            # print("[DEBUG] generated_tokens.shape:", generated_tokens.shape)
            
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions = [preprocess_text(p) for p in decoded_preds]
            ground_truth = [preprocess_text(g) for g in decoded_labels]


            # wer_metric.add_batch(predictions=predictions, 
            #                      references=ground_truth)

            all_preds.extend(predictions)
            all_refs.extend(ground_truth)

            batch_predictions = []
            for i, sid in enumerate(batch["sample_id"]):
                filename = batch["filename"][i]

                pred = predictions[i]
                label = ground_truth[i]

                if eval_args.do_postprocess_text:
                    pred = postprocess_text_via_api(pred)
                    label = postprocess_text_via_api(label)

                    pred = preprocess_text(pred)
                    label = preprocess_text(label)

                sample_wer, S, D, I, N = calculate_wer_per_sample(label, pred)

                # Append prediction info (for final write)
                record = {
                    "sid": sid,
                    "filename": filename,
                    "prediction": pred,
                    "label": label,
                    "wer": sample_wer,
                    "S": S,
                    "D": D,
                    "I": I,
                    "N": N
                }
                predictions_list.append(record)
                batch_predictions.append(record)

            # Incremental saving per batch using unified save API
            save_batch_predictions(
                exp_variant_results_dir,
                eval_args.prediction_filename,
                batch_predictions,
                batch_step=step,
            )
    
    # Calculate WER
    metrics_wer = calculate_wer(all_refs, all_preds, return_details=False)
            
    # Incremental saving is enabled; skip final full overwrite to avoid duplication
    logger.info(f"Predictions were saved incrementally to {incremental_out_path}")

    # Print WER
    print(f"Micro WER: {100 * metrics_wer['micro_wer']:.2f}%")
    print(f"Macro WER: {100 * metrics_wer['macro_wer']:.2f}%")

    # Save metrics
    metrics = {
        "exp_name": exp_args.exp_name,
        "exp_variant": exp_args.exp_variant,
        "micro_wer": float(metrics_wer["micro_wer"]),
        "macro_wer": float(metrics_wer["macro_wer"]),
        "S": int(metrics_wer["S"]),
        "D": int(metrics_wer["D"]),
        "I": int(metrics_wer["I"]),
        "N": int(metrics_wer["N"]),
        "n_samples": int(metrics_wer["n_samples"])
    }

    save_metrics(metrics, exp_variant_results_dir, eval_args.metric_filename)
    print(f"Evaluation completed. Results saved to {exp_variant_results_dir}")
    
if __name__ == "__main__":
    main()