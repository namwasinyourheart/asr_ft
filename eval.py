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
    exp_variant_dir = cfg.exp_manager.exp_variant

    (exp_dir, exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir) = create_exp_dir(exp_name, exp_variant_dir, exps_dir)

    # Save configuration if have any changes from the overrides
    config_path = os.path.join(exp_variant_dir, exp_name + '.yaml')
    save_cfg(cfg, config_path)

    # Set seed
    set_seed(exp_args.seed)

    
    if data_args.is_prepared:
        prepare_data_path = data_args.prepared_data_path
        dataset = joblib.load(prepare_data_path)
    
    else:
        from prepare_data import prepare_data
        dataset, id2meta = prepare_data(exp_args, data_args, model_args, device_args)
        print(dataset)

    
    # Load model and processor
    from transcribe import load_model_for_transcribe
    
    model = load_model_for_transcribe(model_args, device_args)
    # model.generation_config.language = "vi"
    # model.generation_config.task = "transcribe"
    # model.generation_config.forced_decoder_ids = None
    
    processor = load_processor(model_args)

    # from transformers import WhisperProcessor
    # processor = WhisperProcessor.from_pretrained(model_args.pretrained_model_name_or_path, 
    #                                              lang="vi", 
    #                                              task="transcribe")

    
    from collections import defaultdict
    
    # Lưu kết quả theo metadata
    grouped_preds = {
        "region": defaultdict(list),
        "province": defaultdict(list),
        "gender": defaultdict(list),
    }
    grouped_labels = {
        "region": defaultdict(list),
        "province": defaultdict(list),
        "gender": defaultdict(list),
    }
    
    
    test_ds = dataset['test']

    print(test_ds)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
)
    test_dataloader = DataLoader(test_ds, 
                                 batch_size=eval_args.batch_size, 
                                 collate_fn=data_collator)
    
    model.eval()
    model = model.to('cuda')
    from tqdm.auto import tqdm

    from evaluate import load
    wer_metric = load("wer")
    wer_metric_global = load("wer")   # thêm metric cho toàn bộ dataset

    tokenizer = processor.tokenizer
    
    # --- trong loop ---
    for step, batch in enumerate(tqdm(test_dataloader)):

        if step == eval_args.break_step:
                break
        
        with torch.no_grad():
            generated_tokens = model.generate(
                input_features=batch["input_features"].to(model.device, dtype=model.dtype),
                return_dict_in_generate=True,
                max_new_tokens=gen_args.max_new_tokens,
            ).sequences.cpu().numpy()
    
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # print(decoded_preds)
            # print("*"*48)
            # print(decoded_labels)
            # print("---"*48)
    
            # add vào metric chung
            wer_metric_global.add_batch(predictions=decoded_preds, references=decoded_labels)
    
            # add vào nhóm theo metadata
            for i, sid in enumerate(batch["sample_id"]):
                meta = id2meta[sid]
                for key in grouped_preds.keys():
                    grouped_preds[key][meta[key]].append(decoded_preds[i])
                    grouped_labels[key][meta[key]].append(decoded_labels[i])
    
    # --- sau khi loop xong ---
    wer_global = 100 * wer_metric_global.compute()
    print("WER toàn bộ:", wer_global)
    
    # CER theo từng nhóm (như code trước)
    from evaluate import load
    
    wer_by_group = {}
    
    for meta_key in grouped_preds.keys():
        wer_by_group[meta_key] = {}
        for group_value in grouped_preds[meta_key]:
            # tạo metric riêng cho từng group
            metric_tmp = load("wer")
            metric_tmp.add_batch(
                predictions=grouped_preds[meta_key][group_value],
                references=grouped_labels[meta_key][group_value],
            )
            wer_by_group[meta_key][group_value] = 100 * metric_tmp.compute()
    
    
    print("WER by Group:\n", wer_by_group)


    import pandas as pd

    def summarize_metric(metric_by_group: dict, model_name: str = "model_1", top_n_province: int = 5):
        row = {"Model": model_name}
        
        # Region
        for k, v in metric_by_group["region"].items():
            row[f"Region_{k}"] = v
        
        # Gender
        gender_map = {1: "Male", 0: "Female"}
        for k, v in metric_by_group["gender"].items():
            row[f"Gender_{gender_map[k]}"] = v
        
        # Province: lấy top-n có WER cao nhất
        top_provinces = sorted(metric_by_group["province"].items(), key=lambda x: -x[1])[:top_n_province]
        for prov, val in top_provinces:
            row[f"Province_{prov}"] = val
        
        return pd.DataFrame([row])
    
    # Ví dụ
    df_summary = summarize_metric(wer_by_group, model_name="erax-ai__EraX-WoW-Turbo-V1.0", top_n_province=5)
    
    print(df_summary)

    
if __name__ == "__main__":
    main()