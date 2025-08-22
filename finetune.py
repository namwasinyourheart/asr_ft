import os
import joblib
import argparse
from functools import partial

import json

import warnings

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader

from hydra import initialize, compose
from hydra.utils import instantiate

from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils import load_whisper_model, load_processor

from src.utils.exp_utils import setup_environment, create_exp_dir
from src.metrics import compute_metrics_wrapper

from src.utils.data_utils import get_data_subset

warnings.filterwarnings("ignore")

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from prepare_data import prepare_data

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        batch["input_features"] = batch["input_features"].to(dtype=torch.float16)

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

        # print("type(features): ", type(features))
        # print("features[0].keys(): ", features[0].keys())

        # batch["filename"] = [f["filename"] for f in features]
        # batch["sample_id"] = [f["sample_id"] for f in features]

        # return batch

        # giữ riêng extra keys, nhưng không add vào batch trả về cho model
        # extra = {
        #     "filename": [f["filename"] for f in features],
        #     "sample_id": [f["sample_id"] for f in features],
        # }

        # batch["extra"] = extra   # 👈 giữ trong dict con, Trainer sẽ không pass vào model
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



def finetune(
    model, processor, 
    train_ds, val_ds, test_ds,
    exp_args, data_args, 
    model_args, train_args, eval_args, gen_args, device_args, 
    exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir
):
    """
    Trains and evaluates the provided model using the specified datasets and configurations.

    This function fine-tunes a preloaded model on a given training dataset, evaluates it on 
    a validation set, and optionally performs predictions on a test set. It supports various 
    configurations for model training, evaluation, and generation while handling logging, 
    checkpointing, and result saving.

    Args:
        model: The preloaded model to be fine-tuned.
        processor: The processor associated with the model.
        train_ds: The dataset used for training.
        val_ds: The dataset used for validation.
        test_ds: The dataset used for testing (if applicable).
        exp_args: Experiment-related arguments and configurations.
        data_args: Parameters related to dataset processing and handling.
        # tokenizer_args: Arguments for tokenizer configuration.
        # prompt_args: Configuration for prompt engineering or formatting.
        model_args: Model-specific settings and hyperparameters.
        train_args: Training-related parameters, including optimizer settings.
        eval_args: Evaluation-related configurations and metrics.
        gen_args: Arguments for text generation (if applicable).
        device_args: Hardware-related configurations (e.g., GPU, TPU settings).
        exp_variant_dir: Root directory for experiment outputs.
        exp_variant_data_dir: Directory for storing dataset files.
        exp_variant_checkpoints_dir: Path to save model checkpoints.
        exp_variant_results_dir: Directory for saving evaluation results and predictions.
    """

    if train_args.use_peft:
        # Prepare model for training
        model.gradient_checkpointing_enable()
        if model_args.load_in_4bit or model_args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        peft_config = get_peft_config(train_args)
        if exp_args.print_peft_config:
            print(peft_config)
        model = get_peft_model(model, peft_config)


    # Print trainable parameters
    if exp_args.print_trainable_parameters:
        try:
            model.print_trainable_parameters()
        except:
            from src.utils.model_utils import print_trainable_parameters
            print_trainable_parameters(model)

    # Print parameter datatypes
    if exp_args.print_parameter_datatypes:
        from src.utils.model_utils import print_parameter_datatypes
        print_parameter_datatypes(model)
        
    import wandb
    # Initialize W&B
    wandb.init(
        project=exp_args.wandb.project,
    )

    # Training arguments
    training_args = instantiate(
        train_args.train_args, 
        output_dir=exp_variant_checkpoints_dir, 
        report_to="wandb",
        run_name=wandb.run.name
    )

    # Log device info if required
    if exp_args.print_device:
        print(
            f"Process Rank: {training_args.local_rank}, Device: {training_args.device}, N_GPU: {training_args.n_gpu}, "
            f"Distributed Training: {bool(training_args.local_rank != -1)}"
        )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    compute_metrics = compute_metrics_wrapper(processor.tokenizer, 
                                              train_args.eval_metrics, 
                                              model_args.model_type)
    
    from transformers import Seq2SeqTrainer
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    all_metrics = {
        "exp_name": exp_args.exp_name,
        "exp_variant": exp_args.exp_variant,
        "run_name": wandb.run.name
    }

    if training_args.do_train:
        if train_args.do_resume_from_checkpoint and training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)

        if train_args.use_peft:
            trainer.model.save_pretrained(os.path.join(exp_variant_results_dir, 'adapter'))
            print(f"Adapter saved to {os.path.join(exp_variant_results_dir, 'adapter')}")

        else:
            print("Saving finetuned model and processor...")
            trainer.model.save_pretrained(os.path.join(exp_variant_results_dir, 'finetuned_model'))
            processor.save_pretrained(os.path.join(exp_variant_results_dir, 'finetuned_model'))
            print(f"Finetuned model and processor saved to {os.path.join(exp_variant_results_dir, 'finetuned_model')}")

    if training_args.do_eval:
        print("Evaluating...")
        metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)


    if training_args.do_predict:
        pass
        
    # Save metrics
    if training_args.do_train or training_args.do_eval or training_args.do_predict:
        all_metrics_fp = os.path.join(exp_variant_results_dir, train_args.train_metric_filename)
        with open(all_metrics_fp, "w") as fout:
            fout.write(json.dumps(all_metrics))

    # Merge model if required
    if train_args.use_peft:
        if training_args.do_train and train_args.merge_after_train:
            pass

    # Log experiment artifact
    if exp_args.wandb.log_artifact:
        pass

def main():
    setup_environment()

    # Parse arguments
    args, override_args = parse_args()

    # Load configuration
    cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args = load_cfg(args.config_path, override_args)

    # Print configuration
    if exp_args.print_cfg:
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

    # Load dataset
    dataset, id2meta = prepare_data(exp_args, data_args, model_args, device_args)

    train_ds, val_ds, test_ds = dataset['train'], dataset['valid'], dataset['test']

    if train_args.train_n_samples:
        train_ds = get_data_subset(train_args.train_n_samples, train_ds, exp_args.seed)

    if train_args.val_n_samples:
        val_ds = get_data_subset(train_args.val_n_samples, val_ds, exp_args.seed)

    if train_args.test_n_samples:
        test_ds = get_data_subset(train_args.test_n_samples, test_ds, exp_args.seed)


    # Loading model and processor
    model = load_whisper_model(model_args, device_args)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # if model_args.adapter_path:
    #     from peft import PeftModel
    #     model = PeftModel.from_pretrained(model, model_args.adapter_path)
    
    processor = load_processor(model_args)
    

    # Print model and tokenizer
    if exp_args.print_model:
        print(model)

    if exp_args.print_processor:
        print(processor)

    finetune(
        model, processor, 
        train_ds, val_ds, test_ds,
        exp_args, data_args, 
        model_args, train_args, eval_args, gen_args, device_args, 
        exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir
    )
        

if __name__ == "__main__":
    main()
    