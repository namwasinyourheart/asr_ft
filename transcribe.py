import os
import requests
import soundfile as sf
import io
import librosa
from typing import List

from hydra import initialize, compose

from transformers import (
    set_seed
)

from transformers.audio_utils import load_audio_as
from qwen_asr import Qwen3ASRModel

from hydra import initialize, compose
from omegaconf import OmegaConf

from src.utils.exp_utils import setup_environment
from src.utils.model_utils import (
    load_whisper_model, load_whisper_processor, load_model, load_processor,
    load_qwen3_asr_model
)
from src.utils.api_utils import postprocess_text_via_api

from transformers import logging
logging.set_verbosity_error()



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

def load_config(config_path: str, override_args: List[str] = None):
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

    return cfg



def load_audio(audio_path, target_sr=16000):
    # Nếu là URL thì tải về trước
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        response = requests.get(audio_path)
        response.raise_for_status()
        data = io.BytesIO(response.content)
        audio_array, sr = sf.read(data)
    else:
        audio_array, sr = sf.read(audio_path)

    # Resample nếu cần
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Nếu stereo, lấy kênh đầu tiên
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0]

    return audio_array, sr


def load_model_for_transcribe(model_args, device_args):
    """
    Load model for transcription, supporting multiple adapter merges.

    Args:
    - model_args (dict): Model arguments, including `pretrained_model_name_or_path` and optional `adapter_path`.
      adapter_path can be a string or a list of strings.
    - device_args (dict): Device arguments, including `use_cpu` and optional `device_map`.

    Returns:
    - model (PreTrainedModel): The loaded model.
    """
    # model = load_whisper_model(model_args, device_args)

    model = load_model(model_args, device_args)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if model_args.adapter_paths:
        print("Merging adapters...")
        from peft import PeftModel
        adapter_paths = model_args.adapter_paths
        if isinstance(adapter_paths, str):
            adapter_paths = [adapter_paths]  # convert to list if single path

        for path in adapter_paths:
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()  # merge current adapter into base model

    return model

def generate_transcript(model_args, input_args, gen_args, device_args):
    if model_args['architecture'] == 'Qwen3ASRModel':
        model = load_qwen3_asr_model(model_args, device_args)

        result = model.transcribe(
            audio=input_args.audio_path,
            language=["Vietnamese"], # can also be set to None for automatic language detection
            return_time_stamps=True,
        )

        transcription = result.text
    
    else:
        model = load_model_for_transcribe(model_args, device_args)

        processor = load_processor(model_args)

        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        # model.config.use_cache = False
        model.eval()

        # Load audio
        audio_array, sr = load_audio(input_args.audio_path)

        # Prepare input features
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(model.device, dtype=model.dtype)

        # print(inputs)

        input_features = inputs["input_features"]

        # print(f"Device: {model.device}")

        # Generate transcription
        predicted_ids = model.generate(input_features, **gen_args.gen_args)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    if gen_args['do_postprocess_text']:
        transcription = postprocess_text_via_api(transcription)

    return transcription

        


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for generating.")

    args, override_args = parser.parse_known_args()
    return args, override_args


def main():

    # Setup environment
    setup_environment()

    # Parse arguments
    args, override_args = parse_args()

    # Load the generation config file
    cfg = load_config(args.config_path, override_args)

    print(OmegaConf.to_yaml(cfg))


    model_args = cfg.model
    gen_args = cfg.generate
    device_args = cfg.device
    input_args = cfg.input

    # Set seed
    set_seed(gen_args.seed)


    # model = load_model_for_transcribe(model_args, device_args)
    # print(f"Model: {model_args.pretrained_model_name_or_path}")
    # processor = load_processor(model_args)

    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    # # model.config.use_cache = False
    # model.eval()

    # # Load audio
    # audio_array, sr = load_audio(input_args.audio_path)

    # # Prepare input features
    # inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(model.device, dtype=model.dtype)

    # # print(inputs)

    # input_features = inputs["input_features"]

    # # print(f"Device: {model.device}")

    # # Generate transcription
    # predicted_ids = model.generate(input_features, **gen_args.gen_args)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    # transcription = postprocess_text_via_api(transcription)

    transcription = generate_transcript(model_args, input_args, gen_args, device_args)
    print("Transcription:", transcription)
    

if __name__ == "__main__":
    main()
    