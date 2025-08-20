from omegaconf import OmegaConf

import argparse


from transformers import (
    set_seed
)
from src.utils.exp_utils import setup_environment
from src.utils.model_utils import load_whisper_model, load_processor

import requests
import soundfile as sf
import io
import librosa



def load_config(config_path):
    return OmegaConf.load(config_path)


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
    Load model for transcription.

    Args:
    - model_args (dict): Model arguments, including `pretrained_model_name_or_path` and optional `adapter_path`.
    - device_args (dict): Device arguments, including `use_cpu` and optional `device_map`.

    Returns:
    - model (PreTrainedModel): The loaded model.
    """
    model = load_whisper_model(model_args, device_args)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    if model_args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_args.adapter_path)


        # model = model.merge_and_unload()

    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for transcrbing.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the generation config file
    cfg = load_config(args.config_path)

    # Setup environment
    setup_environment()


    # print(OmegaConf.to_yaml(cfg))

    model_args = cfg.model
    gen_args = cfg.generate
    device_args = cfg.device

    # Set seed
    set_seed(gen_args.seed)


    model = load_model_for_transcribe(model_args, device_args)
    processor = load_processor(model_args)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # model.config.use_cache = False
    model.eval()

    # Load audio
    audio_array, sr = load_audio(cfg.input.audio_path)

    # Prepare input features
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(model.device, dtype=model.dtype)

    # print(inputs)

    input_features = inputs["input_features"]

    print(model.device)

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("Transcription:", transcription)
    

if __name__ == "__main__":
    main()
    