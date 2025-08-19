import torch
from omegaconf import OmegaConf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import argparse

# from src.utils.model_utils import set_torch_dtype_and_attn_implementation, get_quantization_config
# from src.utils.exp_utils import setup_environment

from src.utils.model_utils import load_model

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_audio(audio_path, target_sr=16000):
    audio_array, sr = sf.read(audio_path)
    # Resample nếu cần
    if sr != target_sr:
        import librosa
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
    model = load_model(model_args, device_args)

    if model_args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_args.adapter_path)


        model = model.merge_and_unload()

    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for transcrbing.")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config_path)

    # print(cfg)

    # Load model & processor
    model_name = cfg.model.pretrained_model_name_or_path
    processor_name = cfg.model.pretrained_processor_name_or_path or model_name

    

    # device = 'cpu' if cfg.device['use_cpu'] else 'cuda'

    processor = WhisperProcessor.from_pretrained(processor_name)
    model = load_model_for_transcribe(cfg.model, cfg.device)

    # model = model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # model.config.use_cache = False
    model.eval()

    # Load audio
    audio_array, sr = load_audio(cfg.input.audio_path)

    # Prepare input features
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(model.device, dtype=model.dtype)

    print(inputs)

    input_features = inputs["input_features"]

    print(model.device)

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("Transcription:", transcription)
if __name__ == "__main__":
    main()
