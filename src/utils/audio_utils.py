from IPython.display import Audio, display
import torch
import numpy as np
import librosa
import io


def show_sample_by_filename(filename: str, split_ds, split_filename2sid: dict = None):
    """
    Display audio and metadata for a sample given its filename.
    
    Args:
        filename (str): audio filename
        split_ds: HF Dataset split (train/test/valid)
        split_filename2sid (dict, optional): mapping filename -> sample_id. 
            If None, searches the dataset linearly.
    """
    # Get sample index
    if split_filename2sid is not None:
        sample_id = split_filename2sid.get(filename)
        if sample_id is None:
            raise ValueError(f"Filename '{filename}' not found in mapping")
        idx = int(sample_id)
    else:
        # fallback: search linearly
        idx = next((i for i, s in enumerate(split_ds) if s['filename'] == filename), None)
        if idx is None:
            raise ValueError(f"Filename '{filename}' not found in dataset")
    
    sample = split_ds[idx]
    
    # Play audio
    waveform = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    display(Audio(waveform, rate=sr))
    
    # Show metadata
    print("Metadata:")
    for key, value in sample.items():
        if key != 'audio':
            print(f"{key}: {value}")


def listen_audio_by_filename(filename: str, 
                             dataset_split, 
                             split_filename2sid: dict):
    """
    Listen to audio from a filename using a HF Dataset split and a filename->sample_id mapping.

    Args:
        filename (str): the audio filename
        dataset_split: dataset['train'] / ['test'] / ['valid']
        split_filename2sid (dict): mapping filename -> sample_id
    """
    # Get the sample ID
    sample_id = split_filename2sid.get(filename)
    if sample_id is None:
        raise ValueError(f"Filename '{filename}' not found in mapping")

    # sample_id from id2meta is usually a string; convert to int for Dataset indexing
    idx = int(sample_id)

    # Load waveform and sampling rate
    sample = dataset_split[idx]
    waveform = sample['audio']['array']
    sr = sample['audio']['sampling_rate']

    # Play audio
    display(Audio(waveform, rate=sr))




def listen_audio(waveform, sr=16000):
    """
    Nghe audio từ waveform tensor hoặc numpy array trong notebook.
    
    Args:
        waveform: torch.Tensor 1D/2D hoặc np.ndarray
        sr: int, sampling rate
    """
    
    
    # Nếu tensor, chuyển sang numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    # Nếu 2D (stereo), transpose về (num_samples, num_channels)
    if waveform.ndim == 2:
        waveform = waveform.T  # [num_samples, num_channels]
    
    display(Audio(waveform, rate=sr))


def load_audio_from_bytes(data: bytes, target_sr: int = 16000):
    y, sr = librosa.load(io.BytesIO(data), sr=target_sr, mono=True)
    return y.astype(np.float32), sr


# import librosa
# import io
# import soundfile as sf

# audio_bytes = df_train.iloc[0]['audio']['bytes']
# sr = df_train.iloc[0]['audio'].get('sampling_rate', 16000)

# # Chuyển bytes thành file-like object
# audio_buffer = io.BytesIO(audio_bytes)

# # Đọc audio
# waveform, sample_rate = sf.read(audio_buffer)


# # nếu muốn mono
# import numpy as np
# waveform_mono = np.mean(waveform, axis=1)
# # waveform là numpy array
# print(waveform.shape, sample_rate)
