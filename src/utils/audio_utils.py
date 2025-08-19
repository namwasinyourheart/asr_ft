from IPython.display import Audio, display
import torch
import numpy as np
import librosa
import io

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
