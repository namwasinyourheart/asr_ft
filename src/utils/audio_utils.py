import os
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import Audio, Dataset, DatasetDict
import librosa

def disable_audio_decode(dataset, col="audio"):
    """
    Cast the audio column to lazy decoding (AudioDecoder).
    Returns AudioDecoder objects instead of dicts with array/path.
    """
    return dataset.cast_column(col, Audio(decode=False))

def enable_audio_decode(dataset, col="audio"):
    """
    Cast the audio column to decoded mode.
    Returns dicts with keys: 'path', 'array', 'sampling_rate'.
    """
    return dataset.cast_column(col, Audio(decode=True))


def unify_colnames(dataset, col_mapping=None):
    """
    Normalize column names in a Dataset or DatasetDict based on a mapping.
    
    Args:
        dataset (Dataset or DatasetDict): HF dataset object
        col_mapping (dict): mapping {old_name: new_name}
    
    Returns:
        Dataset or DatasetDict: with renamed columns
    """
    if col_mapping is None:
        col_mapping = {
            "region": "dialect",
            "dialect": "dialect",
            "gender": "gender",
        }
    
    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split: ds.rename_columns({k: v for k, v in col_mapping.items() if k in ds.column_names})
            for split, ds in dataset.items()
        })
    elif isinstance(dataset, Dataset):
        return dataset.rename_columns({k: v for k, v in col_mapping.items() if k in dataset.column_names})
    else:
        raise TypeError("Input must be a Dataset or DatasetDict")

def add_sample_id(dataset, col_name="sample_id"):
    """
    Add a sequential sample_id column to each split in a DatasetDict,
    and move it to the first column.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        col_name (str): name of the new column

    Returns:
        DatasetDict: new DatasetDict with sample_id column added
    """
    new_splits = {}
    for split in tqdm(dataset, desc="Adding sample_id"):
        sample_ids = list(range(len(dataset[split])))
        ds_split = dataset[split].add_column(col_name, sample_ids)

        # ép sample_id ra cột đầu tiên
        all_cols = ds_split.column_names
        reordered_cols = [col_name] + [c for c in all_cols if c != col_name]
        ds_split = ds_split.select_columns(reordered_cols)

        new_splits[split] = ds_split

    return DatasetDict(new_splits)

def add_column_filename(dataset, col_audio="audio", col_name="filename"):
    """
    Add a 'filename' column to each split in a DatasetDict,
    extracting the basename from the audio path.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        col_audio (str): name of the audio column
        col_name (str): name of the new column

    Returns:
        DatasetDict: with new 'filename' column
    """
    new_splits = {}
    for split in tqdm(dataset, desc="Adding filename"):
        filenames = []
        for ex in dataset[split]:
            audio_val = ex[col_audio]
            # audio_val có thể là dict (decode=True) hoặc AudioDecoder (decode=False)
            if isinstance(audio_val, dict):
                path = audio_val.get("path", "")
            else:
                # fallback decode nếu chưa cast_column decode
                path = dataset[split].features[col_audio].decode_example(audio_val).get("path", "")
            filenames.append(os.path.basename(path) if path else "")
        new_splits[split] = dataset[split].add_column(col_name, filenames)
    return DatasetDict(new_splits)

def get_sid2meta(dataset, 
                 fields=("filename", "region", "province_name", "gender"), 
                 splits=None):
    """
    Build id2meta dictionary from dataset, grouped by split.

    Returns:
        dict: {split: {sample_id: {field: value, ...}}}
    """
    splits = splits or dataset.keys()
    sid2meta = {}

    for split in splits:
        split_meta = {}
        for ex in tqdm(dataset[split], desc=f"Processing {split}"):
            meta = {field: ex.get(field, "") for field in fields}
            split_meta[ex["sample_id"]] = meta
        sid2meta[split] = split_meta

    return sid2meta

def get_filename2sid(dataset, filename_colname="filename", sid_colname="sample_id", splits=None):
    """
    Build a mapping from filename to sample_id, grouped by split.

    Args:
        dataset: DatasetDict
        filename_colname (str): name of the column containing filenames
        sid_colname (str): name of the column containing sample IDs
        splits: list of splits to process (None = all splits)

    Returns:
        dict: {split: {filename: sample_id, ...}, ...}
    """
    splits = splits or dataset.keys()
    mapping = {}

    for split in splits:
        split_map = {}
        for ex in tqdm(dataset[split], desc=f"Processing {split}"):
            fname = ex.get(filename_colname, "")
            sid = ex.get(sid_colname)
            if fname:
                split_map[fname] = sid
        mapping[split] = split_map

    return mapping



def listen_audio_by_filename(filename: str, 
                             split_dataset, 
                             split_filename2sid: dict):
    """
    Listen to audio from a filename using a HF Dataset split and a filename->sample_id mapping.

    Args:
        filename (str): the audio filename
        split_dataset: dataset['train'] / ['test'] / ['valid']
        split_filename2sid (dict): mapping filename -> sample_id
    """
    import IPython

    split_dataset = enable_audio_decode(split_dataset)
    
    # Get the sample ID
    sample_id = split_filename2sid.get(filename)
    if sample_id is None:
        raise ValueError(f"Filename '{filename}' not found in mapping")

    # sample_id from id2meta is usually a string; convert to int for Dataset indexing
    idx = int(sample_id)

    print(idx)

    # Load waveform and sampling rate
    sample = split_dataset[idx]
    waveform = sample['audio']['array']
    sr = sample['audio']['sampling_rate']

    # Play audio
    IPython.display.display(IPython.display.Audio(waveform, rate=sr))


def show_sample_by_filename(filename: str, split_dataset, split_filename2sid: dict = None):
    """
    Display audio and metadata for a sample given its filename.
    
    Args:
        filename (str): audio filename
        split_dataset: HF Dataset split (train/test/valid)
        split_filename2sid (dict, optional): mapping filename -> sample_id. 
            If None, searches the dataset linearly.
    """
    import IPython
    
    split_dataset = enable_audio_decode(split_dataset)
    
    # Get sample index
    if split_filename2sid is not None:
        sample_id = split_filename2sid.get(filename)
        if sample_id is None:
            raise ValueError(f"Filename '{filename}' not found in mapping")
        idx = int(sample_id)
    else:
        # fallback: search linearly
        idx = next((i for i, s in enumerate(split_dataset) if s['filename'] == filename), None)
        if idx is None:
            raise ValueError(f"Filename '{filename}' not found in dataset")
    
    sample = split_dataset[idx]
    
    # Play audio
    waveform = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    IPython.display.display(IPython.display.Audio(waveform, rate=sr))
    
    # Show metadata
    # print("Metadata:")
    for key, value in sample.items():
        if key != 'audio':
            print(f"{key}: {value}")


def listen_audio(waveform, sr=16000):
    """
    Nghe audio từ waveform tensor hoặc numpy array trong notebook.
    
    Args:
        waveform: torch.Tensor 1D/2D hoặc np.ndarray
        sr: int, sampling rate
    """

    from IPython.display import Audio, display
    
    
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



