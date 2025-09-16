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
            "transcription": "text",
            "sentence": "text",
            "id": "sample_id"
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

from datasets import Dataset, DatasetDict

# def unify_splitnames(dataset, split_mapping=None):
#     """
#     Normalize split names in a DatasetDict based on a mapping.
    
#     Args:
#         dataset (Dataset or DatasetDict): HF dataset object
#         split_mapping (dict): mapping {old_split: new_split}
    
#     Returns:
#         DatasetDict: with renamed splits
#     """
#     if split_mapping is None:
#         split_mapping = {
#             "validation": "val",
#             "valid": "val",
#             "dev": "val",
#             "eval": "val"
#         }

#     if isinstance(dataset, Dataset):
#         # single Dataset doesn't have multiple splits
#         return dataset

#     elif isinstance(dataset, DatasetDict):
#         new_ds = DatasetDict()
#         for split, ds in dataset.items():
#             new_name = split_mapping.get(split, split)
#             if new_name in new_ds:
#                 # merge if multiple splits map to same target
#                 new_ds[new_name] = new_ds[new_name].concatenate(ds)
#             else:
#                 new_ds[new_name] = ds
#         return new_ds

#     else:
#         raise TypeError("Input must be a Dataset or DatasetDict")

from datasets import Dataset, DatasetDict, concatenate_datasets

def unify_splitnames(dataset, split_mapping=None):
    """
    Normalize split names in a DatasetDict so that all validation-like splits become 'val'.
    
    Args:
        dataset (Dataset or DatasetDict): HF dataset object
        split_mapping (dict): mapping {old_split: new_split}
    
    Returns:
        DatasetDict
    """
    if split_mapping is None:
        split_mapping = {
            "validation": "val",
            "valid": "val",
            "dev": "val",
            "eval": "val",
        }

    if isinstance(dataset, Dataset):
        return dataset

    elif isinstance(dataset, DatasetDict):
        new_ds = DatasetDict()
        for split, ds in dataset.items():
            new_name = split_mapping.get(split, split)
            if new_name in new_ds:
                new_ds[new_name] = concatenate_datasets([new_ds[new_name], ds])
            else:
                new_ds[new_name] = ds
        return new_ds

    else:
        raise TypeError("Input must be a Dataset or DatasetDict")



# def add_sample_id(dataset, col_name="sample_id"):
#     """
#     Add a sequential sample_id column to each split in a DatasetDict,
#     and move it to the first column. Skips if the column already exists.

#     Args:
#         dataset (DatasetDict): HuggingFace DatasetDict
#         col_name (str): name of the new column

#     Returns:
#         DatasetDict: new DatasetDict with sample_id column added
#     """
#     new_splits = {}
#     for split in tqdm(dataset, desc="Adding sample_id"):
#         ds_split = dataset[split]
        
#         # Skip if sample_id already exists
#         if col_name in ds_split.column_names:
#             new_splits[split] = ds_split
#             continue

#         # Create the sample_id column
#         sample_ids = list(range(len(ds_split)))
#         ds_split = ds_split.add_column(col_name, sample_ids)

#         # Reorder columns to place sample_id first
#         all_cols = ds_split.column_names
#         reordered_cols = [col_name] + [c for c in all_cols if c != col_name]
#         ds_split = ds_split.select_columns(reordered_cols)

#         new_splits[split] = ds_split

#     return DatasetDict(new_splits)


def add_sample_id(dataset, col_name="sample_id"):
    """
    Add a sequential sample_id column with optional dataset_name prefix to each split in a DatasetDict,
    and move it to the first column. Skips if the column already exists.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        col_name (str): name of the new column

    Returns:
        DatasetDict: new DatasetDict with sample_id column added
    """
    new_splits = {}
    for split in tqdm(dataset, desc="Adding sample_id"):
        ds_split = dataset[split]

        # Skip if sample_id already exists
        if col_name in ds_split.column_names:
            new_splits[split] = ds_split
            continue

        # Determine prefix from dataset_name if available
        if "dataset_name" in ds_split.column_names and len(ds_split["dataset_name"]) > 0 and ds_split["dataset_name"][0] is not None:
            prefix = f"{ds_split['dataset_name'][0]}_"
        else:
            prefix = ""

        # Create the sample_id column with optional prefix
        sample_ids = [f"{prefix}{i}" for i in range(len(ds_split))]
        ds_split = ds_split.add_column(col_name, sample_ids)

        # Reorder columns to place sample_id first
        all_cols = ds_split.column_names
        reordered_cols = [col_name] + [c for c in all_cols if c != col_name]
        ds_split = ds_split.select_columns(reordered_cols)

        new_splits[split] = ds_split

    return DatasetDict(new_splits)


import os
from datasets import DatasetDict
from tqdm import tqdm

def add_column_filename(dataset, col_audio="audio", col_name="filename", prefix=None):
    """
    Add a 'filename' column to each split in a DatasetDict.
    - If audio has valid paths, use basenames.
    - If no paths exist at all, generate synthetic IDs
      (00001.wav, sample_00001.wav, etc.).
      Padding width auto-adjusts to dataset size.
    Skips adding if the column already exists.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        col_audio (str): name of the audio column
        col_name (str): name of the new column
        prefix (str|None): optional prefix for synthetic filenames.
                           If None, use plain zero-padded numbers.

    Returns:
        DatasetDict: with new 'filename' column
    """
    new_splits = {}
    for split in tqdm(dataset, desc="Adding filename"):
        dset = dataset[split]

        if col_name in dset.column_names:
            new_splits[split] = dset
            continue

        # Peek at the first example to see if path exists
        first_ex = dset[0][col_audio]
        if isinstance(first_ex, dict):
            has_path = bool(first_ex.get("path"))
        else:
            has_path = bool(dset.features[col_audio].decode_example(first_ex).get("path"))

        if not has_path:
            # No path info: assign synthetic IDs with dynamic zero-padding
            width = len(str(len(dset)))
            if prefix is None:
                filenames = [f"{i:0{width}d}.wav" for i in range(len(dset))]
            else:
                filenames = [f"{prefix}_{i:0{width}d}.wav" for i in range(len(dset))]
        else:
            # Path info exists: extract basenames
            filenames = []
            for ex in dset:
                audio_val = ex[col_audio]
                if isinstance(audio_val, dict):
                    path = audio_val.get("path", None)
                else:
                    path = dset.features[col_audio].decode_example(audio_val).get("path", None)
                filenames.append(os.path.basename(path) if path else "")

        new_splits[split] = dset.add_column(col_name, filenames)

    return DatasetDict(new_splits)


from datasets import DatasetDict
from tqdm import tqdm

def add_column_datasetname(dataset, ds_name, add_col_dsname_batch_size, col_name="dataset_name"):
    """
    Add a dataset_name column to each split in a DatasetDict.
    - Skips if the column already exists.
    - Uses batched map for efficiency.
    
    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        ds_name (str|None): dataset name to fill
        col_name (str): column name to add (default "dataset_name")

    Returns:
        DatasetDict: with dataset_name column added
    """
    if ds_name is None:
        return dataset  # skip if no name

    new_splits = {}
    for split in tqdm(dataset, desc="Adding dataset_name"):
        dset = dataset[split]

        if col_name in dset.column_names:
            new_splits[split] = dset
            continue

        dset = dset.map(
            lambda batch: {col_name: [ds_name] * len(batch[col_name if col_name in batch else next(iter(batch))])},
            batched=True,
            batch_size=add_col_dsname_batch_size,
            desc=f"Adding {col_name} to {split}"
        )

        new_splits[split] = dset

    return DatasetDict(new_splits)

# from datasets import DatasetDict
# from tqdm import tqdm

# def add_column_datasetname(dataset, ds_name, col_name="dataset_name"):
#     """
#     Add a dataset_name column to each split in a DatasetDict.
#     Uses add_column (fast) instead of map.
#     """
#     if ds_name is None:
#         return dataset

#     new_splits = {}
#     for split in tqdm(dataset, desc="Adding dataset_name"):
#         dset = dataset[split]

#         if col_name in dset.column_names:
#             new_splits[split] = dset
#             continue

#         # Precompute the values just once
#         values = [ds_name] * len(dset)
#         new_splits[split] = dset.add_column(col_name, values)

#     return DatasetDict(new_splits)


def get_sid2meta(dataset, 
                 fields=("filename", "dialect", "province_name", "gender"), 
                 splits=None):
    """
    Build id2meta dictionary from dataset, grouped by split.

    Returns:
        dict: {split: {sample_id: {field: value, ...}}}
    """
    splits = list(splits or dataset.keys())

    dataset_fields = set(dataset[splits[0]].column_names)
    valid_fields = [f for f in fields if f in dataset_fields]

    print("dataset_fields:", dataset_fields)

    missing_fields = set(fields) - dataset_fields
    if missing_fields:
        print(f"These fields are missing in dataset and will be ignored: {missing_fields}")

    sid2meta = {}
    for split in splits:
        split_meta = {}
        for ex in tqdm(dataset[split], desc=f"Processing {split}"):
            meta = {field: ex[field] for field in valid_fields}
            split_meta[ex["sample_id"]] = meta
        sid2meta[split] = split_meta

    return sid2meta
def get_sid2meta(dataset, 
                 fields=("filename", "dialect", "province_name", "gender"), 
                 splits=None):
    """
    Build id2meta dictionary from dataset, grouped by split.
    Handles splits with different schemas.

    Returns:
        dict: {split: {sample_id: {field: value, ...}}}
    """
    splits = list(splits or dataset.keys())

    sid2meta = {}
    for split in splits:
        ds_fields = set(dataset[split].column_names)
        valid_fields = [f for f in fields if f in ds_fields]

        missing_fields = set(fields) - ds_fields
        if missing_fields:
            print(f"Split '{split}' missing fields: {missing_fields} -> ignored")

        split_meta = {}
        for ex in tqdm(dataset[split], desc=f"Processing {split}"):
            meta = {field: ex[field] for field in valid_fields}
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

def show_sample_by_filename(filename: str, 
                            split_dataset, 
                            split_filename2sid: dict = None):
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
        # Look up the sample_id based on the filename
        sample_id = split_filename2sid.get(filename)
        if sample_id is None:
            raise ValueError(f"Filename '{filename}' not found in mapping")
        
    # Find the sample index by its filename
    idx = next((i for i, s in enumerate(split_dataset) if s['filename'] == filename), None)

    
    if idx is None:
        raise ValueError(f"Filename '{filename}' not found in dataset")
    
    # Get the sample using the found index
    sample = split_dataset[idx]
    
    # Play audio
    waveform = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    IPython.display.display(IPython.display.Audio(waveform, rate=sr))
    
    # Show metadata
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


