from datasets import load_dataset
from tqdm.auto import tqdm
import os



dataset_id_list = [
    "nguyendv02/ViMD_Dataset",
    'doof-ferb/LSVSC'
]
raw_data_dir_list = [
    "ViMD/raw",
    "LSVSC/raw",
]


for dataset_id, raw_data_dir in tqdm(zip(dataset_id_list, raw_data_dir_list)):
    if not os.path.exists(raw_data_dir):
        print(f"Downloading dataset {dataset_id}...")
        dataset = load_dataset(dataset_id)
        dataset.save_to_disk(raw_data_dir)
    else:
        print(f"{raw_data_dir} already exists, skipping download.")
