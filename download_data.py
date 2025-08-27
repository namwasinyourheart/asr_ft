import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import os

dataset_id_list = [
    "nguyendv02/ViMD_Dataset",
    "doof-ferb/LSVSC",
]
raw_data_dir_list = [
    "ViMD/raw",
    "LSVSC/raw",
]

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True,
                    help="Root directory where raw data directories will be stored")
args = parser.parse_args()

for dataset_id, raw_data_dir in tqdm(zip(dataset_id_list, raw_data_dir_list)):
    full_path = os.path.join(args.data_root, raw_data_dir)
    if not os.path.exists(full_path):
        print(f"Downloading dataset {dataset_id}...")
        dataset = load_dataset(dataset_id)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        dataset.save_to_disk(full_path)
    else:
        print(f"{full_path} already exists, skipping download.")
