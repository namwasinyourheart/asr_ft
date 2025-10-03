import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import os
from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()

login(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])

dataset_id_list = [
    # "nguyendv02/ViMD_Dataset",
    # "doof-ferb/LSVSC",
    # "linhtran92/viet_bud500"
    # "doof-ferb/vlsp2020_vinai_100h"
    # "google/fleurs",
    # "doof-ferb/fpt_fosd",
    # "doof-ferb/infore1_25hours",


]
raw_data_dir_list = [
    # "ViMD/raw",
    # "LSVSC/raw",
    # "VietBud500/raw"
    # "ASR-VLSP2020-VINAI-100H/raw/hf"
    # "ASR-Fleurs/raw/hf",
    # "ASR-FPT_FOSD/raw/hf",
    # "ASR-INFOR1_25hours/raw/hf",

]

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True,
                    help="Root directory where raw data directories will be stored")
args = parser.parse_args()

for dataset_id, raw_data_dir in tqdm(zip(dataset_id_list, raw_data_dir_list)):
    full_path = os.path.join(args.data_root, raw_data_dir)
    if not os.path.exists(full_path):
        print(f"Downloading dataset {dataset_id}...")
        if dataset_id == "google/fleurs":   
            dataset = load_dataset(dataset_id, "vi_vn", trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_id, trust_remote_code=True)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        dataset.save_to_disk(full_path)
    else:
        print(f"{full_path} already exists, skipping download.")
