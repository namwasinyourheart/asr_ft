models_list = [
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "vinai/phowhisper-large",
    "erax-ai/EraX-WoW-Turbo-V1.0"
]


from huggingface_hub import snapshot_download
from tqdm import tqdm

for model in tqdm(models_list):

    print(f"Download model {model}...")

    snapshot_dir = snapshot_download(
        repo_id=model,
        revision="main",      
        # cache_dir="/mnt/models-vol/huggingface"
    )
