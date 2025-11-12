TEXT_POSTPROCESSING_URL_ENDPOINT = "http://0.0.0.0:13081/asr/v1/postprocess_text"


MODEL_LOCAL_PATHS = [
    "/home/nampv1/.cache/huggingface/hub/models--suzii--vi-whisper-large-v3-turbo-v1/snapshots/dfe4ff961e32e44fd7525a2e71c1bca6abafc471/" # suzii/vi-whisper-large-v3-turbo-v1
]



DATA_LOCAL_PATHS = {
    "LSVSC": "/media/nampv1/hdd/data/vnpost-asr/LSVSC",
    "VietBud500": "/media/nampv1/hdd/data/vnpost-asr/VietBud500",
    "VIVOS": "/media/nampv1/hdd/data/VIVOS",
    "ViMD": "/media/nampv1/hdd/data/ASR-ViMD-100h",
    "VLSP2020": "/media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H",
    "TTS-viVoice-1017h": "/media/nampv1/hdd/data/TTS-viVoice-1017h",
    "INFORE1": "/media/nampv1/hdd/data/ASR-INFORE1-25h",
    "FPT_FOSD": "/media/nampv1/hdd/data/ASR-FPT_FOSD",
}