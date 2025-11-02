import os
import csv

# ===== Cấu hình =====
BASE_DIR = "/media/nampv1/hdd/data/vn_commune_district_province/"
TTS_DIR = os.path.join(BASE_DIR, "tts_generated")

# Danh sách provider / model / voice
structure = {
    "providerA": {
        "modelX": ["voice_1", "voice_2"]
    },
    "providerB": {
        "modelY": ["voice_1"]
    }
}

# ===== Hàm tạo thư mục và file =====
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_metadata_file(path):
    csv_path = os.path.join(path, "metadata.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "utt_id", "text", "audio_path",
                "provider", "model", "voice",
                "sample_rate", "lang", "duration", "gen_date"
            ])
        print(f"✅ Created {csv_path}")
    else:
        print(f"⚠️  Already exists: {csv_path}")

# ===== Tạo cây thư mục =====
for provider, models in structure.items():
    for model, voices in models.items():
        for voice in voices:
            wav_dir = os.path.join(TTS_DIR, provider, model, voice, "wav")
            ensure_dir(wav_dir)
            create_metadata_file(os.path.dirname(wav_dir))

print("\n🎯 Done: Directory structure initialized.")
