from faster_whisper import WhisperModel

model = WhisperModel("/home/nampv1/projects/asr/asr-deployment-app/models/ct2/openai-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")

segments, info = model.transcribe("/home/nampv1/projects/asr/asr_ft/data/examples/example_vietbud500_02.wav")
for seg in segments:
    print(f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}")