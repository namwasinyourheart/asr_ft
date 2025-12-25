python explore_data/analyze_transcripts.py \
--hf_raw_data_dir /media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H/raw/hf \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H/eda \
--log_file /media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H/eda/eda_run.log


python explore_data/analyze_audios.py \
--hf_raw_data_dir /media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H/raw/hf \
--load_from_disk \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H/eda \
--log_file /media/nampv1/hdd/data/ASR-VLSP2020-VINAI-100H/eda/eda_run.log



python explore_data/analyze_transcripts.py \
--hf_raw_data_dir /media/nampv1/hdd/data/TTS-viVoice-1017h/raw/hf/snapshot/data \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/TTS-viVoice-1017h/eda \
--log_file /media/nampv1/hdd/data/TTS-viVoice-1017h/eda/eda_run.log





# VIVOS
python explore_data/analyze_transcripts.py \
--hf_raw_data_dir /media/nampv1/hdd/data/VIVOS/raw/hf \
--text_column text \
--output_dir /media/nampv1/hdd/data/VIVOS/eda \
--log_file /media/nampv1/hdd/data/VIVOS/eda/eda_run.log

python explore_data/analyze_audios.py \
--hf_raw_data_dir /media/nampv1/hdd/data/VIVOS/raw/hf \/media/nampv1/hdd/data/VIVOS/raw/hf
--load_from_disk \
--text_column text \
--output_dir /media/nampv1/hdd/data/VIVOS/eda \
--log_file /media/nampv1/hdd/data/VIVOS/eda/eda_run.log



# VietBud500
python explore_data/analyze_transcripts.py \
--hf_raw_data_dir /media/nampv1/hdd/data/vnpost-asr/VietBud500/raw/VietBud500/raw \
--load_from_disk \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/vnpost-asr/VietBud500/eda \
--log_file /media/nampv1/hdd/data/vnpost-asr/VietBud500/eda/eda_run.log


python explore_data/analyze_audios.py \
--hf_raw_data_dir /media/nampv1/hdd/data/vnpost-asr/VietBud500/raw/VietBud500/raw \
--load_from_disk \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/vnpost-asr/VietBud500/eda \
--log_file /media/nampv1/hdd/data/vnpost-asr/VietBud500/eda/eda_run.log


# LSVSC
python explore_data/analyze_transcripts.py \
--hf_raw_data_dir /media/nampv1/hdd/data/vnpost-asr/LSVSC/raw/ \
--load_from_disk \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/vnpost-asr/LSVSC/eda \
--log_file /media/nampv1/hdd/data/vnpost-asr/LSVSC/eda/eda_run.log


python explore_data/analyze_audios.py \
--hf_raw_data_dir /media/nampv1/hdd/data/vnpost-asr/LSVSC/raw/ \
--load_from_disk \
--text_column transcription \
--output_dir /media/nampv1/hdd/data/vnpost-asr/LSVSC/eda \
--log_file /media/nampv1/hdd/data/vnpost-asr/LSVSC/eda/eda_run.log


# TTS-viVoice
python explore_data/analyze_transcripts.py \
--hf_raw_data_dir /media/nampv1/hdd/data/TTS-viVoice-1017h/raw/hf/ \
--load_from_disk \
--text_column text \
--output_dir /media/nampv1/hdd/data/TTS-viVoice-1017h/eda \
--log_file /media/nampv1/hdd/data/TTS-viVoice-1017h/eda/eda_run.log

python explore_data/analyze_audios.py \
--hf_raw_data_dir /media/nampv1/hdd/data/TTS-viVoice-1017h/raw/hf/ \
--load_from_disk \
--text_column text \
--output_dir /media/nampv1/hdd/data/TTS-viVoice-1017h/eda \
--log_file /media/nampv1/hdd/data/TTS-viVoice-1017h/eda/eda_run.log



# VoA

python /home/nampv1/projects/asr/asr_ft/explore_data/analyze_audios.py \
--hfds_script /home/nampv1/projects/asr/asr_ft/prepare_data/voa_hfds/voa.py \
--local_raw_data_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated \
--subset_name gtts__default__vi \
--text_column text

python /home/nampv1/projects/asr/asr_ft/explore_data/analyze_transcripts.py \
--hfds_script /home/nampv1/projects/asr/asr_ft/prepare_data/voa_hfds/voa.py \
--local_raw_data_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated \
--subset_name gtts__default__vi \
--text_column text
# --output_dir /media/nampv1/hdd/data/Voice-of-Address/eda \
# --log_file /media/nampv1/hdd/data/Voice-of-Address/eda/eda_run.log

python /home/nampv1/projects/asr/asr_ft/explore_data/analyze_transcripts.py \
--hfds_script /home/nampv1/projects/asr/asr_ft/prepare_data/voa_hfds/voa_1.py \
--local_raw_data_dir /media/nampv1/hdd/data/Voice-of-Address/raw/generated \
--load_from_disk \
--text_column text
# --output_dir /media/nampv1/hdd/data/Voice-of-Address/eda \
# --log_file /media/nampv1/hdd/data/Voice-of-Address/eda/eda_run.log

