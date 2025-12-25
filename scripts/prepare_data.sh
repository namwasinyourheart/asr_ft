# Venterprise_Address_Hanoi
python prepare_data_no_metadata.py \
    --config_path="configs/10112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="venterprise_address_hanoi__openai_whisper_large_v3_turbo" \
    exp_manager.exp_variant="prepared_data_subset0.01" \
    model.pretrained_model_name_or_path="openai/whisper-large-v3-turbo" \
    data.continue_prep=true \
    data.use_existing_hfds=true \
    data.root_data_dir=/media/nampv1/hdd/data/Venterprise \
    data.hf_raw_data_dir=/media/nampv1/hdd/data/Venterprise/processed/hfds/hf_Venterprise_Hanoi_Address_processed_splitted_143482_1390_470 \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    data.only_load_test_split=false \
    data.do_split=false \
    data.val_ratio=0.1 \
    data.test_ratio=0.1

# local,using existing hfds
python prepare_data.py \
    --config_path="configs/05112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.use_existing_hfds=true \
    data.continue_prep=true \
    data.root_data_dir="/media/nampv1/hdd/data/Voice-of-Address/" \
    data.hf_raw_data_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/hf1" \
    subset_name=null \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    +data.only_load_test_split=true \
    +data.do_split=false

# VoA
python prepare_data.py \
    --config_path="configs/05112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.continue_prep=true \
    data.dataset_source_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=false \
    data.dataset_script_path="prepare_data/voa_hfds/voa.py" \
    data.subset_name="minimax_selenium__default__bdtl_spk2" \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    +data.only_load_test_split=false \
    +data.do_split=false

# VoA, Chua co hfds
python prepare_data_no_metadata.py \
    --config_path="configs/10112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="voa__vnpost_stt_1.1" \
    exp_manager.exp_variant="prepared_data__cdp_commune_district_province_no_prefix__gtts__default__vi" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    data.root_data_dir=/media/nampv1/hdd/data/VoA \
    data.continue_prep=true \
    data.dataset_script_path="prepare_data/voa_hfds/VoA.py" \
    data.dataset_source_dir="/media/nampv1/hdd/data/VoA/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=true \
    data.hf_raw_data_dir=/media/nampv1/hdd/data/VoA/raw/hf__cdp_commune_district_province_no_prefix__gtts__default__vi \
    data.subset_names=[cdp__no_prefix__gtts__default__vi,commune__no_prefix__gtts__default__vi,district__no_prefix__gtts__default__vi,province__no_prefix__gtts__default__vi] \
    data.num_proc=1 \
    data.subset_ratio=1 \
    data.do_split=false \
    data.val_ratio=0.1 \
    data.test_ratio=0.1

# VoA, modal
python prepare_data_no_metadata.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="voa__openai_whisper_small" \
    exp_manager.exp_variant="prepared_data_gtts" \
    model.pretrained_model_name_or_path="openai/whisper-small" \
    data.continue_prep=true \
    data.dataset_source_dir="/mnt/data-vol/VoA/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=true \
    data.hf_raw_data_dir="/mnt/data-vol/VoA/raw/hfds/hf_gtts" \
    data.dataset_script_path="prepare_data/voa_hfds/VoA.py" \
    data.subset_names="gtts__default__vi" \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=1 \
    data.do_save_prepared_data=true \
    data.recalculate_metadata=false

# VoA, Đã có hfds
python prepare_data_no_metadata.py \
    --config_path="configs/10112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="voa__openai_whisper_small" \
    exp_manager.exp_variant="prepared_data_gtts" \
    model.pretrained_model_name_or_path="openai/whisper-small" \
    data.root_data_dir=/media/nampv1/hdd/data/VoA \
    data.continue_prep=true \
    data.use_existing_hfds=true \
    data.hf_raw_data_dir=/media/nampv1/hdd/data/VoA/raw/hfds/hf_gtts \
    data.num_proc=1 \
    data.subset_ratio=1 \
    data.do_split=false \
    data.val_ratio=0.1 \
    data.test_ratio=0.1

# ---------

# VoW
python prepare_data_no_metadata.py \
    --config_path=configs/10112025/vow__openai_whisper-large-v3-turbo.yaml \
    exp_manager.exp_name="vow__vnpost_stt_1.1" \
    exp_manager.exp_variant="prepared_data_gtts_speedup_x2" \
    model.pretrained_model_name_or_path="openai/whisper-large-v3-turbo" \
    data.continue_prep=true \
    data.root_data_dir=/media/nampv1/hdd/data/VoW \
    data.dataset_source_dir=/media/nampv1/hdd/data/VoW/raw/generated \
    data.use_existing_hfds=true \
    data.save_hfds=true \
    data.hf_raw_data_dir=/media/nampv1/hdd/data/VoW/raw/hfds/hongocduc_Viet74K/hf_gtts_speedup_x2 \
    data.dataset_script_path=prepare_data/vow_hfds/VoW.py \
    data.subset_names=gtts__default__vi \
    data.num_proc=1 \
    data.subset_ratio=1 \
    data.do_split=false \
    data.val_ratio=0.1 \
    data.test_ratio=0.1

# VoW, is using
python prepare_data_no_metadata.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="vow__vnpost_stt_1.1" \
    exp_manager.exp_variant="prepared_data_gtts_subset0.01" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    data.root_data_dir=/media/nampv1/hdd/data/VoW \
    data.continue_prep=true \
    data.dataset_source_dir="/media/nampv1/hdd/data/VoW/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=true \
    data.hf_raw_data_dir=/media/nampv1/hdd/data/VoW/raw/hf1_gtts \
    data.dataset_script_path="prepare_data/vow_hfds/VoW.py" \
    data.subset_names="gtts__default__vi" \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    data.do_split=true \
    data.val_ratio=0.1 \
    data.test_ratio=0.1



#server
python prepare_data.py \
    --config_path="configs/20102025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="prepared_data" \
    data.use_existing_hfds=true \
    data.continue_prep=true \
    data.root_data_dir="/data/nampv1/Voice-of-Address/" \
    data.hf_raw_data_dir="/data/nampv1/Voice-of-Address/raw/hf2" \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=1 \
    +data.only_load_test_split=false \
    +data.do_split=false