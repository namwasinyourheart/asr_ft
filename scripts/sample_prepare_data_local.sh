#!/bin/bash

# VIVOS
CUDA_VISIBLE_DEVICES=0 python prepare_data.py \
    --config_path="configs/12092025/vivos__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.use_existing_hfds=false \
    data.root_data_dir="/media/nampv1/hdd/data/VIVOS/" \
    data.source_data_dir="/media/nampv1/hdd/data/VIVOS/raw/vivos" \
    data.hf_raw_data_dir="/media/nampv1/hdd/data/VIVOS/raw/hf" \
    data.prepared_data_dir="/media/nampv1/hdd/data/VIVOS/exps/toy" \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=0.01

# Voice-of-Address
CUDA_VISIBLE_DEVICES=0 python prepare_data.py \
    --config_path="configs/12092025/voice_of_address__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.use_existing_hfds=false \
    data.root_data_dir="/media/nampv1/hdd/data/Voice-of-Address/" \
    data.source_data_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/voice_of_address" \
    data.hf_raw_data_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/hf1" \
    data.prepared_data_dir="/media/nampv1/hdd/data/Voice-of-Address/exps/toy" \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=0.01

# using existing hfds=true
python prepare_data.py \
    --config_path="configs/20102025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.use_existing_hfds=true \
    data.continue_prep=true \
    data.root_data_dir="/media/nampv1/hdd/data/Voice-of-Address/" \
    data.hf_raw_data_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/hf1" \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    +data.only_load_test_split=true \
    +data.do_split=false


if not os.path.exists(prepared_data_dir) or data_args.continue_prep:
        if os.path.exists(hf_raw_data_dir) and data_args.use_existing_hfds:
            dataset = load_from_disk(hf_raw_data_dir)

            if data_args.only_load_test_split:
                dataset = DatasetDict({"test": dataset["test"]})
        else:
            dataset = create_hf_ds(


