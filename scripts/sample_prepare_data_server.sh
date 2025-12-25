#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python prepare_data.py \
    --config_path="configs/12092025/vivos__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.use_existing_hfds=false \
    data.root_data_dir="/data/nampv1/ASR-VIVOS-HCMUS-15H" \
    data.source_data_dir="/data/nampv1/ASR-VIVOS-HCMUS-15H/raw/vivos" \
    data.hf_raw_data_dir="/data/nampv1/ASR-VIVOS-HCMUS-15H/raw/hf" \
    data.prepared_data_dir="/data/nampv1/ASR-VIVOS-HCMUS-15H/exps/toy" \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=0.01


python prepare_data.py \
    --config_path="configs/20102025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    data.use_existing_hfds=true \
    data.continue_prep=true \
    data.root_data_dir="/data/nampv1/Voice-of-Address/" \
    data.hf_raw_data_dir="/data/nampv1/Voice-of-Address/raw/hf1" \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=False \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    +data.only_load_test_split=true \
    +data.do_split=false




