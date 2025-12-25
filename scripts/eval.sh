# local

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config_path="configs/20102025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    +model.adapter_paths=null \
    data.root_data_dir="/media/nampv1/hdd/data/Voice-of-Address/" \
    data.do_shard_for_feature_computation=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=true

# VoA
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config_path="configs/05112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    +model.adapter_paths=null \
    data.continue_prep=true \
    data.dataset_source_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=false \
    data.dataset_script_path="prepare_data/voa_hfds/voa.py" \
    data.subset_name="minimax_selenium__default__bdtl_spk3" \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    +data.only_load_test_split=false \
    +data.do_split=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=true

# Venterprise
#local
python eval_no_metadata.py \
    --config_path="configs/10112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="venterprise_address_hanoi__openai_whisper_large_v3_turbo" \
    exp_manager.exp_variant="eval_prepared_data" \
    model.pretrained_model_name_or_path="openai/whisper-large-v3-turbo" \
    model.adapter_paths=/home/nampv1/projects/asr/asr_ft/exps/venterprise_address_hanoi__vnp_stt_1.1/ft_bs48_lr1e-5/results/adapter \
    data.continue_prep=false \
    data.root_data_dir=/media/nampv1/hdd/data/Venterprise \
    data.prepared_data_dir=/media/nampv1/hdd/data/Venterprise/exps/venterprise_address_hanoi__openai_whisper_large_v3_turbo__prepared_data \
    evaluate.metric_filename="test_metrics.txt" \
    evaluate.prediction_filename="test_predictions.txt" \
    evaluate.result_filename="test_result.txt" \
    evaluate.break_step=-1 \
    evaluate.batch_size=16 \
    evaluate.include_train_split=false \
    evaluate.do_postprocess_text=true

    model.adapter_paths=/media/nampv1/hdd/models/asr/toy_checkpoints/adapter_vow_stt_1.1_ft_147802_8000/adapter \
python eval_no_metadata.py \
    --config_path="configs/10112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="venterprise_address_hanoi__vnp_stt_1.1" \
    exp_manager.exp_variant="eval_prepared_data" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    model.adapter_paths=/home/nampv1/projects/asr/asr_ft/exps/venterprise_address_hanoi__vnp_stt_1.1/ft_bs48_lr1e-5/results/adapter \
    data.continue_prep=false \
    data.root_data_dir=/media/nampv1/hdd/data/Venterprise \
    data.prepared_data_dir=/media/nampv1/hdd/data/Venterprise/exps/venterprise_address_hanoi__openai_whisper_large_v3_turbo__prepared_data \
    evaluate.metric_filename="test_metrics_adapter_ft_bs48_lr1e-5.txt" \
    evaluate.prediction_filename="test_predictions_adapter_ft_bs48_lr1e-5.txt" \
    evaluate.result_filename="test_result_adapter_ft_bs48_lr1e-5.txt" \
    evaluate.break_step=-1 \
    evaluate.batch_size=16 \
    evaluate.include_train_split=false \
    evaluate.do_postprocess_text=true

#server

# VoA
python eval_no_metadata.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="voa__openai_whisper-small" \
    exp_manager.exp_variant="prepared_data_gtts_toy" \
    model.pretrained_model_name_or_path="openai/whisper-small" \
    data.continue_prep=true \
    data.dataset_source_dir="/media/nampv1/hdd/data/VoA/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=false \
    data.dataset_script_path="/home/nampv1/projects/asr/asr_ft/prepare_data/voa_hfds/VoA.py" \
    data.subset_names="gtts__default__vi" \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    data.recalculate_metadata=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=true \
    evaluate.do_postprocess_text=false

# VoW \
#local
python eval_no_metadata.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="vow__vnpost_stt_1.1" \
    exp_manager.exp_variant="prepared_data_gtts_toy" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    data.continue_prep=true \
    data.dataset_source_dir="/media/nampv1/hdd/data/VoW/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=false \
    data.dataset_script_path="prepare_data/vow_hfds/VoW.py" \
    data.subset_names="gtts__default__vi" \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=1 \
    data.recalculate_metadata=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=true

#local, from existiing prepared_data
python eval_no_metadata.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="vow__vnpost_stt_1.1" \
    exp_manager.exp_variant="prepared_data_gtts_toy" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920/" \
    data.continue_prep=true \
    data.prepared_data_dir="/media/nampv1/hdd/data/VoW/exps/vow__vnpost_stt_1.1__prepared_data_gtts_toy" \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    data.recalculate_metadata=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.prediction_filename="toy_predictions.txt" \
    evaluate.result_filename="toy_results.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=false

# server, from existiing prepared_data
python eval_no_metadata.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="vow__vnpost_stt_1.1" \
    exp_manager.exp_variant="prepared_data_gtts_toy" \
    model.pretrained_model_name_or_path="/data/nampv1/models/asr/models/merged/vnpost_asr_01_20250920/" \
    data.continue_prep=true \
    data.prepared_data_dir="/data/nampv1/VoW/exps/vow__vnpost_stt_1.1__prepared_data_gtts_toy" \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    data.recalculate_metadata=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.prediction_filename="toy_predictions.txt" \
    evaluate.result_filename="toy_results.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=false


#server
python eval.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name="vow__openai_whisper-large-v3-turbo" \
    exp_manager.exp_variant="prepared_data_gtts" \
    model.adapter_paths=["exps/openai_whisper-large-v3-turbo__vietbud500/ct_from_ckpt2100/checkpoints/checkpoint-7434","exps/merge_ds_openai_whisper-large-v3-turbo/lora_r32_a64_bs48_lr1e5/checkpoints/checkpoint-6500"] \
    data.continue_prep=true \
    data.dataset_source_dir="/data/nampv1/VoW/raw/" \
    data.use_existing_hfds=false \
    data.save_hfds=false \
    data.dataset_script_path="prepare_data/vow_hfds/VoW.py" \
    data.subset_names="gtts__default__vi" \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=1 \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=true



# data.prepared_data_dir="/media/nampv1/hdd/data/Voice-of-Address/exps/voa__openai_whisper-large-v3-turbo__toy" \

python prepare_data.py \
    --config_path="configs/05112025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \

    data.continue_prep=true \
    data.dataset_source_dir="/media/nampv1/hdd/data/Voice-of-Address/raw/generated" \
    data.use_existing_hfds=false \
    data.save_hfds=false \
    data.dataset_script_path="prepare_data/voa_hfds/voa.py" \
    data.subset_name="gtts__default__vi" \
    +data.do_merge=false \
    data.do_shard_for_feature_computation=false \
    data.num_proc=1 \
    data.subset_ratio=0.01 \
    +data.only_load_test_split=false \
    +data.do_split=false


# server
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config_path="configs/20102025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="eval_include_train" \
    +model.adapter_paths=["exps/openai_whisper-large-v3-turbo__vietbud500/ct_from_ckpt2100/checkpoints/checkpoint-7434","exps/merge_ds_openai_whisper-large-v3-turbo/lora_r32_a64_bs48_lr1e5/checkpoints/checkpoint-6500"] \
    data.root_data_dir="/data/nampv1/Voice-of-Address/" \
    data.prepared_data_dir="/data/nampv1/Voice-of-Address/exps/voa__openai_whisper-large-v3-turbo__prepared_data" \
    data.do_shard_for_feature_computation=false \
    evaluate.metric_filename="test_metrics_include_train.txt" \
    evaluate.prediction_filename="test_predictions_include_train.txt" \
    evaluate.prediction_filename="test_results_include_train.txt" \
    evaluate.break_step=-1 \
    evaluate.batch_size=48 \
    +evaluate.include_train_split=true

    
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config_path="configs/20102025/voa__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_variant="toy" \
    +model.adapter_paths=["exps/openai_whisper-large-v3-turbo__vietbud5000/ct_from_ckpt2100/checkpoints/checkpoint-7434","exps/merge_ds_openai_whisper-large-v3-turbo/lora_r32_a64_bs48_lr1e5/checkpoints/checkpoint-6500"] \
    data.root_data_dir="/data/nampv1/Voice-of-Address/" \
    data.prepared_data_dir="/data/nampv1/Voice-of-Address/exps/voa__openai_whisper-large-v3-turbo__toy" \
    data.do_shard_for_feature_computation=false \
    evaluate.metric_filename="toy_metrics.txt" \
    evaluate.break_step=1 \
    evaluate.batch_size=1 \
    evaluate.include_train_split=true



    