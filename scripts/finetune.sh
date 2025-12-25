#Venterprise
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.prj_name="improving_stt_1.1" \
    exp_manager.exp_name=venterprise_address_hanoi__vnp_stt_1.1 \
    exp_manager.exp_variant="ft_bs48_lr1e-5" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920" \
    data.root_data_dir="/media/nampv1/hdd/data/Venterprise" \
    data.prepared_data_dir="/media/nampv1/hdd/data/Venterprise/exps/venterprise_address_hanoi__openai_whisper_large_v3_turbo__prepared_data_subset0.01" \
    train.train_args.per_device_train_batch_size=8 \
    train.train_args.per_device_eval_batch_size=16 \
    train.train_args.num_train_epochs=3 \
    train.val_n_samples=1024 \
    train.lora.r=32 \
    train.lora.lora_alpha=64
    

    +data.exps_data_dir="/media/nampv1/hdd/data/VoW/exps" \


# local
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name=vow__vnpost_stt_1 \
    exp_manager.exp_variant="prepared_data_gtts_subset0.01" \
    model.pretrained_model_name_or_path="/media/nampv1/hdd/models/asr/models/merged/vnpost_asr_01_20250920" \
    data.root_data_dir="/media/nampv1/hdd/data/VoW" \
    data.prepared_data_dir="/media/nampv1/hdd/data/VoW/exps/vow__vnpost_stt_1.1__prepared_data_gtts_subset0.01" \
    train.train_args.per_device_train_batch_size=48 \
    train.train_args.per_device_eval_batch_size=192 \
    train.train_args.num_train_epochs=3 \
    train.val_n_samples=1024 \
    train.lora.r=32 \
    train.lora.lora_alpha=64
    

    +data.exps_data_dir="/media/nampv1/hdd/data/VoW/exps" \


#server
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --config_path="configs/10112025/vow__openai_whisper-large-v3-turbo.yaml" \
    exp_manager.exp_name=vow__vnpost_stt_1 \
    exp_manager.exp_variant="prepared_data_gtts_subset0.01" \
    model.pretrained_model_name_or_path="/data/nampv1/models/asr/models/merged/vnpost_asr_01_20250920" \
    data.root_data_dir="/data/nampv1/VoW" \
    data.prepared_data_dir="/data/nampv1/VoW/exps/vow__vnpost_stt_1.1__prepared_data_gtts_subset0.01" \
    train.train_args.per_device_train_batch_size=48 \
    train.train_args.per_device_eval_batch_size=192 \
    train.train_args.num_train_epochs=3 \
    train.val_n_samples=1024 \
    train.lora.r=32 \
    train.lora.lora_alpha=64
    