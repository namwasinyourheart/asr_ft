from pathlib import Path
import hashlib
import os
import csv
import json
import datasets
from typing import Dict, Any


# ============================================================
# 1️⃣ Custom Config
# ============================================================

class VoAConfig(datasets.BuilderConfig):
    """Custom config cho từng subset hoặc combined."""
    def __init__(self, subset_info: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.subset_info = subset_info or {}


# ============================================================
# 2️⃣ Dataset Builder
# ============================================================

class VoA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.2.0")
    BUILDER_CONFIGS = []
    DEFAULT_CONFIG_NAME = "combined"

    # ------------------------------------------------------------
    # Tạo danh sách config động
    # ------------------------------------------------------------
    def _create_builder_configs(self, data_dir, **kwargs):
        """Hook to create the configs for the given data_dir.

        This method is called before the created configs are used.

        Args:
            data_dir: path to the data directory.
            **kwargs: keyword arguments that are passed to `datasets.load_dataset_builder`.

        Returns:
            List[datasets.BuilderConfig]: list of builder configs.
        """
        configs = []
        if data_dir is None:
            print("⚠️ Warning: data_dir is None, only 'combined' config will be available")
            return configs
            
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️ Warning: data_dir '{data_dir}' does not exist, only 'combined' config will be available")
            return configs

        print(f"🔍 Scanning data directory: {data_path}")
        found_configs = 0

        for level in ["commune", "district", "province", "cdp"]:
            for variant in ["with_prefix", "no_prefix"]:
                base_dir = data_path / f"{level}_{variant}"
                if not base_dir.exists():
                    print(f"  ❌ Directory not found: {base_dir}")
                    continue
                
                print(f"  ✅ Found level directory: {base_dir}")
                for provider in base_dir.iterdir():
                    if not provider.is_dir():
                        continue
                    for model in provider.iterdir():
                        if not model.is_dir():
                            continue
                        for voice in model.iterdir():
                            wav_dir = voice / "wav"
                            metadata_json = voice / "metadata.json"
                            text_audio_tsv = voice / "text_audio.tsv"

                            if metadata_json.exists() and text_audio_tsv.exists() and wav_dir.exists():
                                subset_name = f"{level}__{variant}__{provider.name}__{model.name}__{voice.name}"
                                subset_info = {
                                    "level": level,
                                    "variant": variant,
                                    "provider": provider.name,
                                    "model": model.name,
                                    "voice": voice.name,
                                    "metadata_json": str(metadata_json),
                                    "text_audio_tsv": str(text_audio_tsv),
                                    "wav_dir": str(wav_dir),
                                }
                                configs.append(
                                    VoAConfig(
                                        name=subset_name,
                                        description=f"{level} ({subset_info['variant']}) - {provider.name}/{model.name}/{voice.name}",
                                        subset_info=subset_info,
                                        version=datasets.Version("2.2.0"),
                                    )
                                )
                                found_configs += 1
                                print(f"    ✅ Added config: {subset_name}")
                            else:
                                missing_files = []
                                if not metadata_json.exists():
                                    missing_files.append("metadata.json")
                                if not text_audio_tsv.exists():
                                    missing_files.append("text_audio.tsv")
                                if not wav_dir.exists():
                                    missing_files.append("wav/")
                                print(f"    ❌ Skipping {voice} - missing: {', '.join(missing_files)}")

        print(f"📊 Total configs found: {found_configs}")

        # Group configs by (provider, model, voice) to create combined subsets
        grouped_configs = {}
        for cfg in configs:
            info = cfg.subset_info
            key = (info["provider"], info["model"], info["voice"])
            if key not in grouped_configs:
                grouped_configs[key] = []
            grouped_configs[key].append(cfg.name)

        # Create combined configs for each group
        for (provider, model, voice), subset_names in grouped_configs.items():
            if len(subset_names) > 1:  # Only create a group if there's more than one subset
                combined_name = f"{provider}__{model}__{voice}"
                configs.append(
                    VoAConfig(
                        name=combined_name,
                        description=f"Combined dataset for {provider}/{model}/{voice} across all levels and variants.",
                        subset_info={
                            "is_group": True,
                            "subsets_to_combine": subset_names,
                            "data_dir": str(data_path)
                        },
                        version=datasets.Version("2.2.0"),
                    )
                )

        # ✅ Grand combined config (all subsets)
        configs.append(
            VoAConfig(
                name="combined",
                description="Combined dataset across all subsets (single split).",
                subset_info={"combined": True, "data_dir": str(data_path)},
                version=datasets.Version("2.2.0"),
            )
        )
        return configs

    def __init__(self, *args, **kwargs):
        data_dir = kwargs.get("data_dir")
        if data_dir and not self.BUILDER_CONFIGS:
            self.__class__.BUILDER_CONFIGS = self._create_builder_configs(data_dir=data_dir)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------
    def _info(self):
        return datasets.DatasetInfo(
            description="Voice of Address v2 dataset (single train split).",
            features=datasets.Features({
                "utt_id": datasets.Value("string"),
                "text_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=22050),
                "metadata": {
                    "provider": datasets.Value("string"),
                    "model": datasets.Value("string"),
                    "voice": datasets.Value("string"),
                    "tts_type": datasets.Value("string"),
                    "sample_rate": datasets.Value("int32"),
                    "lang": datasets.Value("string"),
                    "duration": datasets.Value("float32"),
                    "gen_date": datasets.Value("string"),
                    "level": datasets.Value("string"),
                    "variant": datasets.Value("string"),
                    "subset_name": datasets.Value("string"),
                }
            }),
            supervised_keys=("audio", "text"),
        )

    # ------------------------------------------------------------
    def _split_generators(self, dl_manager):
        # Convert list to dict for lookup
        builder_configs_dict = {cfg.name: cfg for cfg in self.BUILDER_CONFIGS}
        
        voa_config = builder_configs_dict.get(self.config.name)
        if not voa_config:
            available_configs = list(builder_configs_dict.keys())
            print(f"❌ Config '{self.config.name}' not found.")
            print(f"📋 Available configs: {available_configs}")
            print(f"💡 Tip: Use 'combined' to load all available data, or check your data_dir structure.")
            raise ValueError(f"Config '{self.config.name}' not found in builder_configs. Available: {available_configs}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"config": voa_config}),
        ]

    # ------------------------------------------------------------
    def _generate_examples(self, **kwargs):
        config = kwargs.get("config")
        if not config:
            raise ValueError("VoAConfig not passed to _generate_examples. Check _split_generators.")
        subset_info = config.subset_info
        # Handle grand-combined config
        if subset_info.get("combined", False):
            data_path = Path(subset_info["data_dir"])
            temp_builder = self.__class__()
            all_configs = {cfg.name: cfg for cfg in temp_builder._create_builder_configs(data_dir=data_path)}
            idx = 0
            for cfg_name, cfg in all_configs.items():
                if cfg.subset_info.get("combined") or cfg.subset_info.get("is_group"):
                    continue # Skip combined/group configs to avoid recursion
                for _, example in self._generate_subset_examples(cfg.subset_info, cfg.name):
                    yield idx, example
                    idx += 1

        # Handle grouped-subset config
        elif subset_info.get("is_group", False):
            data_path = Path(subset_info["data_dir"])
            temp_builder = self.__class__()
            all_configs = {cfg.name: cfg for cfg in temp_builder._create_builder_configs(data_dir=data_path)}
            idx = 0
            for subset_name_to_combine in subset_info["subsets_to_combine"]:
                cfg = all_configs.get(subset_name_to_combine)
                if cfg:
                    for _, example in self._generate_subset_examples(cfg.subset_info, cfg.name):
                        yield idx, example
                        idx += 1
        else:
            for idx, example in self._generate_subset_examples(subset_info, config.name):
                yield idx, example

    def _generate_subset_examples(self, subset_info, config_name):
        entries = self._parse_metadata(
            Path(subset_info["metadata_json"]),
            Path(subset_info["text_audio_tsv"]),
            Path(subset_info["wav_dir"]),
            subset_info,
            config_name
        )
        for idx, e in enumerate(entries):
            e["split"] = "train"
            yield idx, e

    # ------------------------------------------------------------
    def _parse_metadata(self, metadata_json_file: Path, text_audio_tsv_file: Path, wav_dir: Path, info: Dict[str, Any], subset_name: str):
        entries = []
        try:
            # 1. Read shared info from metadata.json
            with open(metadata_json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 2. Read individual audio info from text_audio.tsv
            with open(text_audio_tsv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None) # Skip header
                for row in reader:
                    if not row or len(row) < 6:
                        continue
                    
                    utt_id, text_id, text, audio_path, duration, gen_date = row
                    audio_filename = os.path.basename(audio_path)
                    full_audio_path = wav_dir / audio_filename

                    # Handle empty or invalid duration values
                    try:
                        duration_float = float(duration) if duration and duration.strip() else 0.0
                    except (ValueError, TypeError):
                        duration_float = 0.0

                    entries.append({
                        "utt_id": utt_id,
                        "text_id": text_id,
                        "text": text,
                        "audio": str(full_audio_path),
                        "metadata": {
                            "provider": metadata.get("provider"),
                            "model": metadata.get("model"),
                            "voice": metadata.get("voice"),
                            "tts_type": metadata.get("tts_type"),
                            "sample_rate": metadata.get("sampling_rate"),
                            "lang": metadata.get("lang"),
                            "duration": duration_float,
                            "gen_date": gen_date,
                            "level": info.get("level", ""),
                            "variant": info.get("variant", ""),
                            "subset_name": subset_name,
                        }
                    })

        except Exception as e:
            print(f"❌ Error parsing {metadata_json_file} or {text_audio_tsv_file}: {e}")
        return entries
