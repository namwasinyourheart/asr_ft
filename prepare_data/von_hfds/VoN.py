from pathlib import Path
import os
import csv
import json
import datasets
from typing import Dict, Any

# ============================================================
# 1️⃣ Custom Config
# ============================================================

class VoNConfig(datasets.BuilderConfig):
    """Custom config for VoN dataset."""
    def __init__(self, subset_info: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.subset_info = subset_info or {}


# ============================================================
# 2️⃣ Dataset Builder
# ============================================================

class VoN(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = []
    DEFAULT_CONFIG_NAME = "default"

    # ------------------------------------------------------------
    # Create dynamic configs
    # ------------------------------------------------------------
    def _create_builder_configs(self, data_dir, **kwargs):
        """Create configs for the given data_dir."""
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

        for category_dir in data_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            for provider_dir in category_dir.iterdir():
                if not provider_dir.is_dir():
                    continue

                for variant_dir in provider_dir.iterdir():
                    if not variant_dir.is_dir():
                        continue

                    for lang_dir in variant_dir.iterdir():
                        if not lang_dir.is_dir():
                            continue

                        wav_dir = lang_dir / "wav"
                        metadata_json = lang_dir / "metadata.json"
                        text_audio_tsv = lang_dir / "text_audio.tsv"

                        if metadata_json.exists() and text_audio_tsv.exists() and wav_dir.exists():
                            subset_name = f"{category_dir.name}__{provider_dir.name}__{variant_dir.name}__{lang_dir.name}"
                            subset_info = {
                                "category": category_dir.name,
                                "provider": provider_dir.name,
                                "variant": variant_dir.name,
                                "lang": lang_dir.name,
                                "metadata_json": str(metadata_json),
                                "text_audio_tsv": str(text_audio_tsv),
                                "wav_dir": str(wav_dir),
                            }
                            
                            configs.append(
                                VoNConfig(
                                    name=subset_name,
                                    description=f"VoN dataset - {category_dir.name}/{provider_dir.name}/{variant_dir.name}/{lang_dir.name}",
                                    subset_info=subset_info,
                                    version=self.VERSION,
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
                            print(f"    ❌ Skipping {lang_dir} - missing: {', '.join(missing_files)}")

        print(f"📊 Total configs found: {found_configs}")

        # Create combined config for all providers
        if found_configs > 0:
            all_subset_names = [cfg.name for cfg in configs]
            configs.append(
                VoNConfig(
                    name="combined",
                    description="Combined VoN dataset across all categories, providers and languages.",
                    subset_info={
                        "is_combined": True,
                        "subsets_to_combine": all_subset_names,
                        "data_dir": str(data_path)
                    },
                    version=self.VERSION,
                )
            )
            # Add a "default" config that is an alias for "combined"
            configs.append(
                VoNConfig(
                    name="default",
                    description="Default config, loads all data.",
                    subset_info={
                        "is_combined": True,
                        "subsets_to_combine": all_subset_names,
                        "data_dir": str(data_path)
                    },
                    version=self.VERSION,
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
            description="VoN (Voice of Names) dataset.",
            features=datasets.Features({
                "utt_id": datasets.Value("string"),
                "text_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=22050),
                "metadata": {
                    "provider": datasets.Value("string"),
                    "variant": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "duration": datasets.Value("float32"),
                    "sample_rate": datasets.Value("int32"),
                }
            }),
            supervised_keys=("audio", "text"),
        )

    # ------------------------------------------------------------
    def _split_generators(self, dl_manager):
        builder_configs_dict = {cfg.name: cfg for cfg in self.BUILDER_CONFIGS}
        
        von_config = builder_configs_dict.get(self.config.name)
        if not von_config:
            available_configs = list(builder_configs_dict.keys())
            print(f"❌ Config '{self.config.name}' not found.")
            print(f"📋 Available configs: {available_configs}")
            print(f"💡 Tip: Use 'combined' to load all available data, or check your data_dir structure.")
            raise ValueError(f"Config '{self.config.name}' not found in builder_configs. Available: {available_configs}")
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"config": von_config}),
        ]

    # ------------------------------------------------------------
    def _generate_examples(self, **kwargs):
        config = kwargs.get("config")
        if not config:
            raise ValueError("VoNConfig not passed to _generate_examples. Check _split_generators.")
            
        subset_info = config.subset_info
        
        # Handle combined config
        if subset_info.get("is_combined", False):
            data_path = Path(subset_info["data_dir"])
            temp_builder = self.__class__()
            all_configs = {cfg.name: cfg for cfg in temp_builder._create_builder_configs(data_dir=data_path)}
            
            idx = 0
            for subset_name in subset_info["subsets_to_combine"]:
                cfg = all_configs.get(subset_name)
                if cfg and not cfg.subset_info.get("is_combined"):
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
            # Read shared info from metadata.json
            with open(metadata_json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Read individual audio info from text_audio.tsv
            with open(text_audio_tsv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None)  # Skip header
                for row in reader:
                    if not row or len(row) < 6:
                        continue
                    
                    utt_id, text_id, text, audio_path, duration, _ = row
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
                            "provider": info.get("provider"),
                            "variant": info.get("variant", ""),
                            "lang": info.get("lang", ""),
                            "category": info.get("category", ""),
                            "duration": duration_float,
                            "sample_rate": 22050,  # Assuming standard sample rate
                        }
                    })

        except Exception as e:
            print(f"❌ Error parsing {metadata_json_file} or {text_audio_tsv_file}: {e}")
        return entries
