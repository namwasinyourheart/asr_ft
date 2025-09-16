import os
import csv
import datasets


class CommonVoice22Vi(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Common Voice 22 Vietnamese subset",
            features=datasets.Features({
                "sample_id": datasets.Value("uint32"),
                # "client_id": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=48_000),
                "filename": datasets.Value("string"),
                # "path": datasets.Value("string"),
                # "sentence_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                # "sentence_domain": datasets.Value("string"),
                # "up_votes": datasets.Value("int32"),
                # "down_votes": datasets.Value("int32"),
                # "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accents": datasets.Value("string"),
                # "variant": datasets.Value("string"),
                # "locale": datasets.Value("string"),
                # "segment": datasets.Value("string"),
                # "n_words": datasets.Value("int32"),
                # "n_chars": datasets.Value("int32"),
            }),
            supervised_keys=("audio", "text"),
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"tsv_file": os.path.join(data_dir, "train.tsv"),
                            "clips_dir": os.path.join(data_dir, "clips")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"tsv_file": os.path.join(data_dir, "test.tsv"),
                            "clips_dir": os.path.join(data_dir, "clips")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"tsv_file": os.path.join(data_dir, "dev.tsv"),
                            "clips_dir": os.path.join(data_dir, "clips")},
            ),
        ]

    def _generate_examples(self, tsv_file, clips_dir):
        with open(tsv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                # if idx >= 100:
                #     break
                audio_path = os.path.join(clips_dir, row["path"])
                yield idx, {
                    # "client_id": row.get("client_id", ""),
                    "sample_id": idx,
                    "audio": audio_path,
                    "filename": row.get("path", ""),
                    # "sentence_id": row.get("sentence_id", ""),
                    "text": row.get("sentence", ""),
                    # "sentence_domain": row.get("sentence_domain", ""),
                    # "up_votes": int(row["up_votes"]) if row.get("up_votes") else 0,
                    # "down_votes": int(row["down_votes"]) if row.get("down_votes") else 0,
                    # "age": row.get("age", ""),
                    "gender": row.get("gender", ""),
                    "accents": row.get("accents", ""),
                    # "variant": row.get("variant", ""),
                    # "locale": row.get("locale", ""),
                    # "segment": row.get("segment", ""),
                    # "n_words": int(row["n_words"]) if row.get("n_words") else 0,
                    # "n_chars": int(row["n_chars"]) if row.get("n_chars") else 0,
                }
