import os
import datasets


# _CITATION = """\
# @misc{vivos,
#   title = {VIVOS: Vietnamese Speech Corpus},
#   year = {2016},
#   publisher = {Viettel},
# }
# """

# _DESCRIPTION = """\
# VIVOS là một tập dữ liệu tiếng Việt dành cho Automatic Speech Recognition (ASR).
# Bao gồm train/test split với audio .wav, transcript (prompts.txt) và metadata (genders.txt).
# """

# _HOMEPAGE = "https://ailab.hcmus.edu.vn/vivos/"
# _LICENSE = "CC BY-SA 4.0"


class Vivos(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features({
                "sample_id": datasets.Value("uint32"),
                "audio": datasets.Audio(sampling_rate=16000),
                "filename": datasets.Value("string"),
                "speaker_id": datasets.Value("string"),
                "gender": datasets.ClassLabel(names=["m", "f"]),
                # "sentence_id": datasets.Value("string"),
                "text": datasets.Value("string"),
            }),
            supervised_keys=("audio", "text"),
            # homepage=_HOMEPAGE,
            # license=_LICENSE,
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_name": "train",
                    "data_dir": os.path.join(data_dir, "train"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split_name": "test",
                    "data_dir": os.path.join(data_dir, "test"),
                },
            ),
        ]

    def _generate_examples(self, split_name, data_dir):
        # Đọc genders.txt
        genders_path = os.path.join(data_dir, "genders.txt")
        genders = {}
        with open(genders_path, encoding="utf-8") as f:
            for line in f:
                spk, g = line.strip().split()
                genders[spk] = g

        # Đọc prompts.txt
        prompts_path = os.path.join(data_dir, "prompts.txt")
        prompts = {}
        with open(prompts_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    sent_id, text = parts
                    prompts[sent_id] = text

        # Traverse waves/
        wav_dir = os.path.join(data_dir, "waves")
        idx = 0
        for spk in sorted(os.listdir(wav_dir)):
            spk_dir = os.path.join(wav_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            gender = genders.get(spk, None)
            for wav_file in sorted(os.listdir(spk_dir)):
                if not wav_file.endswith(".wav"):
                    continue
                sentence_id = os.path.splitext(wav_file)[0]
                text = prompts.get(sentence_id, "")
                path = os.path.join(spk_dir, wav_file)
                yield idx, {
                    "sample_id": idx,
                    "audio": path,
                    "filename": wav_file,
                    "speaker_id": spk,
                    "gender": gender,
                    # "sentence_id": sentence_id,
                    "text": text,
                }
                idx += 1
