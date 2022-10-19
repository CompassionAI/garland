# Huggingface loader script for the dataset prepared by parallel_dataset_prep
#   See here: https://huggingface.co/docs/datasets/v2.3.2/en/dataset_script
import os
import datasets


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
Dataset of long parallel text made from the 84,000 translations, as well as some augmenting dictionaries. See the
dataset card in the CompassionAI data registry under "processed_datasets/84000-parallel-sentences.card.md".
"""


class ParallelSentences84000Config(datasets.BuilderConfig):
    """BuilderConfig for ParallelSentences84000."""

    def __init__(self, **kwargs):
        """BuilderConfig for ParallelSentences84000.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ParallelSentences84000Config, self).__init__(**kwargs)


class ParallelSentences84000(datasets.GeneratorBasedBuilder):
    """ParallelSentences84000 - a CompassionAI augmented dataset of long Tibetan-English pairs made from the 84,000
        parallel sentences.

    Attributes:
        dataset_locations: Location of the processed dataset files within the data registry.
    """

    dataset_locations = {
        "no_registers": "processed_datasets/84000-parallel-sentences-no-registers",
        "no_registers_no_splits": "processed_datasets/84000-parallel-sentences-no-registers",
        "registers_3_only": "processed_datasets/84000-parallel-sentences-3-registers-only",
        "registers_3_only_no_splits": "processed_datasets/84000-parallel-sentences-3-registers-only",
        "registers_2_only": "processed_datasets/84000-parallel-sentences-2-registers-only",
        "raw_with_context": "processed_datasets/84000-parallel-sentences-raw-with-context",
        "raw_with_context_cased": "processed_datasets/84000-parallel-sentences-raw-with-context-cased",
        "raw_with_bidirectional_context": "processed_datasets/84000-parallel-sentences-raw-with-bidirectional-context",
        "nllb_augmentation": "experiments/nllb-augmentation",
    }

    BUILDER_CONFIGS = [
        ParallelSentences84000Config(
            name="no_registers",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with no registers",
        ),
        ParallelSentences84000Config(
            name="no_registers_no_splits",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with no registers and all splits concatenated",
        ),
        ParallelSentences84000Config(
            name="registers_3_only",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with 3 registers and no augmentation",
        ),
        ParallelSentences84000Config(
            name="registers_3_only_no_splits",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with 3 registers, no augmentation, and all splits concatenated",
        ),
        ParallelSentences84000Config(
            name="registers_2_only",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with 2 registers and no augmentation",
        ),
        ParallelSentences84000Config(
            name="raw_with_context",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with no registers, no augmentation, and context embeddings",
        ),
        ParallelSentences84000Config(
            name="raw_with_context_cased",
            version=datasets.Version("0.2.0", ""),
            description="Cased dataset with no registers, no augmentation, and context embeddings",
        ),
        ParallelSentences84000Config(
            name="raw_with_bidirectional_context",
            version=datasets.Version("0.2.0", ""),
            description="Dataset for a Tibetan encoder with no registers, no augmentation, and context in both the "
                        "source and target languages",
        ),
        ParallelSentences84000Config(
            name="nllb_augmentation",
            version=datasets.Version("0.2.0", ""),
            description="Dataset splits extracted from the NLLB model",
        ),
    ]

    def _info(self):
        # Dataset information
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        # Prepare the files for the available splits for reading
        files_path = os.path.join(os.environ['CAI_DATA_BASE_PATH'], self.dataset_locations[self.config.name])

        if self.config.name.endswith("_no_splits"):
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "source_fn": [
                            os.path.join(files_path, "train.bo"),
                            os.path.join(files_path, "valid.bo"),
                            os.path.join(files_path, "test.bo")
                        ],
                        "target_fn": [
                            os.path.join(files_path, "train.en"),
                            os.path.join(files_path, "valid.en"),
                            os.path.join(files_path, "test.en")
                        ],
                    }
                )
            ]
        elif self.config.name == "nllb_augmentation":
            return [
                datasets.SplitGenerator(
                    name=datasets.NamedSplit("bod_Tibt.ita_Latn"),
                    gen_kwargs={
                        "source_fn": os.path.join(files_path, "bod_Tibt-ita_Latn/train.bo"),
                        "target_fn": os.path.join(files_path, "bod_Tibt-ita_Latn/train.it")
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.NamedSplit("bod_Tibt.eng_Latn"),
                    gen_kwargs={
                        "source_fn": os.path.join(files_path, "bod_Tibt-eng_Latn/train.bo"),
                        "target_fn": os.path.join(files_path, "bod_Tibt-eng_Latn/train.it")
                    }
                )
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "source_fn": os.path.join(files_path, "train.bo"),
                        "target_fn": os.path.join(files_path, "train.en")
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "source_fn": os.path.join(files_path, "valid.bo"),
                        "target_fn": os.path.join(files_path, "valid.en")
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "source_fn": os.path.join(files_path, "test.bo"),
                        "target_fn": os.path.join(files_path, "test.en")
                    }
                ),
            ]

    def _read_from_files(self, source_fn, target_fn):
        if isinstance(source_fn, list):
            if not isinstance(target_fn, list) or not len(source_fn) == len(target_fn):
                raise ValueError("Both input files must be either strings or lists of strings of the same length")
            # Read lists of parallel files for a split
            logger.info(
                f"Loading parallel sentences from source=[{', '.join(source_fn)}] and target=[{', '.join(target_fn)}]")
            for cur_source_fn, cur_target_fn in zip(source_fn, target_fn):
                with open(cur_source_fn, encoding="utf-8") as source_f, \
                     open(cur_target_fn, encoding="utf-8") as target_f \
                :
                    for id_, (source, target) in enumerate(zip(source_f, target_f)):
                        id_ = cur_source_fn + '|' + str(id_)
                        yield id_, source, target
            return

        if isinstance(target_fn, list):
            raise ValueError("Both input files must be either strings or lists of strings of the same length")

        # Read two parallel files for a split
        logger.info(f"Loading parallel sentences from source={source_fn} and target={target_fn}")
        with open(source_fn, encoding="utf-8") as source_f, open(target_fn, encoding="utf-8") as target_f:
            for id_, (source, target) in enumerate(zip(source_f, target_f)):
                id_ = source_fn + '|' + str(id_)
                yield id_, source, target

    def _generate_examples(self, source_fn, target_fn):     # pylint: disable=arguments-differ
        for id_, source, target in self._read_from_files(source_fn, target_fn):
            source, target = source.strip(), target.strip()
            yield id_, {
                "source": source,
                "target": target
            }
