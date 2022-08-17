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
        "raw": "processed_datasets/84000-parallel-sentences-raw",
        "raw_no_splits": "processed_datasets/84000-parallel-sentences-raw",
        "raw_no_dict": "processed_datasets/84000-parallel-sentences-raw-no-dict",
        "raw_no_dict_no_splits": "processed_datasets/84000-parallel-sentences-raw-no-dict",
        "registers_3": "processed_datasets/84000-parallel-sentences-3-registers",
        "registers_3_only": "processed_datasets/84000-parallel-sentences-3-registers-only",
        "registers_3_only_no_splits": "processed_datasets/84000-parallel-sentences-3-registers-only",
        "registers_2_only": "processed_datasets/84000-parallel-sentences-2-registers-only",
    }

    BUILDER_CONFIGS = [
        ParallelSentences84000Config(
            name="no_registers",
            version=datasets.Version("0.1.1", ""),
            description="Dataset for a Tibetan encoder with no registers",
        ),
        ParallelSentences84000Config(
            name="no_registers_no_splits",
            version=datasets.Version("0.1.1", ""),
            description="Dataset for a Tibetan encoder with no registers and all splits concatenated",
        ),
        ParallelSentences84000Config(
            name="raw",
            version=datasets.Version("0.1.1", ""),
            description="Dataset for a Tibetan encoder from the raw parallel sentences",
        ),
        ParallelSentences84000Config(
            name="raw_no_splits",
            version=datasets.Version("0.1.1", ""),
            description="Dataset for a Tibetan encoder from the raw parallel sentences and all splits concatenated",
        ),
        ParallelSentences84000Config(
            name="raw_no_dict",
            version=datasets.Version("0.1.1", ""),
            description="Dataset for a Tibetan encoder from the raw parallel sentences and no dictionary",
        ),
        ParallelSentences84000Config(
            name="raw_no_dict_no_splits",
            version=datasets.Version("0.1.2", ""),
            description="Dataset for a Tibetan encoder from the raw parallel sentences, no dictionary, and all splits "
                        "concatenated",
        ),
        ParallelSentences84000Config(
            name="registers_3",
            version=datasets.Version("0.1.2", ""),
            description="Dataset for a Tibetan encoder with 3 registers",
        ),
        ParallelSentences84000Config(
            name="registers_3_only",
            version=datasets.Version("0.1.2", ""),
            description="Dataset for a Tibetan encoder with 3 registers and no augmentation",
        ),
        ParallelSentences84000Config(
            name="registers_3_only_no_splits",
            version=datasets.Version("0.1.2", ""),
            description="Dataset for a Tibetan encoder with 3 registers, no augmentation, and all splits concatenated",
        ),
        ParallelSentences84000Config(
            name="registers_2_only",
            version=datasets.Version("0.1.2", ""),
            description="Dataset for a Tibetan encoder with 2 registers and no augmentation",
        ),
    ]

    def _info(self):
        # Dataset information
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tibetan": datasets.Value("string"),
                    "english": datasets.Value("string")
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
                        "bo_fn": [
                            os.path.join(files_path, "train.bo"),
                            os.path.join(files_path, "valid.bo"),
                            os.path.join(files_path, "test.bo")
                        ],
                        "en_fn": [
                            os.path.join(files_path, "train.en"),
                            os.path.join(files_path, "valid.en"),
                            os.path.join(files_path, "test.en")
                        ]
                    }
                )
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "bo_fn": os.path.join(files_path, "train.bo"),
                        "en_fn": os.path.join(files_path, "train.en")}),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "bo_fn": os.path.join(files_path, "valid.bo"),
                        "en_fn": os.path.join(files_path, "valid.en")}),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "bo_fn": os.path.join(files_path, "test.bo"),
                        "en_fn": os.path.join(files_path, "test.en")}),
            ]

    def _generate_examples(self, bo_fn, en_fn):     # pylint: disable=arguments-differ
        if isinstance(bo_fn, list):
            if not isinstance(en_fn, list) or not len(bo_fn) == len(en_fn):
                raise ValueError("Both input files must be either strings or lists of strings of the same length")
            # Read lists of parallel files for a split
            logger.info(f"Loading parallel sentences from bo=[{', '.join(bo_fn)}] and en=[{', '.join(en_fn)}]")
            for cur_bo_fn, cur_en_fn in zip(bo_fn, en_fn):
                with open(cur_bo_fn, encoding="utf-8") as bo_f, open(cur_en_fn, encoding="utf-8") as en_f:
                    for id_, (bo, en) in enumerate(zip(bo_f, en_f)):
                        id_ = cur_bo_fn + '|' + str(id_)
                        yield id_, {
                            "tibetan": bo.strip(),
                            "english": en.strip()
                        }
            return

        if isinstance(en_fn, list):
            raise ValueError("Both input files must be either strings or lists of strings of the same length")

        # Read two parallel files for a split
        logger.info(f"Loading parallel sentences from bo={bo_fn} and en={en_fn}")
        with open(bo_fn, encoding="utf-8") as bo_f, open(en_fn, encoding="utf-8") as en_f:
            for id_, (bo, en) in enumerate(zip(bo_f, en_f)):
                id_ = bo_fn + '|' + str(id_)
                yield id_, {
                    "tibetan": bo.strip(),
                    "english": en.strip()
                }
