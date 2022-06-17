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
        dataset_location: Location of the processed dataset files within the data registry.
    """

    dataset_location = "processed_datasets/84000-parallel-sentences"

    BUILDER_CONFIGS = [
        ParallelSentences84000Config(
            name="text_pairs",
            version=datasets.Version("0.0.1", ""),
            description="Text pairs",
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
        files_path = os.path.join(os.environ['CAI_DATA_BASE_PATH'], self.dataset_location)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "bo_fn": os.path.join(files_path, "train.bo"),
                    "en_fn": os.path.join(files_path, "train.en")}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "bo_fn": os.path.join(files_path, "test.bo"),
                    "en_fn": os.path.join(files_path, "test.en")}),
        ]

    def _generate_examples(self, bo_fn, en_fn):
        # Read two parallel files for a split
        logger.info(f"Loading 84000-parallel-sentences from bo={bo_fn} and en={en_fn}")
        with open(bo_fn, encoding="utf-8") as bo_f, open(en_fn, encoding="utf-8") as en_f:
            for id_, (bo, en) in enumerate(zip(bo_f, en_f)):
                yield id_, {
                    "tibetan": bo.strip(),
                    "english": en.strip()
                }
