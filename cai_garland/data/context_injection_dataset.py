import os
import logging

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']

logger = logging.getLogger(__name__)


class ContextInjectionDataset(TorchDataset):
    """A wrapper dataset that takes a dataset object and injects context embeddings into it.

    Args:
        base_dataset (Dataset): A PyTorch Dataset object to wrap.
        context_file (str): The npz file with preprocessed contexts, under $CAI_DATA_BASE_PATH/processed_datasets.
        context_lookup_key (str): The name of the key in the base dataset to use to look up the context.
        context_name_key (str, optional): The name of the context key in the datums returned by the wrapper object.
            Defaults to 'context'.
    """

    def __init__(self, base_dataset, context_file, context_lookup_key, context_name_key="context"):
        super().__init__()
        if not context_file.endswith(".npz"):
            raise ValueError("The context file should be a .npz file, instead got " + context_file)
        self.context_file = os.path.join(DATA_BASE_PATH, f"processed_datasets/{context_file}")
        self.context_lookup_key, self.context_name_key = context_lookup_key, context_name_key
        self.base_dataset = base_dataset

        if len(base_dataset) == 0:
            return

        logger.info("Loading prebuilt contexts")
        self.contexts = np.load(self.context_file)

        self.embed_dim = self.contexts[self.contexts.files[0]].shape[-1]
        self.empty_embedding = np.empty((0, self.embed_dim))

        logger.info("Testing alignment with base dataset")
        self.all_context_keys = set(self.contexts.files)
        not_found = []
        for ex in base_dataset:
            if not ex['english'] in self.all_context_keys:
                not_found.append(ex['english'])
        logger.info(
            f"Not found {len(not_found)} examples, this is {len(not_found) / len(base_dataset):.2%} of the data")
        if len(not_found) > 0:
            logger.info("First 10 bad examples:")
            for key in not_found[:10]:
                logger.info(f"   {key}")

    def __del__(self):
        self.contexts.close()

    def __getitem__(self, index):
        base_item = self.base_dataset[index]
        context_key = base_item[self.context_lookup_key]
        if context_key in self.all_context_keys:
            context_embed = self.contexts[context_key]
        else:
            context_embed = self.empty_embedding
        base_item[self.context_name_key] = context_embed
        return base_item
