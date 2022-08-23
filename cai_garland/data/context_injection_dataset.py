import os
import zarr
import hashlib
import logging

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from tqdm.auto import tqdm


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']

logger = logging.getLogger(__name__)


class ContextInjectionDataset(TorchDataset):
    """A wrapper dataset that takes a dataset object and injects context embeddings into it.

    Args:
        base_dataset (Dataset): A PyTorch Dataset object to wrap.
        context_file (str): The npz file with preprocessed contexts, under $CAI_DATA_BASE_PATH/processed_datasets.
        context_lookup_key (str): The name of the key in the base dataset to use to look up the context.
        context_name_key (str, optional): The name of the context key in the datums returned by the wrapper object.
            Defaults to 'context_embedding'.
    """

    @staticmethod
    def hash_key(key):
        return hashlib.sha224(key.encode("utf-8")).hexdigest()

    def __init__(self, base_dataset, context_file, context_lookup_key, context_name_key="context_embedding"):
        super().__init__()
        if not context_file.endswith(".mdb"):
            raise ValueError("The context filename should end with .mdb, instead got " + context_file)
        self.context_store = os.path.join(DATA_BASE_PATH, f"processed_datasets/{context_file}")
        self.context_lookup_key, self.context_name_key = context_lookup_key, context_name_key
        self.base_dataset = base_dataset

        if len(base_dataset) == 0:
            return

        logger.info("Loading prebuilt contexts")
        self.zarr_store, self.contexts = None, None
        with zarr.LMDBStore(self.context_store) as zarr_store:
            contexts = zarr.group(store=zarr_store)
            self.all_context_keys = set(contexts.array_keys())
            self.embed_dim = contexts[next(iter(self.all_context_keys))].shape[-1]
        self.empty_embedding = np.empty((0, self.embed_dim))

        logger.info("Testing alignment with base dataset")
        not_found = []
        for ex in tqdm(base_dataset):
            if not self.hash_key(ex['english']) in self.all_context_keys:
                not_found.append(ex['english'])
        logger.info(
            f"Not found {len(not_found)} examples, this is {len(not_found) / len(base_dataset):.2%} of the data")
        if len(not_found) > 0:
            logger.info("First 10 bad examples:")
            for key in not_found[:10]:
                logger.info(f"   {key}")

    def __del__(self):
        if self.zarr_store is not None:
            self.zarr_store.close()

    def __getitem__(self, index):
        if self.zarr_store is None:
            self.zarr_store = zarr.LMDBStore(self.context_store)
            self.contexts = zarr.group(store=self.zarr_store)

        base_item = self.base_dataset[index]
        context_key = self.hash_key(base_item[self.context_lookup_key])
        if context_key in self.all_context_keys:
            context_embed = self.contexts[context_key]
        else:
            context_embed = self.empty_embedding
        base_item[self.context_name_key] = context_embed
        base_item[self.context_name_key + "_attention_mask"] = np.ones((context_embed.shape[:-1]))
        return base_item

    def __len__(self):
        return len(self.base_dataset)
