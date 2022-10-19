import os
import zarr
import pickle
import shutil
import hashlib
import logging

import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data.dataset import Dataset as TorchDataset
from tqdm.auto import tqdm


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']
DATA_TEMP_PATH = os.environ['CAI_TEMP_PATH']

logger = logging.getLogger(__name__)


class ContextInjectionDataset(TorchDataset):
    """A wrapper dataset that takes a dataset object and injects context embeddings into it.

    Args:
        base_dataset (Dataset): A PyTorch Dataset object to wrap.
        context_file (str): The npz file with preprocessed contexts, under $CAI_DATA_BASE_PATH.
        context_lookup_key (str): The name of the key in the base dataset to use to look up the context.
        context_name_key (str, optional): The name of the context key in the datums returned by the wrapper object.
            Defaults to 'context_embedding'.
    """

    @staticmethod
    def hash_key(key):
        return hashlib.sha224(key.encode("utf-8")).hexdigest()

    def __init__(self, base_dataset, context_lookup_key, context_name_key="context_embedding", raw_contexts=False):
        super().__init__()
        self.base_dataset = base_dataset
        if len(base_dataset) == 0:
            return

        self.context_lookup_key, self.context_name_key = context_lookup_key, context_name_key
        self.context_store = os.path.join(DATA_TEMP_PATH, "context_encodings.zarr")
        if not os.path.exists(self.context_store):
            raise ValueError("Context encodings Zarr store not found. Call prepare_context_embeddings first.")
        self.raw_contexts = raw_contexts
        logger.info("Loading built contexts")
        with zarr.DirectoryStore(self.context_store) as zarr_store:
            contexts = zarr.group(store=zarr_store)
            self.all_context_keys = set(contexts.array_keys())
            if not self.raw_contexts:
                self.embed_dim = contexts[next(iter(self.all_context_keys))].shape[-1]
            else:
                self.embed_dim = 1
        self.empty_embedding = np.empty((0, self.embed_dim))

        logger.info("Testing alignment with base dataset")
        not_found = []
        for ex in tqdm(self.base_dataset):
            if not self.hash_key(ex[context_lookup_key]) in self.all_context_keys:
                not_found.append(ex[context_lookup_key])
        logger.info(
            f"Not found {len(not_found)} examples, this is {len(not_found) / len(self.base_dataset):.2%} of the "
                "data")
        if len(not_found) > 0:
            logger.info("First 10 bad examples:")
            for key in not_found[:10]:
                logger.info(f"   {key}")

    @staticmethod
    def prepare_context_embeddings(
        context_file,
        context_encoder,
        raw_contexts=False,
        cuda=False,
        batch_size=16,
        overwrite=False
    ):
        context_store = os.path.join(DATA_TEMP_PATH, "context_encodings.zarr")
        if os.path.isdir(context_store):
            if overwrite:
                logger.warning("Resetting temporary Zarr store directory")
                shutil.rmtree(context_store)
            else:
                return

        logger.info("Loading target language contexts")
        if not context_file.endswith(".pkl"):
            raise ValueError("The context filename should end with .pkl, instead got " + context_file)
        with open(os.path.join(DATA_BASE_PATH, context_file), 'rb') as f:
            contexts = pickle.load(f)

        logger.info("Loading context encoding model")
        tokenizer = AutoTokenizer.from_pretrained(context_encoder)
        if not raw_contexts:
            model = AutoModelForMaskedLM.from_pretrained(context_encoder)
            if getattr(model.config, "is_encoder_decoder", False):
                model = model.model.encoder
            model.eval()
            if cuda:
                model.cuda()

        logger.info("Encoding contexts")
        fragments, contexts = zip(*list(contexts.items()))
        fragments, contexts = list(fragments), list(contexts)
        with zarr.DirectoryStore(context_store) as zarr_store:
            seen_hashes = set()
            outputs = zarr.group(store=zarr_store, overwrite=True)
            for batch_idx in tqdm(range(len(contexts) // batch_size), desc="Encoding"):
                batch_fragments = fragments[batch_size * batch_idx : batch_size * (batch_idx + 1)]
                batch = contexts[batch_size * batch_idx : batch_size * (batch_idx + 1)]
                batch = tokenizer(batch, return_tensors="pt", padding=True)
                if raw_contexts:
                    encoded = batch['input_ids'].cpu().detach().numpy()
                else:
                    encoded = model(**batch.to(model.device)).last_hidden_state.cpu().detach().numpy()
                batch = batch.input_ids.cpu().numpy()
                for frag_name, tokens, encoding in zip(batch_fragments, batch, encoded):
                    end_idx = np.where(tokens == tokenizer.eos_token_id)[0][0]
                    frag_name = ContextInjectionDataset.hash_key(frag_name)
                    if frag_name in seen_hashes:
                        raise ValueError("Hash collision!")
                    seen_hashes.add(frag_name)
                    outputs.array(frag_name, encoding[:end_idx + 1])

    def __getitem__(self, index):
        if self.context_store is None and not self.raw_contexts:
            raise ValueError("No prepared contexts found, first call prepare_context_embeddings.")

        base_item = self.base_dataset[index]
        context_key = self.hash_key(base_item[self.context_lookup_key])
        if context_key in self.all_context_keys:
            with zarr.DirectoryStore(self.context_store) as zarr_store:
                contexts = zarr.group(store=zarr_store)
                # The array has to be loaded into memory so that it still exists after the store closes
                z_array = contexts[context_key]
                context_embed = np.array(z_array)
                del z_array
        else:
            context_embed = self.empty_embedding
        base_item[self.context_name_key] = context_embed
        if self.raw_contexts:
            context_mask_shape = context_embed.shape
        else:
            context_mask_shape = context_embed.shape[:-1]
        base_item[self.context_name_key + "_mask"] = np.ones(context_mask_shape, dtype=np.int32)
        if self.raw_contexts:
            for key in [self.context_name_key, self.context_name_key + "_mask"]:
                base_item[key] = base_item[key].tolist()
        return base_item

    def __len__(self):
        return len(self.base_dataset)
