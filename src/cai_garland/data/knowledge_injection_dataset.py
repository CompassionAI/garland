import os
import zarr
import pickle
import shutil
import hashlib
import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data.dataset import Dataset as TorchDataset
from cai_garland.models.factory import make_monolingual_tokenizer
from math import ceil
from tqdm.auto import tqdm


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']
DATA_TEMP_PATH = os.environ['CAI_TEMP_PATH']

logger = logging.getLogger(__name__)


class KnowledgeInjectionDataset(TorchDataset):
    """A wrapper dataset that takes a dataset object and injects exogenous knowledge into it. The exogenous knowledge
        may be context embeddings of preceding target text or a glossary.
    """

    @staticmethod
    def hash_key(key):
        return hashlib.sha224(key.encode("utf-8")).hexdigest()

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset
        self.has_context = False
        self.has_glossary = False
        if len(base_dataset) == 0:
            return

    def inject_context(self, context_lookup_key, context_name_key="context_embedding", raw_contexts=False):
        """Set up injection of context embeddings of preceding target text.

        Args:
            base_dataset (Dataset): A PyTorch Dataset object to wrap.
            context_file (str): The npz file with preprocessed contexts, under $CAI_DATA_BASE_PATH.
            context_lookup_key (str): The name of the key in the base dataset to use to look up the context.
            context_name_key (str, optional): The name of the context key in the datums returned by the wrapper object.
                Defaults to 'context_embedding'.
        """
        self.has_context = True

        if len(self.base_dataset) == 0:
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
            if ex.get('inject_context', True) and not self.hash_key(ex[context_lookup_key]) in self.all_context_keys:
                not_found.append(ex[context_lookup_key])
        logger.info(
            f"Not found {len(not_found)} examples, this is {len(not_found) / len(self.base_dataset):.2%} of the "
                "data")
        if len(not_found) > 0:
            logger.info("First 10 bad examples:")
            for key in not_found[:10]:
                logger.info(f"   {key}")

    def inject_glossary(
        self,
        glossary_dataset,
        source_encoder_name,
        target_decoder_name,
        glossary_name_key="glossary",
        is_deepspeed=False
    ):
        """Set up injection of glossary entries.

        Args:
            base_dataset (Dataset): A PyTorch Dataset object to wrap.
            glossary_dataset (str): The path, under $CAI_DATA_BASE_PATH, to the processed glossary dataset.
            source_encoder_name (str): Name of the source encoder model.
            source_encoder_name (str): Name of the target decoder model.
            glossary_name_key (str, optional): The name of the glossary key in the datums returned by the wrapper
                object. Defaults to 'glossary'.
            is_deepspeed (bool): Turns off certain features, such as token remapping, that don't work with DeepSpeed.
        """
        self.has_glossary = True
        self.glossary_name_key = glossary_name_key

        if len(self.base_dataset) == 0:
            self.glossary = None
            return
        
        glossary_dataset = os.path.join(DATA_BASE_PATH, glossary_dataset)
        with open(os.path.join(glossary_dataset, "train.bo")) as bo_f, \
             open(os.path.join(glossary_dataset, "train.en")) as en_f \
        :
            self.glossary = dict(
                zip(map(lambda x: x.strip(), bo_f.readlines()), map(lambda x: x.strip(), en_f.readlines()))
            )
        
        self.source_tokenizer = make_monolingual_tokenizer(source_encoder_name, is_deepspeed=is_deepspeed)
        self.target_tokenizer = make_monolingual_tokenizer(target_decoder_name, is_deepspeed=is_deepspeed)

        if self.glossary is None:
            return
        self.glossary_tokenized = {}
        for source, target in tqdm(self.glossary.items(), desc="Tokenizing glossary"):
            self.glossary_tokenized[source] = {
                "source_tokens": self.source_tokenizer(source, return_tensors="pt"),
                "target_tokens": self.target_tokenizer(target, return_tensors="pt")
            }

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
        if "" not in contexts:
            contexts[""] = ""

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
            for batch_idx in tqdm(range(ceil(len(contexts) / batch_size)), desc="Encoding"):
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
                    frag_name = KnowledgeInjectionDataset.hash_key(frag_name)
                    if frag_name in seen_hashes:
                        raise ValueError("Hash collision!")
                    seen_hashes.add(frag_name)
                    outputs.array(frag_name, encoding[:end_idx + 1])

    def _add_context_to_item(self, base_item):
        if self.context_store is None and not self.raw_contexts:
            raise ValueError("No prepared contexts found, first call prepare_context_embeddings.")

        if base_item.get('inject_context', True):
            context_key = base_item[self.context_lookup_key]
        else:
            context_key = ""
        context_key = self.hash_key(context_key)
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

    def _add_glossary_to_item(self, base_item):
        # TODO: The lexemes should involve shads as well - maybe strip the tshegs?
        lexemes = [k for k in self.glossary.keys() if k in base_item['source']]
        glossaries = [self.glossary_tokenized[lexeme] for lexeme in lexemes]
        if len(glossaries) == 0:
            base_item[self.glossary_name_key] = {
                'source': {
                    'input_ids': torch.Tensor([]),
                    'attention_mask': torch.Tensor([])
                },
                'target': {
                    'input_ids': torch.Tensor([]),
                    'attention_mask': torch.Tensor([])
                }
            }
        else:
            base_item[self.glossary_name_key] = {
                'source': {
                    'input_ids': torch.cat([g['source_tokens']['input_ids'][0] for g in glossaries]),
                    'attention_mask': torch.cat([g['source_tokens']['attention_mask'][0] for g in glossaries])
                },
                'target': {
                    'input_ids': torch.cat([g['target_tokens']['input_ids'][0] for g in glossaries]),
                    'attention_mask': torch.cat([g['target_tokens']['attention_mask'][0] for g in glossaries])
                }
            }
        return base_item

    def __getitem__(self, index):
        base_item = self.base_dataset[index]

        if self.has_context:
            base_item = self._add_context_to_item(base_item)
        if self.has_glossary:
            base_item = self._add_glossary_to_item(base_item)
        
        return base_item

    def __len__(self):
        return len(self.base_dataset)
