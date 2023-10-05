import torch

import numpy as np
from transformers import DataCollatorForSeq2Seq


class KnowledgeDataCollatorForSeq2Seq:
    """
    Wrapper class for the Hugging Face seq2seq data collator to enable context injection.
    """

    def __init__(self, base_collator, return_tensors="pt"):
        self.base_collator = base_collator
        self.return_tensors = return_tensors
        self.has_context = False
        self.has_glossary = False

    def setup_context(self, context_key, raw_context, raw_pad_token_id):
        self.has_context = True
        self.context_key = context_key
        self.raw_pad_token_id = raw_pad_token_id
        self.raw_context = raw_context

    def setup_glossary(self, glossary_key):
        self.has_glossary = True
        self.glossary_key = glossary_key

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if not return_tensors == "pt":
            raise ValueError("Only PyTorch tensors currently supported")

        knowledge_features = set()
        if self.has_context:
            context_key, mask_key = self.context_key, self.context_key + "_mask"
            knowledge_features |= {context_key, mask_key}
        if self.has_glossary:
            knowledge_features |= {self.glossary_key}

        base_features = [
            {
                key: val
                for key, val in feature.items()
                if key not in knowledge_features
            }
            for feature in features
        ]
        res = self.base_collator(base_features)

        if self.has_context:
            if self.raw_context:
                if self.raw_pad_token_id is None:
                    raise ValueError("Specify the pad token id for raw context datasets")

                pad_to = max([len(f[self.context_key]) for f in features])
                res[mask_key] = [
                    f + [0] * (pad_to - len(f))
                    for f in [f[mask_key] for f in features]
                ]
                res[context_key] = [
                    f + [self.raw_pad_token_id] * (pad_to - len(f))
                    for f in [f[context_key] for f in features]
                ]
                res[mask_key] = torch.LongTensor(res[mask_key])
                res[context_key] = torch.LongTensor(res[context_key])
            else:
                pad_to = max([f[self.context_key].shape[0] for f in features])

                key = mask_key
                stack = []
                for feature in features:
                    stack.append(np.pad(feature[key], (0, pad_to - feature[key].shape[0])))
                res[key] = np.vstack(stack)

                key = context_key
                stack = []
                for feature in features:
                    stack.append(np.pad(feature[key], ((0, pad_to - feature[key].shape[0]), (0, 0)))[np.newaxis, ...])
                res[key] = np.vstack(stack)

                res[mask_key] = torch.LongTensor(res[mask_key])
                res[context_key] = torch.FloatTensor(res[context_key])
        if self.has_glossary:
            res[self.glossary_key] = {
                'source': self.base_collator([f[self.glossary_key]['source'] for f in features]),
                'target': self.base_collator([f[self.glossary_key]['target'] for f in features])
            }

        return res