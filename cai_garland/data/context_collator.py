import torch

import numpy as np

from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq

@dataclass
class ContextDataCollatorForSeq2Seq:
    """
    Wrapper class for the Hugging Face seq2seq data collator to enable context injection.
    """

    base_collator: DataCollatorForSeq2Seq
    context_key: str
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        context_key, attention_key = self.context_key, self.context_key + "_attention_mask"
        context_features = {context_key, attention_key}
        base_features = [
            {
                key: val
                for key, val in feature.items()
                if key not in context_features
            }
            for feature in features
        ]
        res = self.base_collator(base_features)
        
        pad_to = max([f[self.context_key].shape[0] for f in features])

        key = attention_key
        stack = []
        for feature in features:
            stack.append(np.pad(feature[key], (0, pad_to - feature[key].shape[0])))
        res[key] = np.vstack(stack)

        key = context_key
        stack = []
        for feature in features:
            stack.append(np.pad(feature[key], ((0, pad_to - feature[key].shape[0]), (0, 0)))[np.newaxis, ...])
        res[key] = np.vstack(stack)

        if not return_tensors == "pt":
            raise ValueError("Only PyTorch tensors currently supported")
        res[attention_key] = torch.LongTensor(res[attention_key])
        res[context_key] = torch.FloatTensor(res[context_key])

        return res