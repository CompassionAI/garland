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
    raw_context: bool
    return_tensors: str = "pt"
    raw_pad_token_id: int = None

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if not return_tensors == "pt":
            raise ValueError("Only PyTorch tensors currently supported")

        context_key, mask_key = self.context_key, self.context_key + "_mask"
        context_features = {context_key, mask_key}
        base_features = [
            {
                key: val
                for key, val in feature.items()
                if key not in context_features
            }
            for feature in features
        ]
        res = self.base_collator(base_features)
        
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

        return res