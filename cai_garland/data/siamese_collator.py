from dataclasses import dataclass

from transformers import DataCollatorForSeq2Seq

@dataclass
class SiameseDataCollatorForSeq2Seq:
    """
    Wrapper class for the Hugging Face seq2seq data collator to enable Siamese encoders. Pads each register separately.
    """

    base_collator: DataCollatorForSeq2Seq

    def __call__(self, features, return_tensors=None):
        import ipdb; ipdb.set_trace()
