from dataclasses import dataclass

from transformers import DataCollatorForSeq2Seq


def _split_list(list_, split_idxs):
    if len(split_idxs) == 0:
        return [list_]
    res, last_idx = [], 0
    for cur_idx in split_idxs:
        res.append(list_[last_idx:cur_idx])
        last_idx = cur_idx + 1
    return res + [list_[last_idx:]]


@dataclass
class SiameseDataCollatorForSeq2Seq:
    """
    Wrapper class for the Hugging Face seq2seq data collator to enable Siamese encoders. Pads each register separately.

    *NB*: This wrapper class assumes the base encoder has the bos and eos tokens at the start and end of the input.
    """

    base_collator: DataCollatorForSeq2Seq
    num_registers: int
    eor_token_id: int

    def __call__(self, features, return_tensors=None):
        registers_features = [
            [
                {
                    key: [value[0], value[-1]]      # Assumes the base encoder has bos/eos tokens at the start and end!
                    for key, value in feature.items() if not key == "labels"
                }
                for feature in features
            ]
            for _ in range(self.num_registers)
        ]

        for feat_idx, feature in enumerate(features):
            tokens = feature['input_ids'][1:-1]         # Strip bos and eos, will add back in each register
            register_splits = [idx for idx, token in enumerate(tokens) if token == self.eor_token_id]

            for key, values in feature.items():
                if not key == "labels":     # We add the labels at the end instead of duplicating them in each register
                    bos, eos = values[0], values[-1]
                    register_values = _split_list(values[1:-1], register_splits)
                    for register_idx, register_value in enumerate(register_values):
                        registers_features[register_idx][feat_idx][key] = [bos] + register_value + [eos]

        features_registers = {
            key: [None for _ in range(self.num_registers)]
            for key in features[0].keys() if not key == "labels"
        }
        for register_idx, register_features in enumerate(registers_features):
            collated = self.base_collator(register_features, return_tensors=return_tensors)
            for key, value in collated.items():
                features_registers[key][register_idx] = value

        bos, eos = features[0]['input_ids'][0], features[0]['input_ids'][-1]
        features_labels = [
            {
                "input_ids": [bos, eos],        # Need some kind of stub here, safer to use something valid
                "labels": feature['labels']
            }
            for feature in features
        ]
        collated_labels = self.base_collator(features_labels, return_tensors=return_tensors)

        return collated_labels | features_registers     # We excluded labels from features_registers, so this is safe
