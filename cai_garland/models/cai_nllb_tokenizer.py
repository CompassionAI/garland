import os
import torch

from transformers import NllbTokenizerFast


class CAINllbTokenizerFast(NllbTokenizerFast):
    """A wrapper class for the Transformers NLLB tokenizer that does two things:
        1. Fixes the language token placement for the target language. This is toggleable in case they fix it upstream
            via the fix_nllb_tokenizer_target_language_tokens class member.
        2. Allows for remapping the token space to a subset of the tokens. Useful for making Nllb less wasteful when
            only using a subset of the 200 languages it supports.
    """

    fix_nllb_tokenizer_target_language_tokens = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_remapping_forward = None
        self.tokenizer_remapping_backward = None
        self.used_tokens = range(self.vocab_size)

    def remap_tokens(self, remapping_file):
        with open(os.path.join(os.environ['CAI_DATA_BASE_PATH'], remapping_file), 'r') as f:
            self.used_tokens = list(map(int, f.readlines()))
        self.tokenizer_remapping_forward = {
            token_id: line_num
            for line_num, token_id in enumerate(self.used_tokens)
        }
        self.tokenizer_remapping_backward = {
            line_num: token_id
            for line_num, token_id in enumerate(self.used_tokens)
        }

    def make_remapping_file(self, lines, remapping_file, lang_names):
        """The lines are intended to be all lines from all splits of a dataset that we want to tokenize."""
        all_chars = set([c for l in lines for c in l] + ["▁"])
        res_vocab = list(filter(lambda v: all([c in all_chars for c in v]), self.vocab.keys()))
        special_tokens = [
            self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id, self.mask_token_id]
        special_tokens.extend([self.lang_code_to_id[lang] for lang in lang_names])
        res_vocab = sorted(self.convert_tokens_to_ids(res_vocab) + special_tokens)
        with open(os.path.join(os.environ['CAI_DATA_BASE_PATH'], remapping_file), 'w') as f:
            f.writelines(map(lambda x: str(x) + '\n', res_vocab))

    def language_id(self, language_code):
        return self.tokenizer_remapping_forward[self.lang_code_to_id[language_code]]

    def _fix_target_tokens(self, input_):
        if getattr(input_, "tokens_fixed", False):
            return input_
        input_ids, attention_mask = input_.input_ids, input_.attention_mask

        if isinstance(input_ids, list):
            repack = not isinstance(input_ids[0], list)
        else:
            repack = len(input_ids.shape) == 1
        if repack:
            input_ids, attention_mask = [input_ids], [attention_mask]

        if self.fix_nllb_tokenizer_target_language_tokens:
            input_ids = [x[-2:] + x[0:-2] + [x[-2]] for x in input_.input_ids]
            attention_mask = [x + [1] for x in input_.attention_mask]
        if self.tokenizer_remapping_forward is not None:
            input_ids = [[self.tokenizer_remapping_forward[int(x)] for x in ex] for ex in input_.input_ids]

        if repack:
            input_ids, attention_mask = input_ids[0], attention_mask[0]
        if isinstance(input_["input_ids"], torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=input_["input_ids"].dtype, device=input_["input_ids"].device)
            attention_mask = torch.tensor(
                attention_mask, dtype=input_["attention_mask"].dtype, device=input_["attention_mask"].device)
        input_["input_ids"], input_["attention_mask"] = input_ids, attention_mask

        input_["tokens_fixed"] = [1]
        return input_

    def _encode_plus(self, *args, **kwargs):
        return self._fix_target_tokens(super()._encode_plus(*args, **kwargs))

    def _batch_encode_plus(self, *args, **kwargs):
        return self._fix_target_tokens(super()._batch_encode_plus(*args, **kwargs))

    def _decode(self, *args, **kwargs):
        if self.tokenizer_remapping_backward is not None:
            kwargs['token_ids'] = [self.tokenizer_remapping_backward[int(x)] for x in kwargs['token_ids']]
        return super()._decode(*args, **kwargs)

    @property
    def is_remapped(self):
        return self.tokenizer_remapping_forward is not None
