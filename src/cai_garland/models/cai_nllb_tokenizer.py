import os
import torch

from tqdm.auto import tqdm
from transformers import NllbTokenizerFast
from tokenizers import processors

from cai_common.models.utils import get_local_file


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
        self.apply_token_remapping = False
        self.used_tokens = range(self.vocab_size)

    def remap_tokens(self, remapping_file):
        with open(get_local_file(remapping_file), 'r') as f:
            self.used_tokens = list(map(int, f.readlines()))
        self.tokenizer_remapping_forward = {
            token_id: line_num
            for line_num, token_id in enumerate(self.used_tokens)
        }
        self.tokenizer_remapping_backward = {
            line_num: token_id
            for line_num, token_id in enumerate(self.used_tokens)
        }
        self.apply_token_remapping = True

    def make_remapping_file(self, lines, remapping_file, lang_names, lines_tokens_only=False):
        """The lines are intended to be all lines from all splits of a dataset that we want to tokenize. Pass in
            lines_tokens_only=True if you want only tokens that actually appear in the lines argument. Otherwise it will
            remap to tokens that have characters that appear in the lines argument."""
        special_tokens = [
            self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id, self.mask_token_id]
        special_tokens.extend([self.lang_code_to_id[lang] for lang in lang_names])
        if lines_tokens_only:
            res_vocab = sorted(list(set([t for l in tqdm(lines) for t in self.encode(l)]) | set(special_tokens)))
        else:
            all_chars = set([c for l in lines for c in l] + ["▁"])
            res_vocab = sorted(
                self.convert_tokens_to_ids(
                    list(filter(lambda v: all([c in all_chars for c in v]), self.vocab.keys()))
                ) + special_tokens
            )
        with open(os.path.join(os.environ['CAI_DATA_BASE_PATH'], remapping_file), 'w') as f:
            f.writelines(map(lambda x: str(x) + '\n', res_vocab))

    def language_id(self, language_code):
        if not self.apply_token_remapping or self.tokenizer_remapping_forward is None:
            return self.lang_code_to_id[language_code]
        return self.tokenizer_remapping_forward[self.lang_code_to_id[language_code]]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No suffix and prefix=[eos, tgt_lang_code].
            NB that this appears to be wrong in the HF NLLB tokenizer!"""
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        self.prefix_tokens = [self.cur_lang_code, self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

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
        if self.apply_token_remapping and self.tokenizer_remapping_forward is not None:
            input_ids = [
                [self.tokenizer_remapping_forward.get(int(x), self.unk_token_id) for x in ex]
                for ex in input_.input_ids
            ]

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
        if self.apply_token_remapping and self.tokenizer_remapping_backward is not None:
            kwargs['token_ids'] = [
                self.tokenizer_remapping_backward.get(int(x), self.unk_token_id) for x in kwargs['token_ids']]
        return super()._decode(*args, **kwargs)

    @property
    def is_remapped(self):
        return self.tokenizer_remapping_forward is not None
