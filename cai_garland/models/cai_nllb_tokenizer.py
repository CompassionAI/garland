import os

from transformers import NllbTokenizerFast


class CAINllbTokenizerFast(NllbTokenizerFast):
    """A wrapper class for the Transformers NLLB tokenizer that does two things:
        1. Fixes the language token placement for the target language. This is toggleable in case they fix it upstream
            via the fix_nllb_tokenizer_target_language_tokens class member.
        2. Allows for remapping the token space to a subset of the tokens. Useful for making Nllb less wasteful when
            only using a subset of the 200 languages it supports.
    """

    fix_nllb_tokenizer_target_language_tokens = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_mode = False
        self.target_tokenizer_remapping_forward = None
        self.target_tokenizer_remapping_backward = None
        self.used_tokens = range(self.vocab_size)

    def remap_tokens(self, remapping_file):
        with open(os.path.join(os.environ['CAI_DATA_BASE_PATH'], remapping_file), 'r') as f:
            self.used_tokens = list(map(int, f.readlines()))
        self.target_tokenizer_remapping_forward = {
            token_id: line_num
            for line_num, token_id in enumerate(self.used_tokens)
        }
        self.target_tokenizer_remapping_backward = {
            line_num: token_id
            for line_num, token_id in enumerate(self.used_tokens)
        }

    def make_remapping_file(self, lines, remapping_file):
        """The lines are intended to be all lines from all splits of a dataset that we want to tokenize."""
        all_chars = set([c for l in lines for c in l] + ["▁"])
        res_vocab = list(filter(lambda v: all([c in all_chars for c in v]), self.vocab.keys()))
        res_vocab = sorted(
            self.convert_tokens_to_ids(res_vocab) + \
                [self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
        )
        with open(os.path.join(os.environ['CAI_DATA_BASE_PATH'], remapping_file), 'w') as f:
            f.writelines(map(lambda x: str(x) + '\n', res_vocab))

    def _switch_to_input_mode(self):
        self._target_mode = False
        return super()._switch_to_input_mode()

    def _switch_to_target_mode(self):
        self._target_mode = True
        return super()._switch_to_target_mode()

    def _fix_target_tokens(self, input):
        if self.fix_nllb_tokenizer_target_language_tokens:
            input["input_ids"] = [x[-2:] + x[0:-2] + [x[-2]] for x in input.input_ids]
            input["attention_mask"] = [x + [1] for x in input.attention_mask]
        if self.target_tokenizer_remapping_forward is not None:
            input["input_ids"] = [[self.target_tokenizer_remapping_forward[x] for x in ex] for ex in input.input_ids]
        return input

    def _encode_plus(self, *args, **kwargs):
        if self._target_mode:
            return self._fix_target_tokens(super()._encode_plus(*args, **kwargs))
        return super()._encode_plus(*args, **kwargs)

    def _batch_encode_plus(self, *args, **kwargs):
        if self._target_mode:
            return self._fix_target_tokens(super()._batch_encode_plus(*args, **kwargs))
        return super()._batch_encode_plus(*args, **kwargs)

    def _decode(self, *args, **kwargs):
        if self.target_tokenizer_remapping_backward is not None:
            kwargs['token_ids'] = [self.target_tokenizer_remapping_backward[x] for x in kwargs['token_ids']]
        return self._tokenizer._decode(*args, **kwargs)
