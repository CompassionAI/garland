from contextlib import contextmanager

from transformers import PreTrainedTokenizerBase


class BilingualTokenizer(PreTrainedTokenizerBase):
    def __init__(self, source_tokenizer, target_tokenizer, *args, **kwargs):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self._tokenizer = self.source_tokenizer
        super().__init__(
            bos_token=self.source_tokenizer.bos_token,
            eos_token=self.source_tokenizer.eos_token,
            unk_token=self.source_tokenizer.unk_token,
            sep_token=self.source_tokenizer.sep_token,
            pad_token=self.source_tokenizer.pad_token,
            cls_token=self.source_tokenizer.cls_token,
            mask_token=self.source_tokenizer.mask_token,
            *args,
            **kwargs)

    def _tokenize(self, text, **kwargs):
        return self._tokenizer._tokenize(text, **kwargs)

    def tokenize(self, text, **kwargs):
        return self._tokenizer.tokenize(text, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens = False):
        return self._tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def _convert_token_to_id(self, token):
        return self._tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        return self._tokenizer._convert_id_to_token(index)

    def _batch_encode_plus(self, *args, **kwargs):
        return self._tokenizer._batch_encode_plus(*args, **kwargs)

    def _decode(self, *args, **kwargs):
        return self._tokenizer._decode(*args, **kwargs)

    def save_pretrained(
        self,
        save_directory,
        legacy_format = None,
        filename_prefix = None
    ):
        prefix_array = [] if filename_prefix is None else [filename_prefix]
        self.source_tokenizer.save_pretrained(
            save_directory, legacy_format=legacy_format, filename_prefix='-'.join(prefix_array + ["source"]))
        self.target_tokenizer.save_pretrained(
            save_directory, legacy_format=legacy_format, filename_prefix='-'.join(prefix_array + ["target"]))

    @contextmanager
    def as_target_tokenizer(self):
        """Switches to the target tokenizer within the context returned by the manager."""
        self._tokenizer = self.target_tokenizer
        yield
        self._tokenizer = self.source_tokenizer
