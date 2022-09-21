# pylint: disable=protected-access
from contextlib import contextmanager

from transformers import PreTrainedTokenizer


class BilingualTokenizer(PreTrainedTokenizer):
    def __init__(self, source_tokenizer, target_tokenizer, *args, **kwargs):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self._tokenizer = self.source_tokenizer
        self._target_mode = False
        self.remap_source, self.remap_target = True, True
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

    def _encode_plus(self, *args, **kwargs):
        if self._target_mode:
            with self._tokenizer.as_target_tokenizer():
                self._tokenizer.apply_token_remapping = self.remap_target
                return self._tokenizer._encode_plus(*args, **kwargs)
        self._tokenizer.apply_token_remapping = self.remap_source
        return self._tokenizer._encode_plus(*args, **kwargs)

    def _tokenize(self, text, **kwargs):
        if self._target_mode:
            with self._tokenizer.as_target_tokenizer():
                self._tokenizer.apply_token_remapping = self.remap_target
                return self._tokenizer._tokenize(text, **kwargs)
        self._tokenizer.apply_token_remapping = self.remap_source
        return self._tokenizer._tokenize(text, **kwargs)

    def tokenize(self, text, **kwargs):
        if self._target_mode:
            with self._tokenizer.as_target_tokenizer():
                self._tokenizer.apply_token_remapping = self.remap_target
                return self._tokenizer.tokenize(text, **kwargs)
        self._tokenizer.apply_token_remapping = self.remap_source
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
        if self._target_mode:
            with self._tokenizer.as_target_tokenizer():
                self._tokenizer.apply_token_remapping = self.remap_target
                return self._tokenizer._batch_encode_plus(*args, **kwargs)
        self._tokenizer.apply_token_remapping = self.remap_source
        return self._tokenizer._batch_encode_plus(*args, **kwargs)

    def _decode(self, *args, **kwargs):
        if self._target_mode:
            with self._tokenizer.as_target_tokenizer():
                self._tokenizer.apply_token_remapping = self.remap_target
                return self._tokenizer._decode(*args, **kwargs)
        self._tokenizer.apply_token_remapping = self.remap_source
        return self._tokenizer._decode(*args, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1 = None):
        return self._tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1=token_ids_1)

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return self._tokenizer.save_vocabulary(save_directory, filename_prefix)

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def save_pretrained(
        self,
        save_directory,
        legacy_format = None,
        filename_prefix = None,
        push_to_hub = False,
        **kwargs,
    ):
        if push_to_hub:
            raise ValueError("Bilingual tokenizer cannot be pushed to Hugging Face hub")
        prefix_array = [] if filename_prefix is None else [filename_prefix]
        self.source_tokenizer.save_pretrained(
            save_directory, legacy_format=legacy_format, filename_prefix='-'.join(prefix_array + ["source"]), **kwargs)
        self.target_tokenizer.save_pretrained(
            save_directory, legacy_format=legacy_format, filename_prefix='-'.join(prefix_array + ["target"]), **kwargs)

    def _switch_to_input_mode(self):
        self._tokenizer = self.source_tokenizer
        self._target_mode = False
        return super()._switch_to_input_mode()

    def _switch_to_target_mode(self):
        self._tokenizer = self.target_tokenizer
        self._target_mode = True
        return super()._switch_to_target_mode()

    @contextmanager
    def as_target_tokenizer(self):
        """Switches to the target tokenizer within the context returned by the manager."""
        self._tokenizer = self.target_tokenizer
        self._target_mode = True
        self._in_target_context_manager = True
        yield
        self._target_mode = False
        self._in_target_context_manager = False
        self._tokenizer = self.source_tokenizer
