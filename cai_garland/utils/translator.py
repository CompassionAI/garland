# pylint: disable=dangerous-default-value
import logging
from typing import Optional, Dict, Any
from tqdm.auto import tqdm

from cai_common.models.utils import get_local_ckpt, get_cai_config
from cai_garland.models.factory import make_bilingual_tokenizer
from cai_garland.utils.segmenters import SegmenterNone, SegmenterOpeningShad
from cai_garland.models.siamese_encoder import SiameseEncoderModel

from transformers import EncoderDecoderModel


logger = logging.getLogger(__name__)


class TokenizationTooLongException(Exception):
    pass


def _warm_start_constraints(_batch_id, input_ids, target_tkns):
    if len(input_ids) < len(target_tkns):
        return [target_tkns[len(input_ids) - 1]]
    else:
        return slice(0, None)


class Translator:
    """A machine translation utility class, abstracting the pipeline from a potentially very long source document to a
    machine translated output. See cai_garland.cli.translate for usage examples.

    Attributes:
        model: An EncoderDecoderModel for the fine-tuned encoder-decoder translation stack.
        tokenizer: A BilingualTokenizer for the source and target languages.
        num_beams: Number of beams to use in the beam search (default is 20).
        hard_segmenter: Which hard segmenter from cai_garland.utils.segmenters to use for batch translation.
        soft_segmenter: Which soft segmenter from cai_garland.utils.segmenters to use for batch translation.
        preprocessors: List of preprocessors from cai_garland.utils.str_processors to use for batch translation.
        postprocessors: List of postprocessors from cai_garland.utils.str_processors to use for batch translation.
    """

    hard_segmenter = SegmenterOpeningShad()
    preprocessors = []
    soft_segmenter = SegmenterNone()
    postprocessors = []

    def __init__(self, model_ckpt: str) -> None:
        """Loads all the relevant data and models for machine translation.

        Args:
            model_ckpt:  Name of the fine-tuned model checkpoint in the data registry to use for translation. For
                example, olive-cormorant-bart."""
        local_ckpt = get_local_ckpt(model_ckpt, model_dir=True)
        logger.debug(f"Local model checkpoint {model_ckpt} resolved to {local_ckpt}")

        self.model = EncoderDecoderModel.from_pretrained(local_ckpt)
        logger.debug(f"Encoder: {self.model.encoder}")
        logger.debug(f"Decoder: {self.model.decoder}")

        logger.debug("Loading CAI translation model config")
        cai_base_config = get_cai_config(model_ckpt)
        encoder_name = cai_base_config['encoder_model_name']
        encoder_length = cai_base_config['encoder_max_length']
        decoder_name = cai_base_config['decoder_model_name']
        decoder_length = cai_base_config['decoder_max_length']
        logger.debug(f"Encoder name={encoder_name}, length={encoder_length}")
        logger.debug(f"Decoder name={decoder_name}, length={decoder_length}")
        self.model.encoder.max_length = encoder_length
        self.model.decoder.max_length = decoder_length

        logger.debug("Loading bilingual tokenizer")
        self.tokenizer = make_bilingual_tokenizer(encoder_name, decoder_name)

        logger.debug("Configuring model")
        self.model.eval()

        self.num_beams = 20
        self._cuda = False

    def cuda(self) -> None:
        self._cuda = True
        self.model.cuda()

    def cpu(self) -> None:
        self._cuda = False
        self.model.cpu()

    def translate(self, bo_text: str, prefix: Optional[str]=None, generator_kwargs: Dict[Any, Any]={}) -> str:
        """Translate the input Tibtean.

        Args:
            bo_text: The Tibetan text (not tokens) to translate, as a unicode string.

        Returns:
            The translated text (not tokens)."""

        bo_tokens = self.tokenizer(bo_text, return_tensors="pt").input_ids

        if isinstance(self.model.encoder, SiameseEncoderModel):
            splits = self.model.encoder.split_tokens_into_registers(bo_tokens)['input_ids']
            if any(len(tokens[0]) > self.model.encoder.max_length for tokens in splits):
                raise TokenizationTooLongException(f"Translation input too long: encoder maximum length is "
                    f"{self.model.encoder.max_length}.")
        else:
            if len(bo_tokens[0]) > self.model.encoder.max_length:
                raise TokenizationTooLongException(f"Translation input too long: encoder maximum length is "
                    f"{self.model.encoder.max_length}, input tokenizes to {len(bo_tokens[0])} "
                    f"tokens.")

        logger.debug(f"Tokenized input: {bo_tokens[0]}")
        logger.debug(f"Tokenized input length: {len(bo_tokens[0])}")

        if prefix is not None:
            with self.tokenizer.as_target_tokenizer():
                prefix_tokens = self.tokenizer(prefix).input_ids[:-1]
                prefix_fn = lambda batch_id, input_ids: _warm_start_constraints(
                    batch_id, input_ids, prefix_tokens)
        else:
            prefix_fn = None

        if self._cuda:
            bo_tokens = bo_tokens.cuda()
        preds = self.model.generate(
            bo_tokens,
            max_length=self.model.decoder.max_length,
            num_beams=self.num_beams,
            prefix_allowed_tokens_fn=prefix_fn,
            **generator_kwargs
        )[0]
        if self._cuda:
            preds = preds.cpu()

        logger.debug(f"Generated tokens: {preds}")
        logger.debug(f"Generated tokens length: {len(preds)}")
        with self.tokenizer.as_target_tokenizer():
            return self.tokenizer.decode(preds, skip_special_tokens=True).strip()


    def batch_translate(
        self,
        bo_text,
        tqdm=tqdm,      # pylint: disable=redefined-outer-name
        hard_segmenter_kwargs={},
        soft_segmenter_kwargs={},
        retrospective_registers=False,
        throw_translation_errors=False,
        generator_kwargs={}
    ):
        hard_segments = self.hard_segmenter(bo_text, translator=self, **hard_segmenter_kwargs)

        for hard_segment in tqdm(hard_segments, desc="Hard segments", leave=False):
            for preproc_func in self.preprocessors:
                hard_segment = preproc_func(hard_segment)

            soft_segments = self.soft_segmenter(hard_segment, translator=self, **soft_segmenter_kwargs)

            if retrospective_registers:
                src_registers, tgt_registers = [], []
                num_registers = self.model.encoder.config.num_registers
            for soft_segment in tqdm(soft_segments, desc="Soft segments", leave=False):
                if retrospective_registers:
                    input_ = self.tokenizer.source_tokenizer.eor_token.join(src_registers + [soft_segment])
                    prefix = ' '.join(tgt_registers)
                else:
                    input_ = soft_segment
                    prefix = None

                translation_err = False
                try:
                    tgt_segment = self.translate(input_, prefix=prefix, generator_kwargs=generator_kwargs)
                except TokenizationTooLongException as err:
                    if throw_translation_errors:
                        raise err
                    else:
                        translation_err = True
                        tgt_segment = "SEGMENT TOKENIZATION TOO LONG FOR ENCODER MODEL"

                if retrospective_registers:
                    if translation_err:
                        src_registers, tgt_registers = [], []
                    else:
                        tgt_segment = tgt_segment[len(prefix):].strip()
                        src_registers.append(soft_segment)
                        tgt_registers.append(tgt_segment)
                        if len(src_registers) == num_registers:
                            src_registers.pop(0)
                            tgt_registers.pop(0)

                for postproc_func in self.postprocessors:
                    tgt_segment = postproc_func(tgt_segment)

                yield soft_segment, tgt_segment
