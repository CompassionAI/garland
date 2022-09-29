# pylint: disable=dangerous-default-value
# pylint: disable=no-member
# pylint: disable=protected-access
import logging
from typing import Optional, Dict, Any
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from cai_common.models.utils import get_local_ckpt, get_cai_config
from cai_garland.models.factory import make_bilingual_tokenizer
from cai_garland.utils.segmenters import SegmenterNone, SegmenterOpeningShad
from cai_garland.models.siamese_encoder import SiameseEncoderModel, BaseModelOutputWithAttentionMask

from cai_garland.models.cai_encoder_decoder import CAIEncoderDecoderModel


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
        logger.info(f"Local model checkpoint {model_ckpt} resolved to {local_ckpt}")

        self.model = CAIEncoderDecoderModel.from_pretrained(local_ckpt)
        logger.debug(f"Encoder: {self.model.encoder}")
        logger.debug(f"Decoder: {self.model.decoder}")

        logger.info("Loading CAI translation model config")
        cai_base_config = get_cai_config(model_ckpt)
        encoder_name = cai_base_config['encoder_model_name']
        encoder_length = cai_base_config['encoder_max_length']
        decoder_name = cai_base_config['decoder_model_name']
        decoder_length = cai_base_config['decoder_max_length']
        logger.info(f"Encoder name={encoder_name}, length={encoder_length}")
        logger.info(f"Decoder name={decoder_name}, length={decoder_length}")
        self.model.encoder.max_length = encoder_length
        self.model.decoder.max_length = decoder_length

        logger.info("Loading bilingual tokenizer")
        self.tokenizer = make_bilingual_tokenizer(encoder_name, decoder_name)

        if cai_base_config.get("reset_token_vocab", False):
            logger.info("Resetting token vocabulary")
            from transformers import M2M100ForConditionalGeneration
            nllb_model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M")
            self.model.decoder.model.decoder.embed_tokens = nllb_model.model.shared
            self.model.decoder.lm_head = nllb_model.lm_head
            self.tokenizer.remap_target = False

        logger.info("Configuring model")
        self.model.eval()

        self.num_beams = 20
        self._cuda = False
        self.context_encoder = None
        self.decoding_length = self.model.decoder.max_length
        self._bad_words, self._bad_word_tokens = [], []

    def cuda(self) -> None:
        self._cuda = True
        self.model.cuda()
        if self.context_encoder is not None:
            self.context_encoder.cuda()

    def cpu(self) -> None:
        self._cuda = False
        self.model.cpu()
        if self.context_encoder is not None:
            self.context_encoder.cpu()

    @property
    def bad_words(self):
        return self._bad_words

    @bad_words.setter
    def bad_words(self, value):
        self._bad_words = value
        with self.tokenizer.as_target_tokenizer():
            self._bad_word_tokens = [self.tokenizer.encode(w, add_special_tokens=False) for w in self._bad_words]

    def prepare_context_encoder(self, hf_model_name):
        """Prepare a context encoder for translation with context.
        
        Args:
            hf_model_name: A model name in the Hugging Face model registry for context encoding.
        """
        logger.info("Loading context tokenizer")
        self.context_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        logger.info("Loading context encoder")
        self.context_encoder = AutoModelForMaskedLM.from_pretrained(hf_model_name)
        if getattr(self.context_encoder.config, "is_encoder_decoder", False):
            self.context_encoder = self.context_encoder.model.encoder
        self.context_encoder.eval()
        if self._cuda:
            self.context_encoder.cuda()

    def _encode_text(
        self,
        bo_text: str
    ) -> Any:
        # Utility function to run only the encoder on source text. Does _not_ accept eor tokens.
        bo_tokens = self.tokenizer(bo_text, return_tensors="pt").input_ids

        if len(bo_tokens[0]) > self.model.encoder.max_length:
            raise TokenizationTooLongException(f"Translation input too long: encoder maximum length is "
                f"{self.model.encoder.max_length}, input tokenizes to {len(bo_tokens[0])} "
                f"tokens.")

        if self._cuda:
            bo_tokens = bo_tokens.cuda()
        bo_tokens = [bo_tokens]
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            bo_tokens, self.model.config.bos_token_id, model_kwargs={})
        model_kwargs["attention_mask"] = [self.model._prepare_attention_mask_for_generation(
            inputs_tensor[0],
            self.model.config.pad_token_id,
            self.model.config.eos_token_id
        )]
        model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name)

        del bo_tokens
        encoder_outputs = model_kwargs['encoder_outputs']
        del model_kwargs
        if self._cuda:
            last_hidden_state = encoder_outputs.last_hidden_state.cpu().detach().numpy()
            attention_mask = encoder_outputs.attention_mask.cpu().detach().numpy()
        torch.cuda.empty_cache()
        return {
            "last_hidden_state": last_hidden_state,
            "attention_mask": attention_mask
        }

    def translate(
        self,
        bo_text: str,
        prefix: Optional[str]=None,
        context: Optional[str]=None,
        target_language_code: Optional[str]=None,
        encoder_outputs: Any = None,
        generator_kwargs: Dict[Any, Any]={},
    ) -> str:
        """Translate the input Tibtean.

        Args:
            bo_text: The Tibetan text (not tokens) to translate, as a unicode string.
            prefix (optional): Prefix text, in the target language, to force the generator to produce.
            context (optional): Context, in the target language, for the the causal language model. Note that this
                requires a model with a decoder that accepts context and is trained with context, otherwise your
                translation will crash. You need to initialize the context model by calling prepare_context_encoder
                before running this.
            target_language_code (optional): The language code for the target language for multilingual decoders. For
                example: ita_Latn.
            encoder_outputs (optional): Pre-computed encoder outputs. If not specified, the model will run the encoder.
            generator_kwargs (optional): Any additional keyword arguments to pass to the generator function.

        Returns:
            The translated text (not tokens)."""

        if context is not None and self.context_encoder is None:
            raise ValueError("Must specify a context encoder if translating with context. Call prepare_context_encoder "
                             "to do this.")

        bo_tokens = self.tokenizer(bo_text, return_tensors="pt").input_ids
        if context is not None:
            ctx_tokens = self.context_tokenizer(context, return_tensors="pt")

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
        if context is not None:
            if len(ctx_tokens.input_ids[0]) > self.context_tokenizer.model_max_length:
                raise TokenizationTooLongException(f"Context too long: context encoder maximum length is "
                    f"{self.context_encoder.max_length}, input tokenizes to {len(ctx_tokens.input_ids[0])} "
                    f"tokens.")

        logger.debug(f"Tokenized input: {bo_tokens[0]}")
        logger.debug(f"Tokenized input length: {len(bo_tokens[0])}")
        if context is not None:
            logger.debug(f"Tokenized context length: {len(ctx_tokens.input_ids[0])}")

        if prefix is not None:
            with self.tokenizer.as_target_tokenizer():
                prefix_tokens = self.tokenizer(prefix).input_ids[:-1]
                prefix_fn = lambda batch_id, input_ids: _warm_start_constraints(
                    batch_id, input_ids, prefix_tokens)
        else:
            prefix_fn = None

        if context is not None:
            logger.debug("Encoding context")
            ctx_embedding = self.context_encoder(**ctx_tokens.to(self.context_encoder.device)).last_hidden_state
            ctx_mask = ctx_tokens.attention_mask
        else:
            ctx_embedding, ctx_mask = None, None

        if isinstance(encoder_outputs, list):
            encoder_outputs = BaseModelOutputWithAttentionMask(
                last_hidden_state=torch.cat(
                    [
                        torch.FloatTensor(encoder_output['last_hidden_state'])
                        for encoder_output in encoder_outputs
                    ],
                    dim=1
                ),
                attention_mask=torch.cat(
                    [
                        torch.LongTensor(encoder_output['attention_mask'])
                        for encoder_output in encoder_outputs
                    ],
                    dim=1
                )
            )
        # Need type(...) here because all classes are isinstance of dict
        elif type(encoder_outputs) is dict:     # pylint: disable=unidiomatic-typecheck
            encoder_outputs = BaseModelOutputWithAttentionMask(
                last_hidden_state=torch.FloatTensor(encoder_outputs['last_hidden_state']),
                attention_mask=torch.LongTensor(encoder_outputs['attention_mask'])
            )
        if self._cuda:
            bo_tokens = bo_tokens.cuda()
            if encoder_outputs is not None:
                if isinstance(encoder_outputs.last_hidden_state, list):
                    encoder_outputs.last_hidden_state = [t.cuda() for t in encoder_outputs.last_hidden_state]
                    encoder_outputs.attention_mask = [t.cuda() for t in encoder_outputs.attention_mask]
                else:
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.cuda()
                    encoder_outputs.attention_mask = encoder_outputs.attention_mask.cuda()
        if encoder_outputs is not None:
            generator_kwargs['encoder_outputs'] = encoder_outputs
            generator_kwargs['attention_mask'] = encoder_outputs.attention_mask
        with self.model.prepare_model_for_generation(ctx_embedding, ctx_mask):
            if target_language_code is None:
                language_token = self.model.forced_bos_token_id(self.tokenizer)
            else:
                language_token = self.tokenizer.target_tokenizer.lang_code_to_id[target_language_code]
            preds = self.model.generate(
                bo_tokens,
                max_length=self.decoding_length,
                forced_bos_token_id=language_token,
                num_beams=self.num_beams,
                prefix_allowed_tokens_fn=prefix_fn,
                bad_words_ids=None if len(self._bad_word_tokens) == 0 else self._bad_word_tokens,   # Needs to be None instead of empty, otherwise HF throws ValueError
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
        retrospective_decoding=False,
        retrospective_registers=False,
        retrospection_window=None,
        contextual_decoding=False,
        context_window_words=50,
        context_window_characters=1000,
        throw_translation_errors=False,
        target_language_code=None,
        generator_kwargs={}
    ):
        if retrospective_decoding and contextual_decoding:
            raise ValueError("Specify either retrospective or contextual decoding, not both.")

        hard_segments = self.hard_segmenter(bo_text, translator=self, **hard_segmenter_kwargs)

        for hard_segment in tqdm(hard_segments, desc="Hard segments", leave=False):
            for preproc_func in self.preprocessors:
                hard_segment = preproc_func(hard_segment)

            soft_segments = self.soft_segmenter(hard_segment, translator=self, **soft_segmenter_kwargs)

            for preproc_func in self.soft_segment_preprocessors:
                soft_segments = list(map(preproc_func, soft_segments))

            if getattr(self, "soft_segment_combiner_config", None) is not None:
                new_soft_segments = soft_segments[:self.soft_segment_combiner_config.skip_first_N]
                for seg_idx in range(
                    self.soft_segment_combiner_config.skip_first_N,
                    len(soft_segments),
                    self.soft_segment_combiner_config.combine_window
                ):
                    new_soft_segments.append(
                        ' '.join(soft_segments[seg_idx:seg_idx + self.soft_segment_combiner_config.combine_window]))
                soft_segments = new_soft_segments

            if retrospective_decoding:
                src_registers, tgt_registers = [], []
                if retrospection_window is None:
                    retrospection_window = self.model.encoder.config.num_registers

                # if retrospective_registers:
                #     all_encoder_outputs = []
                #     for soft_segment in tqdm(soft_segments, desc="Encoding soft segments", leave=False):
                #         all_encoder_outputs.append(self._encode_text(soft_segment))
            if contextual_decoding:
                context_window = ""
            else:
                context_window = None

            for seg_idx, soft_segment in tqdm(
                enumerate(soft_segments),
                total=len(soft_segments),
                desc="Translating soft segments",
                leave=False
            ):
                if retrospective_decoding:
                    encoder_outputs = None
                    if retrospective_registers:
                        input_ = self.tokenizer.source_tokenizer.eor_token.join(src_registers + [soft_segment])
                        # encoder_outputs = [
                        #     all_encoder_outputs[seg_idx - i] for i in reversed(range(len(src_registers) + 1))]
                    else:
                        input_ = ' '.join(src_registers + [soft_segment])
                    prefix = ''.join(tgt_registers)
                    if len(prefix) > 0:
                        prefix += ' '
                else:
                    input_ = soft_segment
                    prefix = None
                    encoder_outputs = None

                translation_err = False
                try:
                    tgt_segment = self.translate(
                        input_,
                        prefix=prefix,
                        context=context_window,
                        target_language_code=target_language_code,
                        encoder_outputs=encoder_outputs,
                        generator_kwargs=generator_kwargs
                    )
                except TokenizationTooLongException as err:
                    if throw_translation_errors:
                        raise err
                    else:
                        translation_err = True
                        tgt_segment = "SEGMENT TOKENIZATION TOO LONG FOR ENCODER MODEL"

                for postproc_func in self.postprocessors:
                    tgt_segment = postproc_func(tgt_segment)

                if retrospective_decoding:
                    if translation_err:
                        src_registers, tgt_registers = [], []
                    else:
                        tgt_segment = tgt_segment[len(prefix):].strip()
                        src_registers.append(soft_segment)
                        tgt_registers.append(tgt_segment)
                        if len(src_registers) == retrospection_window:
                            src_registers.pop(0)
                            tgt_registers.pop(0)
                if contextual_decoding:
                    if not tgt_segment.startswith(' ') and not context_window.endswith(' '):
                        context_window += ' '
                    context_window = (context_window + tgt_segment)[-context_window_characters:]
                    context_window = ' '.join(context_window.split(' ')[-context_window_words:]).strip()

                yield soft_segment, tgt_segment
