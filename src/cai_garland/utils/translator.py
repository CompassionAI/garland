# pylint: disable=dangerous-default-value
# pylint: disable=no-member
# pylint: disable=protected-access
import logging
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, LogitsProcessor, LogitsProcessorList

from cai_common.models.utils import get_local_ckpt, get_cai_config
from cai_garland.models.factory import make_bilingual_tokenizer
from cai_garland.utils.segmenters import SegmenterNone, SegmenterOpeningShad
from cai_garland.models.siamese_encoder import SiameseEncoderModel, BaseModelOutputWithAttentionMask
from cai_garland.data.knowledge_injection_dataset import KnowledgeInjectionDataset
from cai_garland.models.cai_encoder_decoder import CAIEncoderDecoderModel

try:
    import deepspeed
    _has_deepspeed = True
except:
    _has_deepspeed = False


logger = logging.getLogger(__name__)


class TokenizationTooLongException(Exception):
    pass


@dataclass
class RerankedSmoothedBeamSearchSettings:
    num_beams: int
    num_return_sequences: int
    smoothing_indices: List[float]
    reranking_model: str


class LabelSmoothingLogitsProcessor(LogitsProcessor):
    source_tokens = None

    def __init__(self, smoothing_index):
        super().__init__()
        self.smoothing_index = smoothing_index
    
    def __call__(self, input_ids, scores):
        if self.smoothing_index == 0:
            return scores
        clamp_idx = int(self.smoothing_index)
        clamp_val = torch.topk(scores.view(-1), clamp_idx).values[-1]
        return torch.clamp(scores, max=clamp_val)


@contextmanager
def _reranker_ctx(reranker, ctx_embedding, ctx_mask):
    if reranker is not None:
        with reranker.model.prepare_model_for_generation(ctx_embedding, ctx_mask) as ctx:
            yield ctx
    else:
        yield None


def _warm_start_constraints(_batch_id, input_ids, target_tkns):
    if len(input_ids) < len(target_tkns):
        return [target_tkns[len(input_ids) - 1]]
    else:
        return slice(0, None)


def _excluded_prefix_constraints(batch_id, input_ids, prefix_excluded, generated_prefix, vocab_size):
    if len(input_ids) < 4:
        return generated_prefix[len(input_ids)]
    if len(input_ids) < len(prefix_excluded) + 4:
        return [t for t in range(vocab_size) if not t == prefix_excluded[len(input_ids) - 4]]
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
        method: Generation method to use. Possible values are beam-search and reranked-smoothed-beam-search.
        method_settings: Any needed settings for the generation method.
        add_score: Append a score to the translation with a pipe.
    """

    hard_segmenter = SegmenterOpeningShad()
    preprocessors = []
    soft_segmenter = SegmenterNone()
    postprocessors = []
    method = "beam-search"
    method_settings: Optional[RerankedSmoothedBeamSearchSettings] = None
    add_score = False

    def __init__(self, model_ckpt: str, deepspeed_cfg: str = None) -> None:
        """Loads all the relevant data and models for machine translation.

        Args:
            model_ckpt: Name of the fine-tuned model checkpoint in the data registry to use for translation. For
                example, olive-cormorant-bart.
            deepspeed_cfg: Name of the JSON file configuring DeepSpeed for inference, if any."""
        is_deepspeed = deepspeed_cfg is not None
        if is_deepspeed and not _has_deepspeed:
            raise ImportError("Requires DeepSpeed but DeepSpeed import failed, likely not installed!")

        local_ckpt = get_local_ckpt(model_ckpt, model_dir=True)
        logger.info(f"Local model checkpoint {model_ckpt} resolved to {local_ckpt}")

        self.model = CAIEncoderDecoderModel.from_pretrained(local_ckpt)
        if is_deepspeed:
            self.ds_engine = deepspeed.init_inference(model=self.model, config=deepspeed_cfg)
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
        self.tokenizer = make_bilingual_tokenizer(encoder_name, decoder_name, is_deepspeed=is_deepspeed)

        if cai_base_config.get("reset_token_vocab", False) and not is_deepspeed:
            logger.info("Resetting token vocabulary")
            from transformers import M2M100ForConditionalGeneration
            nllb_model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M")
            self.model.decoder.model.decoder.embed_tokens = nllb_model.model.shared
            self.model.decoder.lm_head = nllb_model.lm_head
            self.tokenizer.remap_target = False

        logger.info("Configuring model")
        self.model.eval()

        self.num_beams = 20
        self.device = torch.device("cpu")
        self.context_encoder = None
        self.reranker = None
        self.decoding_length = self.model.decoder.max_length
        self._bad_words, self._bad_word_tokens = [], []

        logger.info("Loading glossary")
        self.glossary = KnowledgeInjectionDataset(None)
        self.glossary.inject_glossary(
            "processed_datasets/tibetan-sanskrit-glossary",
            source_encoder_name="cai:albert-olive-cormorant/base",
            target_decoder_name="hf:facebook/bart-base",
        )

    def to(self, device):
        self.device = device
        self.model.to(device)
        if self.context_encoder is not None:
            self.context_encoder.to(device)
        if self.reranker is not None:
            self.reranker.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def mps(self) -> None:
        self.to(torch.device("mps"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

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
        self.context_encoder.to(self.device)

    def load_reranker(self, reranker_model):
        """Load a reranking model for reranked smoothed beam search.
        
        Args:
            model_name: A CAI model name for the reranking model. Should be a seq2seq model, the loss is used as the
                ranking score.
        """
        logger.info(f"Loading reranking model: {self.method_settings.reranking_model}")
        self.reranker = Translator(self.method_settings.reranking_model)
        self.reranker.to(self.device)

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

        bo_tokens = [bo_tokens.to(self.device)]
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
        last_hidden_state = encoder_outputs.last_hidden_state.to(self.device).detach().numpy()
        attention_mask = encoder_outputs.attention_mask.to(self.device).detach().numpy()
        torch.cuda.empty_cache()
        return {
            "last_hidden_state": last_hidden_state,
            "attention_mask": attention_mask
        }

    def _generate_bs(self, source_inputs, source_text, generator_kwargs={}, **kwargs):
        kwargs |= generator_kwargs
        kwargs['glossary'] = self.glossary.get_glossary(source_text)
        gen_res = self.model.generate(
            source_inputs.input_ids, return_dict_in_generate=True, output_scores=True, **kwargs
        )
        return gen_res.sequences.cpu().tolist(), gen_res.sequences_scores.cpu().tolist()

    def _compute_rank_score(self, source_inputs, target_text, translator):
        with translator.tokenizer.as_target_tokenizer():
            target_tokens = translator.tokenizer(target_text, return_tensors="pt")
        model_inputs = {
            'input_ids': source_inputs.input_ids.to(translator.model.device),
            'attention_mask': source_inputs.attention_mask.to(translator.model.device),
            'labels': target_tokens['input_ids'].to(translator.model.device),
        }
        return -float(translator.model(**model_inputs).loss.to("cpu"))

    def _rerank(self, source, candidates, candidate_tokens):
        # The generator and reranker tokenizers may be different, so need to be careful which one to use
        losses = [
            (tokens, self._compute_rank_score(source, candidate, self.reranker))
            for candidate, tokens in tqdm(
                zip(candidates, candidate_tokens), desc="Reranking scores", leave=False, total=len(candidates)
            )
        ]
        losses = sorted(losses, key = lambda x: -x[1])
        return list(zip(*losses))

    def _generate_rs_bs(self, source_inputs, source_text, generator_kwargs={}, **kwargs):
        if not isinstance(self.method_settings, RerankedSmoothedBeamSearchSettings):
            raise ValueError("Set method_settings to a filled out RerankedSmoothedBeamSearchSettings.")

        del kwargs['num_beams']

        candidates, tokens = [], []
        for smoothing_index in tqdm(self.method_settings.smoothing_indices, desc="Smoothing index", leave=False):
            preds, _ = self._generate_bs(
                source_inputs,
                source_text,
                generator_kwargs=generator_kwargs,
                logits_processor=LogitsProcessorList([LabelSmoothingLogitsProcessor(smoothing_index)]),
                num_beams=self.method_settings.num_beams,
                num_return_sequences=self.method_settings.num_beams,
                **kwargs
            )
            tokens.extend(preds)
            with self.tokenizer.as_target_tokenizer():
                for pred in preds:
                    candidates.append(self.tokenizer.decode(pred, skip_special_tokens=True).strip())

        candidates, scores  = self._rerank(source_inputs, candidates, tokens)
        return candidates[:self.method_settings.num_return_sequences], \
            scores[:self.method_settings.num_return_sequences]

    def _generate(self, source_inputs, source_text, generator_kwargs={}, **kwargs):
        if self.method == 'beam-search':
            return self._generate_bs(source_inputs, source_text, generator_kwargs=generator_kwargs, **kwargs)
        elif self.method == 'reranked-smoothed-beam-search':
            return self._generate_rs_bs(source_inputs, source_text, generator_kwargs=generator_kwargs, **kwargs)
        else:
            raise NotImplementedError(f"Unknown generation method {self.method}")

    def translate(
        self,
        bo_text: str,
        prefix: Optional[str]=None,
        context: Optional[str]=None,
        target_language_code: Optional[str]=None,
        encoder_outputs: Any = None,
        generator_kwargs: Dict[Any, Any]={},
        prefix_excluded: Optional[List[int]]=None,
        return_full_results: bool = False
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
            prefix_excluded (optional): Force the generated text to not start with this prefix.
            return_full_results (optional): Return HuggingFace results object.

        Returns:
            The translated text (not tokens)."""

        if context is not None and self.context_encoder is None:
            raise ValueError("Must specify a context encoder if translating with context. Call prepare_context_encoder "
                             "to do this.")

        bo_inputs = self.tokenizer(bo_text, return_tensors="pt")
        bo_tokens = bo_inputs.input_ids
        if context is not None:
            ctx_tokens = self.context_tokenizer(context, return_tensors="pt")

        if prefix_excluded is not None:
            with self.tokenizer.as_target_tokenizer():
                prefix_excluded = self.tokenizer.encode(prefix_excluded, add_special_tokens=False)

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
            # We embed the context here, instead of passing the raw tokens. Otherwise every time the beam search (or
            #   whatever generation method) calls the model forward, it will encode the context again. The model will
            #   slice and reshape it as needed.
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
        bo_inputs = bo_inputs.to(self.device)
        if encoder_outputs is not None:
            if isinstance(encoder_outputs.last_hidden_state, list):
                encoder_outputs.last_hidden_state = [t.to(self.device) for t in encoder_outputs.last_hidden_state]
                encoder_outputs.attention_mask = [t.to(self.device) for t in encoder_outputs.attention_mask]
            else:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.to(self.device)
                encoder_outputs.attention_mask = encoder_outputs.attention_mask.to(self.device)
        if encoder_outputs is not None:
            generator_kwargs['encoder_outputs'] = encoder_outputs
            generator_kwargs['attention_mask'] = encoder_outputs.attention_mask
        with (
            self.model.prepare_model_for_generation(ctx_embedding, ctx_mask),
            _reranker_ctx(self.reranker, ctx_embedding, ctx_mask)
        ):
            if target_language_code is None:
                language_token = self.model.forced_bos_token_id(self.tokenizer)
            else:
                language_token = self.tokenizer.target_tokenizer.lang_code_to_id[target_language_code]
            if prefix_excluded is not None:
                if prefix_fn is not None:
                    raise ValueError("Cannot provide excluded prefix with prefix-conditioned generation")
                prefix_fn = lambda batch_id, input_ids: _excluded_prefix_constraints(
                    batch_id,
                    input_ids,
                    prefix_excluded,
                    self.model.decoder.generated_prefix(self.tokenizer.target_tokenizer),
                    self.tokenizer.target_tokenizer.vocab_size
                )
            preds, scores = self._generate(
                bo_inputs,
                bo_text,
                max_length=self.decoding_length,
                # forced_bos_token_id=language_token,
                num_beams=self.num_beams,
                prefix_allowed_tokens_fn=prefix_fn,
                bad_words_ids=None if len(self._bad_word_tokens) == 0 else self._bad_word_tokens,   # Needs to be None instead of empty, otherwise HF throws ValueError
                generator_kwargs=generator_kwargs
            )

        logger.debug(f"Generated tokens: {preds}")
        logger.debug(f"Generated main hypothesis length: {len(preds[0])}")
        translations = []
        with self.tokenizer.as_target_tokenizer():
            for t, pred in enumerate(preds):
                translation = self.tokenizer.decode(pred, skip_special_tokens=True).strip()
                if self.add_score:
                    translation += "|" + str(scores[t])
                translations.append(translation)
        if len(translations) == 1:
            translations = translations[0]
        if return_full_results:
            raise NotImplementedError()
            # return translations, gen_res
        return translations

    def segment(
        self,
        bo_text,
        tqdm=tqdm,      # pylint: disable=redefined-outer-name
        hard_segmenter_kwargs={},
        soft_segmenter_kwargs={}
    ):
        hard_segments = self.hard_segmenter(bo_text, translator=self, **hard_segmenter_kwargs)

        for hard_seg_count, hard_segment in tqdm(
            enumerate(hard_segments), total=len(hard_segments), desc="Hard segments", leave=False
        ):
            for preproc_func in self.preprocessors:
                hard_segment = preproc_func(hard_segment)

            self.soft_segmenter.hard_segment_counter = hard_seg_count
            soft_segments = self.soft_segmenter(hard_segment, translator=self, tqdm=tqdm, **soft_segmenter_kwargs)

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

            for soft_segment in soft_segments:
                yield soft_segment

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
        previous_results=[],
        generator_kwargs={}
    ):
        if retrospective_decoding and contextual_decoding:
            raise ValueError("Specify either retrospective or contextual decoding, not both.")

        hard_segments = self.hard_segmenter(bo_text, translator=self, **hard_segmenter_kwargs)

        for hard_seg_count, hard_segment in tqdm(
            enumerate(hard_segments), total=len(hard_segments), desc="Hard segments", leave=False
        ):
            for preproc_func in self.preprocessors:
                hard_segment = preproc_func(hard_segment)

            self.soft_segmenter.hard_segment_counter = hard_seg_count
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

            with tqdm(
                total=len(soft_segments),
                desc="Translating soft segments",
                initial=len(previous_results),
                leave=False
            ) as pbar:
                for seg_idx, soft_segment in enumerate(soft_segments):
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
                    if seg_idx < len(previous_results):
                        src_prev_res, tgt_segment = previous_results[seg_idx]
                        if not src_prev_res == input_:
                            raise ValueError("Previous results appear to be corrupt, segments are not matching. "
                                            f"Previous results give {src_prev_res}, but segmentation gives {input_}.")
                    else:
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
                        pbar.update()

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
