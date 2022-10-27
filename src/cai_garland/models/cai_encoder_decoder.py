import torch

from contextlib import contextmanager

from transformers import AutoModel, EncoderDecoderModel, EncoderDecoderConfig
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList


class CAIEncoderDecoderConfig(EncoderDecoderConfig):
    start_token_repetitions = 2
    forced_bos_language_code = "eng_Latn"


class CAIEncoderDecoderModel(EncoderDecoderModel):
    """A wrapper class for the encoder-decoder model to include various CompassionAI features, such as context
        embeddings in the forward."""

    config_class = CAIEncoderDecoderConfig
    base_model_prefix = "encoder_decoder_with_context"

    force_preparing_model_for_generation = False

    def __init__(
        self,
        config=None,
        encoder=None,
        decoder=None
    ):
        super().__init__(config, encoder, decoder)
        self.cur_context_embedding = None
        self.cur_context_embedding_mask = None
        if hasattr(config.decoder, "bos_token_id"):
            self.config.bos_token_id = config.decoder.bos_token_id
        if hasattr(config.decoder, "decoder_start_token_id"):
            self.config.decoder_start_token_id = config.decoder.decoder_start_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        context_embedding=None,
        context_embedding_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if context_embedding is None:
            context_embedding = self.cur_context_embedding
            context_embedding_mask = self.cur_context_embedding_mask
        if context_embedding is not None:
            kwargs["decoder_context_embedding"] = context_embedding
            kwargs["decoder_context_embedding_mask"] = context_embedding_mask
        return super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs
        )

    def forced_bos_token_id(self, tokenizer):
        if self.config.forced_bos_language_code is not None:
            tokenizer.target_tokenizer.apply_token_remapping = tokenizer.remap_target
            return tokenizer.target_tokenizer.language_id(self.config.forced_bos_language_code)
        return None

    @contextmanager
    def prepare_model_for_generation(self, context_embedding, context_embedding_mask):
        if context_embedding is not None:
            self.cur_context_embedding = context_embedding.to(self.device)
        if context_embedding_mask is not None:
            self.cur_context_embedding_mask = context_embedding_mask.to(self.device)
        yield
        self.cur_context_embedding = None
        self.cur_context_embedding_mask = None

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        decoder_start_token_id=None,
        bos_token_id=None,
        model_kwargs=None,
        device=None,
    ):
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            if device is None:
                device = self.device
            return torch.ones(
                (batch_size, self.config.start_token_repetitions),
                dtype=torch.long,
                device=device
            ) * decoder_start_token_id

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        typical_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        force_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        num_return_sequences=None,
        max_time=None,
        max_new_tokens=None,
        decoder_start_token_id=None,
        use_cache=None,
        num_beam_groups=None,
        diversity_penalty=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
        renormalize_logits=None,
        stopping_criteria=StoppingCriteriaList(),
        constraints=None,
        output_attentions=None,
        output_hidden_states=None,
        output_scores=None,
        return_dict_in_generate=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        remove_invalid_values=None,
        synced_gpus=False,
        exponential_decay_length_penalty=None,
        **model_kwargs,
    ):
        if self.cur_context_embedding is None:
            if self.force_preparing_model_for_generation:
                raise ValueError("Specify context embeddings for generation using prepare_model_for_generation")
        else:
            num_beams = num_beams if num_beams is not None else self.config.num_beams
            self.cur_context_embedding = self.cur_context_embedding.repeat_interleave(num_beams, dim=0)
            self.cur_context_embedding_mask = self.cur_context_embedding_mask.repeat_interleave(
                num_beams, dim=0
            )
        return super().generate(
            inputs,
            max_length,
            min_length,
            do_sample,
            early_stopping,
            num_beams,
            temperature,
            top_k,
            top_p,
            typical_p,
            repetition_penalty,
            bad_words_ids,
            force_words_ids,
            bos_token_id,
            pad_token_id,
            eos_token_id,
            length_penalty,
            no_repeat_ngram_size,
            encoder_no_repeat_ngram_size,
            num_return_sequences,
            max_time,
            max_new_tokens,
            decoder_start_token_id,
            use_cache,
            num_beam_groups,
            diversity_penalty,
            prefix_allowed_tokens_fn,
            logits_processor,
            renormalize_logits,
            stopping_criteria,
            constraints,
            output_attentions,
            output_hidden_states,
            output_scores,
            return_dict_in_generate,
            forced_bos_token_id,
            forced_eos_token_id,
            remove_invalid_values,
            synced_gpus,
            exponential_decay_length_penalty,
            **model_kwargs
        )


# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModel.register(CAIEncoderDecoderConfig, CAIEncoderDecoderModel)
