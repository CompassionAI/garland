import torch

from contextlib import contextmanager

from transformers import (
    AutoModel, EncoderDecoderModel, EncoderDecoderConfig, LogitsProcessorList, StoppingCriteriaList, GenerationConfig)


class CAIEncoderDecoderConfig(EncoderDecoderConfig):
    start_token_repetitions = 2
    forced_bos_language_code = "eng_Latn"

    @property
    def hidden_size(self):
        return max(self.encoder.hidden_size, self.decoder.hidden_size)



class CAIEncoderDecoderModel(EncoderDecoderModel):
    """A wrapper class for the encoder-decoder model to include various CompassionAI features, such as context
        embeddings in the forward."""

    config_class = CAIEncoderDecoderConfig
    base_model_prefix = "encoder_decoder_with_context"

    force_preparing_model_for_generation = False
    label_smoothing_factor = 0
    fc_layer_reg_lambda = 0

    def __init__(
        self,
        config=None,
        encoder=None,
        decoder=None
    ):
        super().__init__(config, encoder, decoder)
        self.context_embedding = None
        self.context_embedding_mask = None
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
        glossary=None,
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
            context_embedding = getattr(self, "cur_context_embedding", self.context_embedding)
            context_embedding_mask = getattr(self, "cur_context_embedding_mask", self.context_embedding_mask)
        if context_embedding is not None:
            kwargs["decoder_context_embedding"] = context_embedding
            kwargs["decoder_context_embedding_mask"] = context_embedding_mask
        if glossary is not None:
            kwargs["decoder_glossary_source"] = {
                'embeddings': self.encoder.get_input_embeddings()(glossary['source']['input_ids']),
                'attention_mask': glossary['source']['attention_mask']
            }
            kwargs["decoder_glossary_target"] = {
                'embeddings': self.decoder.get_input_embeddings()(glossary['target']['input_ids']),
                'attention_mask': glossary['target']['attention_mask']
            }

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.fc_layer_reg_lambda == 0:
            if not return_dict:
                raise ValueError("Must have return_dict=True in forward for activation regularization.")
            output_hidden_states = True
        res = super().forward(
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
        if self.training and labels is not None:
            update_loss = False
            if not self.label_smoothing_factor == 0:
                logits = res.logits if return_dict else res[1]
                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing_factor)
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
                update_loss = True
            if not self.fc_layer_reg_lambda == 0:
                fc_layers_l1_norm = torch.norm(torch.cat([l.view(-1) for l in res.decoder_hidden_states]))
                loss = res.loss + self.fc_layer_reg_lambda * fc_layers_l1_norm
                update_loss = True
            if update_loss:
                if not return_dict:
                    res[0] = loss
                else:
                    res.loss = loss
        return res

    def forced_bos_token_id(self, tokenizer):
        if self.config.forced_bos_language_code is not None:
            tokenizer.target_tokenizer.apply_token_remapping = tokenizer.remap_target
            return tokenizer.target_tokenizer.language_id(self.config.forced_bos_language_code)
        return None

    @contextmanager
    def prepare_model_for_generation(self, context_embedding, context_embedding_mask):
        cur_dtype = next(self.parameters()).dtype
        if context_embedding is not None:
            self.context_embedding = context_embedding.to(device=self.device, dtype=cur_dtype)
        if context_embedding_mask is not None:
            self.context_embedding_mask = context_embedding_mask.to(device=self.device, dtype=cur_dtype)
        yield
        delattr(self, "context_embedding")
        delattr(self, "context_embedding_mask")

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_input_name,
        model_kwargs,
        decoder_start_token_id,
        bos_token_id,
        device
    ):
        if not model_input_name == 'input_ids':
            raise ValueError(
                f"CAIEncoderDecoderModel can only handle model_input_ids=='input_ids' but given {model_input_name}")
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            if device is None:
                device = self.device
            decoder_input_ids = torch.ones(
                (batch_size, self.config.start_token_repetitions),
                dtype=torch.long,
                device=device
            ) * decoder_start_token_id
        model_kwargs['decoder_input_ids'] = decoder_input_ids
        return super()._prepare_decoder_input_ids_for_generation(
            batch_size,
            model_input_name,
            model_kwargs,
            decoder_start_token_id,
            bos_token_id,
            device
        )

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
        stopping_criteria=StoppingCriteriaList(),
        synced_gpus=False,
        num_beams=None,
        **model_kwargs,
    ):
        if self.context_embedding is None:
            if self.force_preparing_model_for_generation:
                raise ValueError("Specify context embeddings for generation using prepare_model_for_generation")
        else:
            num_beams = num_beams if num_beams is not None else self.config.num_beams
            self.cur_context_embedding = self.context_embedding.repeat_interleave(num_beams, dim=0)
            self.cur_context_embedding_mask = self.context_embedding_mask.repeat_interleave(
                num_beams, dim=0
            )
        return super().generate(
            inputs=inputs,
            num_beams=num_beams,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            **model_kwargs
        )


# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModel.register(CAIEncoderDecoderConfig, CAIEncoderDecoderModel)
