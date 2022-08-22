from transformers import AutoModel, EncoderDecoderModel, EncoderDecoderConfig


class CAIEncoderDecoderConfig(EncoderDecoderConfig):
    pass


class CAIEncoderDecoderModel(EncoderDecoderModel):
    """A wrapper class for the encoder-decoder model to include various CompassionAI features, such as context
        embeddings in the forward."""

    config_class = CAIEncoderDecoderConfig
    base_model_prefix = "encoder_decoder_with_context"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        context_embedding=None,
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


# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModel.register(CAIEncoderDecoderConfig, CAIEncoderDecoderModel)
