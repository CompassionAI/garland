# pylint: disable=no-member
import copy
from typing import Optional

from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModel, BartConfig, BartPretrainedModel, BartForCausalLM
from transformers.models.bart.modeling_bart import BartDecoder, CausalLMOutputWithCrossAttentions
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BartWithPooledContextConfig(BartConfig):
    """Configuration class to store the composed configuration of a BART decoder with pooled context."""
    model_type = "pooled-context-bart"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_pooled_context = True


# This injects our new model config type into the Hugging Face factory for configs without having to modify their
#   code base. If ever this is contributed to the Transformers main branch, this should be moved.
AutoConfig.register(BartWithPooledContextConfig.model_type, BartWithPooledContextConfig)


class BartDecoderWithPooledContext(BartDecoder):
    """A BART decoder with a tweaked token embedding layer that injects a learned layer to pool encoded target language
    context into the token embedding slot for the starting token.
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # EUG!!!
        if embed_tokens is None:
            embed_tokens = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        # embed_tokens = wrapper_with_pooled_context_injection(embed_tokens)
        super().__init__(config, embed_tokens=embed_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        context_embedding=None,
        context_embedding_attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return super().forward(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )


class BartDecoderWithPooledContextWrapper(BartPretrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = BartDecoderWithPooledContext(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class BartWithPooledContextForCausalLM(BartForCausalLM):
    """A wrapper for the pooled context decoder for causal LM."""

    config_class = BartWithPooledContextConfig

    def __init__(self, config: BartWithPooledContextConfig):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        BartPretrainedModel.__init__(self, config)
        self.model = BartDecoderWithPooledContextWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        context_embedding=None,
        context_embedding_attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Clone of the parent forward, but with context passed to the decoder"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            context_embedding=context_embedding,
            context_embedding_attention_mask=context_embedding_attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModel.register(BartWithPooledContextConfig, BartWithPooledContextForCausalLM)
