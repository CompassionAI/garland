# pylint: disable=no-member
import copy
from typing import Optional
from enum import Enum

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BartConfig, BartPretrainedModel, BartForCausalLM
from transformers.activations import GELUActivation
from transformers.models.bart.modeling_bart import (
    BartDecoder,
    CausalLMOutputWithCrossAttentions,
    BartEncoderLayer,
    _expand_mask
)
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


class ContextArchitecture(Enum):
    DenseFeatureTransformer = 1
    BartEncoderLayerOnTop = 2
    FullBartEncoder = 3
    BartEncoderFirstLayerOnly = 4
    FrozenEmbeddingsWithTwoLayers = 5


class BartDecoderWithPooledContext(BartDecoder):
    """A BART decoder with a tweaked token embedding layer that injects a learned layer to pool encoded target language
    context into the token embedding slot for the starting token.
    """

    context_architecture = ContextArchitecture.BartEncoderLayerOnTop 

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens=embed_tokens)

        if BartDecoderWithPooledContext.context_architecture == ContextArchitecture.DenseFeatureTransformer:
            self.context_fc = nn.Linear(in_features=config.d_model, out_features=config.d_model)
            self.context_activation_fn = GELUActivation()
        elif BartDecoderWithPooledContext.context_architecture == ContextArchitecture.BartEncoderLayerOnTop:
            self.context_layer = BartEncoderLayer(config)
        elif BartDecoderWithPooledContext.context_architecture == ContextArchitecture.FullBartEncoder:
            model = AutoModel.from_pretrained("facebook/bart-base")
            self.context_encoder = model.encoder
        elif BartDecoderWithPooledContext.context_architecture == ContextArchitecture.BartEncoderFirstLayerOnly:
            model = AutoModel.from_pretrained("facebook/bart-base")
            model.encoder.layers = model.encoder.layers[:1]
            self.context_encoder = model.encoder
        elif BartDecoderWithPooledContext.context_architecture == ContextArchitecture.FrozenEmbeddingsWithTwoLayers:
            model = AutoModel.from_pretrained("facebook/bart-base")
            model.encoder.layers = model.encoder.layers[:2]
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = False
            for p in model.encoder.embed_positions.parameters():
                p.requires_grad = False
            self.context_encoder = model.encoder
        else:
            raise ValueError("Unknown context architecture")

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
        context_embedding_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if inputs_embeds is not None:
            raise ValueError("Don't specify inputs_embeds to a decoder with context!")
        input = input_ids
        input_shape = input.shape
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input) * self.embed_scale

        if context_embedding is not None:
            if context_embedding_mask is None:
                raise ValueError("If passing in a context embedding, must also pass in a context attention mask")
            if BartDecoderWithPooledContext.context_architecture == ContextArchitecture.DenseFeatureTransformer:
                if not len(context_embedding.shape) == 3:
                    raise ValueError("Context embedding should have 3 dimensions. Are you sure you're not feding a raw "
                                     "context dataset?")
                context_embedding_mask = (
                    context_embedding_mask.unsqueeze(-1).expand(-1, -1, self.config.d_model)
                )
                features = self.context_fc(context_embedding_mask * context_embedding)
                features = self.context_activation_fn(features)
                features = features.sum(axis=1)
            elif BartDecoderWithPooledContext.context_architecture == ContextArchitecture.BartEncoderLayerOnTop:
                if not len(context_embedding.shape) == 3:
                    raise ValueError("Context embedding should have 3 dimensions. Are you sure you're not feding a raw "
                                     "context dataset?")
                context_embedding_mask = _expand_mask(context_embedding_mask, inputs_embeds.dtype)
                features = self.context_layer(
                    context_embedding,
                    context_embedding_mask,
                    layer_head_mask=None,
                    output_attentions=False,
                )[0]
                features = features[:,0,:]
            elif BartDecoderWithPooledContext.context_architecture == ContextArchitecture.BartEncoderLayerOnTop or \
                 BartDecoderWithPooledContext.context_architecture == ContextArchitecture.FullBartEncoder or \
                 BartDecoderWithPooledContext.context_architecture == ContextArchitecture.BartEncoderFirstLayerOnly or \
                 BartDecoderWithPooledContext.context_architecture == ContextArchitecture.FrozenEmbeddingsWithTwoLayers:
                features = self.context_encoder(
                    input_ids=context_embedding, attention_mask=context_embedding_mask).last_hidden_state[:,0,:]
            else:
                raise ValueError("Unknown context architecture")

            features = features.unsqueeze(1)

            if inputs_embeds.shape[1] > 1:
                inputs_embeds = torch.cat([inputs_embeds[:,0:1,:], features, inputs_embeds[:,2:,:]], dim=1)

        return super().forward(
            None,
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
            **kwargs
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
        context_embedding_mask=None,
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
            context_embedding_mask=context_embedding_mask,
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
AutoModelForCausalLM.register(BartWithPooledContextConfig, BartWithPooledContextForCausalLM)
