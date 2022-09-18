# pylint: disable=no-member
import copy
from typing import Optional, Union
from enum import Enum

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    M2M100Config,
    M2M100PreTrainedModel,
    M2M100ForConditionalGeneration,
    M2M100Model
)
from transformers.activations import GELUActivation
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, M2M100Decoder
from transformers.models.bart.modeling_bart import (
    BartEncoderLayer,
    BartForCausalLM,
    BartPretrainedModel,
    CausalLMOutputWithCrossAttentions,
    _expand_mask
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class M2MWithPooledContextForCausalLMConfig(M2M100Config):
    """Configuration class to store the composed configuration of a BART decoder with pooled context."""
    model_type = "pooled-context-bart"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remapped_tokens = kwargs.get("remapped_tokens", False)
        self.with_pooled_context = kwargs.get("with_pooled_context", True)
        self.context_architecture = kwargs.get("context_architecture", "no-context-injection")
        self.bos_token_id = kwargs.get("bos_token_id", self.eos_token_id)
        self.decoder_start_token_id = kwargs.get("decoder_start_token_id", self.eos_token_id)


# This injects our new model config type into the Hugging Face factory for configs without having to modify their
#   code base. If ever this is contributed to the Transformers main branch, this should be moved.
AutoConfig.register(M2MWithPooledContextForCausalLMConfig.model_type, M2MWithPooledContextForCausalLMConfig)


class ContextArchitecture(Enum):
    NoContextInjection = "no-context-injection"
    DenseFeatureTransformer = "dense-feature-transformer"
    BartEncoderLayerOnTop = "bart-encoder-layer-on-top"
    FullBartEncoder = "full-bart-encoder"
    BartEncoderFirstLayerOnly = "bart-encoder-first-layer-only"
    FrozenEmbeddingsWithTwoLayers = "frozen-embeddings-with-two-layers"


class M2MRemappedEncoderConfig(M2M100Config):
    model_type = "m2m-remapped-encoder"


AutoConfig.register(M2MRemappedEncoderConfig.model_type, M2MRemappedEncoderConfig)


class M2MRemappedEncoder(M2M100Encoder):
    """An M2M encoder that allows for remapped token embeddings to specialize to a specific language."""

    config_class = M2MRemappedEncoderConfig

    def __init__(
        self,
        config: Union[M2M100Config, M2MRemappedEncoderConfig],
        embed_tokens: Optional[nn.Embedding] = None
    ):
        if type(config) is M2M100Config:
            config = M2MRemappedEncoderConfig.from_dict(config.to_dict())
        super().__init__(config, embed_tokens)

    def remap_tokens(self, tokenizer):
        new_weights = self.embed_tokens.weight[tokenizer.used_tokens, :]
        new_embed_tokens = torch.nn.Embedding(
            new_weights.shape[0], new_weights.shape[1], padding_idx=self.embed_tokens.padding_idx)
        new_embed_tokens.weight = torch.nn.Parameter(new_weights)
        self.embed_tokens = new_embed_tokens

        self.config.remapped_tokens = True
        self.config.vocab_size = len(tokenizer.used_tokens)


AutoModel.register(M2MRemappedEncoderConfig, M2MRemappedEncoder)


class M2M100ModelWithRemappedEncoder(M2M100Model):
    """Vanilla M2M pretrained model but with the CAI encoder"""
    def __init__(self, config: M2M100Config):
        M2M100PreTrainedModel.__init__(self, config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = M2MRemappedEncoder(config, self.shared)
        self.decoder = M2M100Decoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()


class M2MForCGWithRemappedEncoder(M2M100ForConditionalGeneration):
    """Vanilla M2M encoder-decoder stack but with the CAI encoder"""
    def __init__(self, config: M2M100Config):
        M2M100PreTrainedModel.__init__(self, config)
        self.model = M2M100ModelWithRemappedEncoder(config)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class M2MDecoderWithPooledContext(M2M100Decoder):
    """An M2M decoder with a tweaked token embedding layer that injects a learned layer to pool encoded target language
    context into the token embedding slot for the starting token.
    """

    context_architecture = ContextArchitecture.BartEncoderLayerOnTop 

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens=embed_tokens)

        self.context_architecture = ContextArchitecture(config.context_architecture)

        if self.context_architecture == ContextArchitecture.NoContextInjection:
            return
        elif self.context_architecture == ContextArchitecture.DenseFeatureTransformer:
            self.context_fc = nn.Linear(in_features=config.d_model, out_features=config.d_model)
            self.context_activation_fn = GELUActivation()
        elif self.context_architecture == ContextArchitecture.BartEncoderLayerOnTop:
            self.context_layer = BartEncoderLayer(AutoConfig.from_pretrained("facebook/bart-base"))
        elif self.context_architecture == ContextArchitecture.FullBartEncoder:
            model = AutoModel.from_pretrained("facebook/bart-base")
            self.context_encoder = model.encoder
        elif self.context_architecture == ContextArchitecture.BartEncoderFirstLayerOnly:
            model = AutoModel.from_pretrained("facebook/bart-base")
            model.encoder.layers = model.encoder.layers[:1]
            self.context_encoder = model.encoder
        elif self.context_architecture == ContextArchitecture.FrozenEmbeddingsWithTwoLayers:
            model = AutoModel.from_pretrained("facebook/bart-base")
            model.encoder.layers = model.encoder.layers[:2]
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = False
            for p in model.encoder.embed_positions.parameters():
                p.requires_grad = False
            self.context_encoder = model.encoder
        else:
            raise ValueError("Unknown context architecture")
        self.adapter_layer = torch.nn.Linear(768, 1024)

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

        if not self.context_architecture == ContextArchitecture.NoContextInjection and context_embedding is not None:
            if context_embedding_mask is None:
                raise ValueError("If passing in a context embedding, must also pass in a context attention mask")
            if self.context_architecture == ContextArchitecture.DenseFeatureTransformer:
                if not len(context_embedding.shape) == 3:
                    raise ValueError("Context embedding should have 3 dimensions. Are you sure you're not feding a raw "
                                     "context dataset?")
                context_embedding_mask = (
                    context_embedding_mask.unsqueeze(-1).expand(-1, -1, self.config.d_model)
                )
                features = self.context_fc(context_embedding_mask * context_embedding)
                features = self.context_activation_fn(features)
                features = features.sum(axis=1)
            elif self.context_architecture == ContextArchitecture.BartEncoderLayerOnTop:
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
                if features.shape[1] > 0:
                    features = features[:,0,:]
                else:
                    features = None
            elif self.context_architecture == ContextArchitecture.FullBartEncoder or \
                 self.context_architecture == ContextArchitecture.BartEncoderFirstLayerOnly or \
                 self.context_architecture == ContextArchitecture.FrozenEmbeddingsWithTwoLayers:
                features = self.context_encoder(
                    input_ids=context_embedding, attention_mask=context_embedding_mask).last_hidden_state[:,0,:]
            else:
                raise ValueError("Unknown context architecture")
            if features is not None:
                features = self.adapter_layer(features)

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


class M2MDecoderWithPooledContextWrapper(M2M100PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = M2MDecoderWithPooledContext(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class M2MWithPooledContextForCausalLM(BartForCausalLM):
    """A wrapper for the pooled context decoder for causal LM."""

    config_class = M2MWithPooledContextForCausalLMConfig

    def __init__(self, config: M2MWithPooledContextForCausalLMConfig):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        BartPretrainedModel.__init__(self, config)
        self.model = M2MDecoderWithPooledContextWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.remapped_tokens:
            self.resize_token_embeddings(self.config.vocab_size)

    def remap_tokens(self, tokenizer):
        new_weights = self.model.decoder.embed_tokens.weight[tokenizer.used_tokens, :]
        new_embed_tokens = torch.nn.Embedding(
            new_weights.shape[0], new_weights.shape[1], padding_idx=self.model.decoder.embed_tokens.padding_idx)
        new_embed_tokens.weight = torch.nn.Parameter(new_weights)
        self.model.decoder.embed_tokens = new_embed_tokens

        new_weights = self.lm_head.weight[tokenizer.used_tokens, :]
        new_lm_head = torch.nn.Linear(new_weights.shape[0], new_weights.shape[1], bias=False)
        new_lm_head.weight = torch.nn.Parameter(new_weights)
        self.lm_head = new_lm_head

        self.config.remapped_tokens = True
        self.config.vocab_size = len(tokenizer.used_tokens)

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
AutoModelForCausalLM.register(M2MWithPooledContextForCausalLMConfig, M2MWithPooledContextForCausalLM)
