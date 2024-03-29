# pylint: disable=no-member
import copy
from typing import Any, Optional, Union
from enum import Enum

import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm

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
from .cai_encoder_decoder import CAIEncoderDecoderModel


logger = logging.get_logger(__name__)


class M2MWithPooledContextForCausalLMConfig(M2M100Config):
    """Configuration class to store the composed configuration of a BART decoder with pooled context."""
    model_type = "pooled-context-bart"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_language_code = kwargs.get('target_language_code', 'eng_Latn')
        self.remapped_tokens = kwargs.get("remapped_tokens", False)
        self.with_pooled_context = kwargs.get("with_pooled_context", True)
        self.context_architecture = kwargs.get("context_architecture", "no-context-injection")
        self.normalize_context = kwargs.get("normalize_context", False)
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
    BartEncoderTopLayerUnfrozen = "bart-encoder-top-layer-unfrozen"



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


class M2MPositionEmbeddingsFromAttentionMask(nn.Module):
    def __init__(self, embed_positions, fill_token=12345):
        super().__init__()
        self.embed_positions = embed_positions
        self.attention_mask = None
        self.fill_token = fill_token
    
    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if self.attention_mask is None:
            raise ValueError("Need to set the attention mask to make position embeddings from it!")
        if input_ids is not None:
            output_shape = input_ids.size()
        elif inputs_embeds is not None:
            output_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("One of input_ids or inputs_embeds must not be None")
        if len(output_shape) > 2:
            raise ValueError("Input tensor has too many dimensions")
        elif len(output_shape) == 2:
            if not output_shape[0] == self.attention_mask.size()[0]:
                raise ValueError("Batch sizes of inputs and attention mask do not match")
            cut_attn_mask = self.attention_mask[:,:output_shape[-1]]
        else:
            cut_attn_mask = self.attention_mask[:output_shape[-1]]
        return self.embed_positions(
            cut_attn_mask * self.fill_token + (1 - cut_attn_mask) * self.embed_positions.padding_idx,
            None,
            past_key_values_length
        )


class M2MDecoderWithPooledContext(M2M100Decoder):
    """An M2M decoder with a tweaked token embedding layer that injects a learned layer to pool encoded target language
    context into the token embedding slot for the starting token.
    """

    context_architecture = ContextArchitecture.BartEncoderLayerOnTop 
    normalize_context = False
    regularize_context = False
    regularization_sigma = 0.1

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens=embed_tokens)

        self.context_architecture = ContextArchitecture(config.context_architecture)
        self.normalize_context = config.normalize_context
        self.embed_positions = M2MPositionEmbeddingsFromAttentionMask(self.embed_positions)

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
        elif self.context_architecture == ContextArchitecture.BartEncoderTopLayerUnfrozen:
            model = AutoModel.from_pretrained("facebook/bart-base")
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = False
            for p in model.encoder.embed_positions.parameters():
                p.requires_grad = False
            for l in model.encoder.layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False
            self.context_encoder = model.encoder
        else:
            raise ValueError("Unknown context architecture")
        self.context_adapter_layer = torch.nn.Linear(768, self.embed_tokens.embedding_dim)
        self.glossary_adapter_layer = torch.nn.Linear(128, self.embed_tokens.embedding_dim)

        if self.normalize_context:
            self.normalizer = LayerNorm([768], elementwise_affine=False)    # No need for affine transform because of
                                                                            #   adapter layer.

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
        glossary_source=None,
        glossary_target=None,
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
        can_prepend_knowledge = past_key_values is None    # inputs_embeds.shape[1] > 1

        if not self.context_architecture == ContextArchitecture.NoContextInjection and context_embedding is not None:
            if context_embedding_mask is None:
                raise ValueError("If passing in a context embedding, must also pass in a context attention mask")
            if self.context_architecture == ContextArchitecture.DenseFeatureTransformer:
                if not len(context_embedding.shape) == 3:
                    raise ValueError("Context embedding should have 3 dimensions. Are you sure you're not feeding a "
                                     "raw context dataset?")
                context_embedding_mask = (
                    context_embedding_mask.unsqueeze(-1).expand(-1, -1, self.config.d_model)
                )
                features = self.context_fc(context_embedding_mask * context_embedding)
                features = self.context_activation_fn(features)
                features = features.sum(axis=1)
            elif self.context_architecture == ContextArchitecture.BartEncoderLayerOnTop:
                if not len(context_embedding.shape) == 3:
                    raise ValueError("Context embedding should have 3 dimensions. Are you sure you're not feeding a "
                                     "raw context dataset?")
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
                 self.context_architecture == ContextArchitecture.FrozenEmbeddingsWithTwoLayers or \
                 self.context_architecture == ContextArchitecture.BartEncoderTopLayerUnfrozen:
                if np.prod(context_embedding.size()) == 0:
                    context_embedding = torch.LongTensor([[1]] * context_embedding.size()[0]).to(
                        context_embedding.device)
                    context_embedding_mask = torch.LongTensor([[0]] * context_embedding.size()[0]).to(
                        context_embedding.device)
                if context_embedding.size()[-1] == self.context_encoder.embed_tokens.embedding_dim:
                    features = context_embedding[:,0,:]
                else:
                    features = self.context_encoder(
                        input_ids=context_embedding.to(torch.int), attention_mask=context_embedding_mask
                    ).last_hidden_state[:,0,:]
            else:
                raise ValueError("Unknown context architecture")
            if features is not None:
                if self.normalize_context:
                    features = self.normalizer(features)
                if self.regularize_context and self.training:
                    noise = torch.randn_like(features)
                    features = self.regularization_sigma * noise + features     # The noise is detached from the
                                                                                #   backprop by default

                features = self.context_adapter_layer(features)

                features = features.unsqueeze(1)

                if can_prepend_knowledge:
                    inputs_embeds = torch.cat([inputs_embeds[:,0:1,:], features, inputs_embeds[:,2:,:]], dim=1)

        if (glossary_source is None) != (glossary_target is None):
            raise ValueError("Both glossary inputs have to either be None or not None")
        if glossary_source is not None:
            glossary_embeds = torch.cat(
                [self.glossary_adapter_layer(glossary_source['embeddings']), glossary_target['embeddings']], dim=1
            )
            glossary_attentions = torch.cat(
                [glossary_source['attention_mask'], glossary_target['attention_mask']], dim=1
            )
            if can_prepend_knowledge:
                if CAIEncoderDecoderModel.decoder_based_glossary:
                    inputs_embeds = torch.cat([glossary_embeds, inputs_embeds], dim=1)
                    attention_mask = torch.cat([glossary_attentions, attention_mask], dim=1)
                else:
                    self.embed_positions.attention_mask = glossary_attentions
                    positional_embeds = self.embed_positions(torch.ones(glossary_embeds.size()[:-1]).to(torch.int))
                    glossary_embeds = glossary_embeds + positional_embeds
                    encoder_hidden_states = torch.cat([encoder_hidden_states, glossary_embeds], dim=1)
                    encoder_attention_mask = torch.cat([encoder_attention_mask, glossary_attentions], dim=1)

        self.embed_positions.attention_mask = attention_mask
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
        config.tie_word_embeddings = not config.remapped_tokens
        BartPretrainedModel.__init__(self, config)
        self.model = M2MDecoderWithPooledContextWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.remapped_tokens:
            self.resize_token_embeddings(self.config.vocab_size)

    def generated_prefix(self, tokenizer):
        lang_token = tokenizer.encode(self.config.target_language_code, add_special_tokens=False)[0]
        return [self.config.decoder_start_token_id]*2 + [lang_token]*2

    def remap_tokens(self, tokenizer):
        new_weights = self.model.decoder.embed_tokens.weight[tokenizer.used_tokens, :]
        new_embed_tokens = torch.nn.Embedding(
            new_weights.shape[0], new_weights.shape[1], padding_idx=self.model.decoder.embed_tokens.padding_idx)
        new_embed_tokens.weight = torch.nn.Parameter(new_weights)
        self.model.decoder.embed_tokens = new_embed_tokens

        new_weights = self.lm_head.weight[tokenizer.used_tokens, :]
        self.lm_head = torch.nn.Linear(new_weights.shape[1], new_weights.shape[0], bias=False)
        self.lm_head.weight = torch.nn.Parameter(new_weights)
        self.lm_head.weight.data = new_weights.data

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
        glossary_source=None,
        glossary_target=None,
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
            glossary_source=glossary_source,
            glossary_target=glossary_target,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            if glossary_source is not None:
                raise NotImplementedError("The loss function should ignore the glossary tokens but this is not "
                                          "implemented in the decoder yet, only in the encoder-decoder model")

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
