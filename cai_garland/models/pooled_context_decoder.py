# pylint: disable=no-member
import copy
from typing import Optional

from torch import nn

from transformers import AutoConfig, AutoModel, BartConfig, BartPretrainedModel, BartForCausalLM
from transformers.models.bart.modeling_bart import BartDecoder
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


# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModel.register(BartWithPooledContextConfig, BartWithPooledContextForCausalLM)
