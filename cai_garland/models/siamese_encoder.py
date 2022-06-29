import copy
from typing import Optional, Dict

import torch

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from transformers.utils import logging
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class SiameseEncoderConfig(PretrainedConfig):
    """Configuration class to store the composed configuration of a Siamese stack of base encoders."""
    model_type = "siamese-encoder"
    is_composition = True

    def __init__(self, **kwargs):
        assert (
            "encoder" in kwargs
        ), "Siamese encoder config has to be initialized with a base encoder config"
        assert (
            "num_registers" in kwargs
        ), "Siamese encoder config has to be initialized with a number of registers"

        super().__init__(**kwargs)

        encoder_config = kwargs["encoder"]
        encoder_model_type = encoder_config.pop("model_type")
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.num_registers = kwargs["num_registers"]

        self.hidden_size = self.encoder.hidden_size * self.num_registers

    def to_dict(self) -> Dict[any, any]:
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["model_type"] = self.__class__.model_type
        output["transformers_version"] = self.encoder.transformers_version
        return output

    @classmethod
    def from_base_encoder_config(
        cls,
        encoder_config: PretrainedConfig,
        num_registers: int,
        **kwargs
    ) -> PretrainedConfig:
        """Initialize a configuration for a Siamese encoder stack.

        Args:
            encoder_config:
                Configuration object for the base encoder.
            tokenizer:
                Tokenizer for the base encoder, needed for the special tokens.
            num_registers:
                Number of Siamese replicas (registers) to make for the base encoder.
        """

        return cls(encoder=encoder_config.to_dict(), num_registers=num_registers, **kwargs)


# This injects our new model config type into the Hugging Face factory for configs without having to modify their
#   code base. If ever this is contributed to the Transformers main branch, this should be moved.
CONFIG_MAPPING.register(SiameseEncoderConfig.model_type, SiameseEncoderConfig)


class SiameseEncoderModel(PreTrainedModel):
    """This utility class wraps a headless encoder, such as BERT or AlBERT, and creates a parallel Siamese stack of
    `num_registers` number of this encoder. We call each of these Siamese encoders a "register". This wrapper class
    splits input along the eor token in the tokenizer, feeds each entry of the split into the corresponding register
    encoder, and concatenates the outputs so as to look like an ordinary encoder to any subsequent decoder it is
    combined with via the EncoderDecoderModel class.

    *NB*: This wrapper class assumes the base encoder has the bos and eos tokens at the start and end of the input.
    """

    config_class = SiameseEncoderConfig
    base_model_prefix = "siamese_encoder"

    def __init__(self, base_encoder: PreTrainedModel, config: SiameseEncoderConfig):
        """

        Args:
            base_encoder (:obj:`PreTrainedModel`):
                An instance of the base encoder model to replicate in the Siamese stack.
            config (:obj:`SiameseEncoderConfig`):
                A configuration object for the Siamese stack.
        """

        super().__init__(config)
        self.base_encoder = base_encoder

        if self.base_encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Encoder {self.base_encoder.__class__} config overwritten by input config: {self.config.encoder}"
            )
        self.base_encoder.config = self.config.encoder

    def get_encoder(self):
        return self.base_encoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return None     # This model is always headless

    def set_output_embeddings(self, new_embeddings):
        raise ValueError("Siamese encoders are always headless, they have no output embedding layer")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        import ipdb; ipdb.set_trace()

        if len(input_ids) > 1:
            raise ValueError("Batch forward not implemented")

        bos, eos = input_ids[0][0], input_ids[0][-1]    # Assumes the base encoder has bos and eos tokens at the start
                                                        #   and end of the input!
        src_tokens = src_tokens[0]
        split_idxs = (src_tokens == self.eor_token_id).nonzero(as_tuple=True)[0].tolist()
        tokens_res, cur_split_idx = [], 0
        for new_split_idx in split_idxs:
            tokens_res.append(src_tokens[cur_split_idx:new_split_idx].view(1, -1))
            cur_split_idx = new_split_idx + 1
        tokens_res.append(src_tokens[cur_split_idx:].view(1, -1))
        for _ in range(len(tokens_res), self.num_registers):
            tokens_res.append(torch.LongTensor([bos, eos]).view(1, -1))
        lengths_res = [torch.LongTensor([cur_tokens.size(1)]) for cur_tokens in tokens_res]
        tokens_res, lengths_res = [[t.to(src_tokens) for t in ts] for ts in [tokens_res, lengths_res]]
        src_tokens, src_lengths = tokens_res, lengths_res


        encoder_outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        import ipdb; ipdb.set_trace()
