# pylint: disable=no-member
import copy
from dataclasses import dataclass
from typing import Optional, Dict, Union, Any

import torch

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class BaseModelOutputWithAttentionMask(BaseModelOutput):
    attention_mask: Optional[torch.FloatTensor] = None


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
        self.eor_token_id = kwargs["eor_token_id"]

        self.hidden_size = self.encoder.hidden_size

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
        eor_token_id: int,
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
            eor_token_id:
                The ID of the [eor] token in the tokenizer
        """

        return cls(encoder=encoder_config.to_dict(), num_registers=num_registers, eor_token_id=eor_token_id, **kwargs)


# This injects our new model config type into the Hugging Face factory for configs without having to modify their
#   code base. If ever this is contributed to the Transformers main branch, this should be moved.
AutoConfig.register(SiameseEncoderConfig.model_type, SiameseEncoderConfig)


class SiameseEncoderModel(PreTrainedModel):
    """This utility class wraps a headless encoder, such as BERT or AlBERT, and creates a parallel Siamese stack of
    `num_registers` number of this encoder. We call each of these Siamese encoders a "register". This wrapper class
    splits input along the eor token in the tokenizer, feeds each entry of the split into the corresponding register
    encoder, and concatenates the outputs so as to look like an ordinary encoder to any subsequent decoder it is
    combined with via the EncoderDecoderModel class.
    """

    config_class = SiameseEncoderConfig
    base_model_prefix = "siamese_encoder"

    def __init__(self, config: SiameseEncoderConfig, base_encoder: PreTrainedModel = None):
        """

        Args:
            base_encoder (:obj:`PreTrainedModel`):
                An instance of the base encoder model to replicate in the Siamese stack.
            config (:obj:`SiameseEncoderConfig`):
                A configuration object for the Siamese stack.
        """

        super().__init__(config)
        if base_encoder is None:
            base_encoder = AutoModel.from_config(config.encoder)

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

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        return super().estimate_tokens({
            self.main_input_name: torch.cat(input_dict[self.main_input_name], dim=1)
        })

    def split_tokens_into_registers(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """Accepts tokenizer encoded tokens and splits them into registers, together with the appropriate attention
            masks.
        """
        if len(input_ids) > 1:
            raise NotImplementedError("Siamese encoder does not currently support batch inference")

        from ..data.siamese_collator import _split_list
        to_device = input_ids.device
        tokens = input_ids[0].cpu().tolist()
        bos, eos = tokens[0], tokens[-1]
        tokens = tokens[1:-1]
        register_splits = [idx for idx, token in enumerate(tokens) if token == self.config.eor_token_id]
        input_ids = [
            torch.IntTensor([[bos] + reg + [eos]]) for reg in _split_list(tokens, register_splits)
        ]
        for _ in range(len(input_ids), self.config.num_registers):
            input_ids.append(torch.IntTensor([[bos, eos]]))
        attention_mask = [torch.IntTensor([[1]*len(reg[0])]) for reg in input_ids]
        return {
            "input_ids": [tensor.to(to_device) for tensor in input_ids],
            "attention_mask": [tensor.to(to_device) for tensor in attention_mask]
        }


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
        if inputs_embeds:
            raise NotImplementedError("Siamese encoder does not currently support inputs_embeds")
        if output_attentions:
            raise NotImplementedError("Siamese encoder does not currently support outputting attentions")

        if isinstance(input_ids, torch.Tensor):
            _splits = self.split_tokens_into_registers(input_ids)
            input_ids = _splits['input_ids']
            attention_mask = _splits['attention_mask']

        encoder_outputs = []
        for reg_input_ids, reg_attention_mask in zip(input_ids, attention_mask):
            encoder_outputs.append(
                self.base_encoder(
                    input_ids=reg_input_ids,
                    attention_mask=reg_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
            )

        catted_attention_mask = torch.cat(attention_mask, dim=1)
        last_hidden_state = torch.cat([enc_out.last_hidden_state for enc_out in encoder_outputs], dim=1)
        if output_hidden_states:
            hidden_states = tuple(
                [
                    torch.cat([enc_out.hidden_states[state_idx] for enc_out in encoder_outputs], dim=1)
                    for state_idx in range(len(encoder_outputs[0].hidden_states))
                ]
            )
        else:
            hidden_states = None
        if not return_dict:
            # Add more to here if implemented
            return (last_hidden_state,) + tuple(v for v in [hidden_states] if v is not None)
        return BaseModelOutputWithAttentionMask(
            last_hidden_state=last_hidden_state,
            attention_mask=catted_attention_mask,
            hidden_states=hidden_states,
            attentions=None,
        )


# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModel.register(SiameseEncoderConfig, SiameseEncoderModel)
