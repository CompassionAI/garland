import logging

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel
)

from cai_manas.tokenizer import TibertTokenizer
from .bilingual_tokenizer import BilingualTokenizer
from cai_common.models.utils import get_local_ckpt, get_cai_config


logger = logging.getLogger(__name__)


def _unpack_name(packed_name, hf_model_factory):
    # Apply the names rules in make_encoder_decoder. Returns a tokenizer and a model
    if packed_name.startswith('cai:'):
        logging.debug(f"Loading {packed_name} from CompassionAI data registry")

        cai_name = packed_name[4:].strip()
        local_ckpt = get_local_ckpt(cai_name)
        cai_config = get_cai_config(cai_name)

        tokenizer_name = cai_config['tokenizer_name']
        hf_config_name = cai_config['hf_base_model_name']

        logger.debug(f"Loading tokenizer {tokenizer_name}")
        tokenizer = TibertTokenizer.from_pretrained(TibertTokenizer.get_local_model_dir(tokenizer_name))

        logger.debug(f"Loading Huggingface base model config")
        model_cfg = AutoConfig.from_pretrained(hf_config_name)

        logger.debug(f"Loading model")
        model = hf_model_factory.from_pretrained(local_ckpt, config=model_cfg)
        model.resize_token_embeddings(len(tokenizer))

        return tokenizer, model
    elif packed_name.startswith('hf:'):
        logging.debug(f"Loading {packed_name} from Hugging Face")

        hf_name = packed_name[3:].strip()
        
        logger.debug(f"Loading tokenizer {hf_name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

        logger.debug(f"Loading model {hf_name}")
        model = hf_model_factory.from_pretrained(hf_name)

        return tokenizer, model
    else:
        raise ValueError("Model name needs to start with either cai: or hf:")


def make_encoder_decoder(encoder_name: str, decoder_name: str):
    """This is a configurable factory for the various models we experiment with for machine translation.

    The rules for the names are:
        cai:<model_name> - name of a checkpoint in the data registry.
        hf:<model_name> - name of a pretrained model in the Hugging Face model registry.

    Args:
        encoder_name: Name of the encoder model.
        decoder_name: Name of the decoder model.
    
    Returns:
        A tuple of the model, source language tokenizer, and target language tokenizer.
    """

    encoder_tokenizer, encoder = _unpack_name(encoder_name, AutoModel)
    decoder_tokenizer, decoder = _unpack_name(decoder_name, AutoModelForCausalLM)

    encoder.config.is_decoder = False
    encoder.config.add_cross_attention = False

    decoder.config.is_decoder = True
    decoder.config.add_cross_attention = True

    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)
    tokenizer = BilingualTokenizer(encoder_tokenizer, decoder_tokenizer)

    model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    return model, tokenizer
