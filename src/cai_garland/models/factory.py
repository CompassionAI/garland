import os
import logging

from cai_common.models.utils import get_local_ckpt, get_cai_config, get_local_file
from cai_manas.tokenizer import CAITokenizer
from cai_garland.models.cai_nllb_tokenizer import CAINllbTokenizerFast
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from .bilingual_tokenizer import BilingualTokenizer
from .siamese_encoder import SiameseEncoderConfig, SiameseEncoderModel
from .pooled_context_decoder_bart import BartWithPooledContextForCausalLM
from .pooled_context_decoder_m2m import M2MWithPooledContextForCausalLM, M2MForCGWithRemappedEncoder
from .cai_encoder_decoder import CAIEncoderDecoderModel, CAIEncoderDecoderConfig


logger = logging.getLogger(__name__)


_tokenizer_classes = {
    "monolingual": CAITokenizer,
    "nllb": CAINllbTokenizerFast,
    "default": CAITokenizer
}


_causal_LM_classes = {
    "bart": BartWithPooledContextForCausalLM,
    "m2m": M2MWithPooledContextForCausalLM,
    "m2m_with_remapped_encoder": M2MForCGWithRemappedEncoder,
    "default": BartWithPooledContextForCausalLM
}


def _make_named_tokenizer(packed_name, hf_tokenizer_factory=AutoTokenizer):
    # Apply the names rules in make_encoder_decoder. Returns a tokenizer
    if packed_name.startswith('cai:'):
        logging.debug(f"Loading tokenizer {packed_name} from CompassionAI data registry")

        cai_name = packed_name[4:].strip()
        cai_config = get_cai_config(cai_name)

        if cai_config.get('is_derived', False):
            if 'tokenizer_override' in cai_config:
                tokenizer_cfg = cai_config['tokenizer_override']
                tokenizer = _make_named_tokenizer(
                    cai_config['base_model_name'],
                    hf_tokenizer_factory=_tokenizer_classes[tokenizer_cfg.get("tokenizer_class", "default")]
                )
                if 'remapping_file' in tokenizer_cfg:
                    tokenizer.remap_tokens(get_local_file(tokenizer_cfg['remapping_file']))
                return tokenizer

            base_tokenizer = _make_named_tokenizer(cai_config['base_model_name'])
            if cai_config.get('siamese', False):
                base_tokenizer.enable_siamese()
            return base_tokenizer

        tokenizer_name = cai_config['tokenizer_name']
        logger.debug(f"Loading tokenizer {tokenizer_name}")
        tokenizer = CAITokenizer.from_pretrained(CAITokenizer.get_local_model_dir(tokenizer_name))
    elif packed_name.startswith('hf:'):
        logging.debug(f"Loading tokenizer {packed_name} from Hugging Face")

        hf_name = packed_name[3:].strip()

        logger.debug(f"Loading tokenizer {hf_name}")
        tokenizer = hf_tokenizer_factory.from_pretrained(hf_name, src_lang="bod_Tibt", tgt_lang="eng_Latn")
    else:
        raise ValueError("Model name needs to start with either cai: or hf:")
    return tokenizer


def _make_named_model(packed_name, hf_model_factory, tokenizer=None, config_args={}):
    # Apply the names rules in make_encoder_decoder. Returns a model.
    #   *NB:* The optional tokenizer should be the source/target tokenizer, NOT a bilingual tokenizer.
    if packed_name.startswith('cai:'):
        logging.debug(f"Loading {packed_name} from CompassionAI data registry")

        cai_name = packed_name[4:].strip()
        cai_config = get_cai_config(cai_name)

        if cai_config.get('siamese', False):
            if tokenizer is None:
                raise ValueError("Need a tokenizer to construct a Siamese encoder")
            logger.debug("Constructing Siamese model")
            logger.debug("    Loading base model")
            base_model = _make_named_model(cai_config['base_model_name'], hf_model_factory, tokenizer=tokenizer)
            logger.debug("    Constructing Siamese config")
            siamese_config = SiameseEncoderConfig.from_base_encoder_config(
                base_model.config,
                cai_config['num_registers'],
                tokenizer.eor_token_id
            )
            logger.debug("    Constructing Siamese wrapper model")
            model = SiameseEncoderModel(siamese_config, base_encoder=base_model)
        elif cai_config.get('pooled-context-injection', False):
            logger.debug("Constructing a decoder with pooled context injection")
            logger.debug("    Constructing PCI wrapper model")
            model = _make_named_model(
                cai_config['base_model_name'],
                _causal_LM_classes[cai_config.get("model_class", "default")],
                tokenizer=tokenizer,
                config_args=cai_config.get("config-args", {})
            )
        elif cai_config.get('extract_encoder', False):
            logger.debug("Extracting the encoder from an encoder-decoder stack")
            model = _make_named_model(
                cai_config['base_model_name'],
                _causal_LM_classes[cai_config.get("model_class", "default")],
                tokenizer=tokenizer,
                config_args=cai_config.get("config-args", {})
            )
            model = model.model.encoder
        else:
            local_ckpt = get_local_ckpt(cai_name)

            hf_config_name = cai_config['hf_base_model_name']

            logger.debug("Loading Huggingface base model config")
            model_cfg = AutoConfig.from_pretrained(hf_config_name)

            logger.debug("Loading model")
            model = hf_model_factory.from_pretrained(local_ckpt, config=model_cfg)
            if tokenizer is not None:
                model.resize_token_embeddings(len(tokenizer))
        if isinstance(tokenizer, CAINllbTokenizerFast) and tokenizer.is_remapped:
            model.remap_tokens(tokenizer)
    elif packed_name.startswith('hf:'):
        logging.debug(f"Loading {packed_name} from Hugging Face")

        hf_name = packed_name[3:].strip()

        logger.debug(f"Loading model {hf_name}")
        config = getattr(hf_model_factory, "config_class", AutoConfig).from_pretrained(hf_name)
        for key, val in config_args.items():
            setattr(config, key, val)
        model = hf_model_factory.from_pretrained(hf_name, config=config)
    else:
        raise ValueError("Model name needs to start with either cai: or hf:")
    return model


def make_encoder(encoder_name: str):
    """This is a configurable factory for the various models we experiment with for machine translation.

    The rules for the names are:
        cai:<model_name> - name of a checkpoint in the data registry.
        hf:<model_name> - name of a pretrained model in the Hugging Face model registry.

    Args:
        encoder_name: Name of the encoder model.

    Returns:
        A tuple of the model and its tokenizer.
    """

    tokenizer = _make_named_tokenizer(encoder_name)
    encoder = _make_named_model(encoder_name, AutoModel, tokenizer=tokenizer)
    return encoder, tokenizer


def make_bilingual_tokenizer(encoder_name: str, decoder_name: str):
    """This is a configurable factory for our bilingual tokenizers we use for machine translation.

    The rules for the names are:
        cai:<model_name> - name of a model (not tokenizer) checkpoint in the data registry.
        hf:<model_name> - name of a pretrained model in the Hugging Face model registry.

    Args:
        encoder_name: Name of the encoder tokenizer.
        decoder_name: Name of the decoder tokenizer.

    Returns:
        A bilingual tokenizer.
    """

    tokenizer = BilingualTokenizer(_make_named_tokenizer(encoder_name), _make_named_tokenizer(decoder_name))
    return tokenizer


def make_encoder_decoder(encoder_name: str, decoder_name: str):
    """This is a configurable factory for the various models we experiment with for machine translation.

    The rules for the names are:
        cai:<model_name> - name of a checkpoint in the data registry.
        hf:<model_name> - name of a pretrained model in the Hugging Face model registry.

    Args:
        encoder_name: Name of the encoder model.
        decoder_name: Name of the decoder model.

    Returns:
        A tuple of the model and a bilingual tokenizer.
    """

    tokenizer = make_bilingual_tokenizer(encoder_name, decoder_name)

    encoder = _make_named_model(encoder_name, AutoModel, tokenizer=tokenizer.source_tokenizer)
    decoder = _make_named_model(decoder_name, AutoModelForCausalLM, tokenizer=tokenizer.target_tokenizer)

    encoder.config.is_decoder = False
    encoder.config.add_cross_attention = False

    decoder.config.is_decoder = True
    decoder.config.add_cross_attention = True

    config = CAIEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    model = CAIEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    # model.config.decoder_start_token_id = tokenizer.target_tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.target_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    return model, tokenizer
