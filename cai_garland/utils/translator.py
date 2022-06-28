import logging

from transformers import EncoderDecoderModel

from cai_common.models.utils import get_local_ckpt, get_cai_config
from cai_garland.models.factory import make_bilingual_tokenizer


logger = logging.getLogger(__name__)


class Translator:
    """A machine translation utility class, abstracting the pipeline from a potentially very long source document to a
    machine translated output. See cai_garland.cli.translate for usage examples.

    Attributes:
        model: An EncoderDecoderModel for the fine-tuned encoder-decoder translation stack.
        tokenizer: A BilingualTokenizer for the source and target languages.
        num_beams: Number of beams to use in the beam search (default is 20).
    """

    def __init__(self, model_ckpt: str) -> None:
        """Loads all the relevant data and models for machine translation.
        
        Args:
            model_ckpt:  Name of the fine-tuned model checkpoint in the data registry to use for translation. For
                example, olive-cormorant-bart."""
        local_ckpt = get_local_ckpt(model_ckpt, model_dir=True)
        logger.debug(f"Local model checkpoint {model_ckpt} resolved to {local_ckpt}")

        self.model = EncoderDecoderModel.from_pretrained(local_ckpt)
        logger.debug(f"Encoder: {self.model.encoder}")
        logger.debug(f"Decoder: {self.model.decoder}")

        logger.debug(f"Loading CAI translation model config")
        cai_base_config = get_cai_config(model_ckpt)
        encoder_name = cai_base_config['encoder_model_name']
        encoder_length = cai_base_config['encoder_max_length']
        decoder_name = cai_base_config['decoder_model_name']
        decoder_length = cai_base_config['decoder_max_length']
        logger.debug(f"Encoder name={encoder_name}, length={encoder_length}")
        logger.debug(f"Decoder name={decoder_name}, length={decoder_length}")
        self.model.encoder.max_length = encoder_length
        self.model.decoder.max_length = decoder_length

        logger.debug("Loading bilingual tokenizer")
        self.tokenizer = make_bilingual_tokenizer(encoder_name, decoder_name)

        logger.debug("Configuring model")
        self.model.eval()

        self.num_beams = 20

    def translate(self, bo_text: str) -> str:
        """Translate the input Tibtean.
        
        Args:
            bo_text: The Tibetan text (not tokens) to translate, as a unicode string.
        
        Returns:
            The translated text (not tokens)."""

        bo_tokens = self.tokenizer(bo_text, return_tensors="pt").input_ids
        if len(bo_tokens[0]) > self.model.encoder.max_length:
            raise ValueError(f"Translation input too long: encoder maximum length is {self.model.encoder.max_length}, "
                             f"input tokenizes to {len(bo_tokens[0])} tokens.")
        logger.debug(f"Tokenized input: {bo_tokens[0]}")
        logger.debug(f"Tokenized input length: {len(bo_tokens[0])}")
        preds = self.model.generate(bo_tokens, max_length=self.model.decoder.max_length, num_beams=self.num_beams)[0]
        logger.debug(f"Generated tokens: {preds}")
        logger.debug(f"Generated tokens length: {len(preds)}")
        with self.tokenizer.as_target_tokenizer():
            return self.tokenizer.decode(preds, skip_special_tokens=True).strip()
