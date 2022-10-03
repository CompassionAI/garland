import logging

from typing import List, Any
from tqdm.auto import tqdm
from torch.nn.functional import softmax
from transformers import AutoConfig, AlbertForSequenceClassification
from cai_common.models.utils import get_local_ckpt, get_cai_config
from cai_manas.tokenizer.tokenizer import CAITokenizer


logger = logging.getLogger(__name__)


def _prepend_shad_if_needed(bo_text):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return bo_text


class SegmenterBase:
    def __init__(self, *_args, translator=None) -> None:
        self.translator = translator


class SegmenterNone(SegmenterBase):
    def __call__(self, bo_text: str, **kwargs: Any) -> List[str]:
        return [bo_text]


class SegmenterOpeningShad(SegmenterBase):
    def __call__(self, bo_text: str, **kwargs: Any) -> List[str]:
        bo_text = _prepend_shad_if_needed(bo_text)
        return ['།' + sent if not sent[0] == '།' else sent for sent in bo_text.strip().split(' །') if len(sent) > 0]


class SegmenterClosingShad(SegmenterBase):
    def __init__(self, *_args, prepend_shad: bool = True) -> None:
        super().__init__(translator=None)
        self.prepend_shad = prepend_shad

    def __call__(self, bo_text: str, **kwargs: Any) -> List[str]:
        if self.prepend_shad:
            bo_text = _prepend_shad_if_needed(bo_text)
        return [x.strip() + '།' for x in bo_text.strip().split('། ') if len(x.strip()) > 0]


class SegmenterOpeningOrClosingShad(SegmenterBase):
    def __call__(self, bo_text: str, translator=None, **kwargs: Any) -> List[str]:
        if translator is None:
            raise ValueError("SegmenterOpeningOrClosingShad needs to have the translator helper class passed in")
        bo_segments = SegmenterOpeningShad()(bo_text)
        secondary_segmenter = SegmenterTargetTokenCount()
        max_length = kwargs.get("max_length", translator.model.encoder.max_length)
        available_space = max_length - translator.tokenizer.num_special_tokens_to_add(pair=False)

        res = []
        for bo_segment in bo_segments:
            if len(translator.tokenizer.encode(bo_segment, add_special_tokens=False)) > available_space:
                res.extend(secondary_segmenter(bo_segment, max_length=max_length, translator=translator))
            else:
                res.append(bo_segment)
        return res


class SegmenterDoubleShad(SegmenterBase):
    def __call__(self, bo_text: str, **kwargs: Any) -> List[str]:
        bo_text = _prepend_shad_if_needed(bo_text)
        return [x.strip() for x in bo_text.strip().split('།།') if len(x.strip()) > 0]


class SegmenterLineBreak(SegmenterBase):
    def __call__(self, bo_text: str, **kwargs: Any) -> List[str]:
        bo_text = _prepend_shad_if_needed(bo_text)
        return [x.strip() for x in bo_text.split('\n') if len(x.strip()) > 0]


class SegmenterTargetTokenCount(SegmenterBase):
    def __call__(self, bo_text: str, translator=None, tqdm=tqdm, **kwargs: Any) -> List[str]:       # pylint: disable=redefined-outer-name
        # This segmenter packs bo_text into registers. Each register is of the longest possible length that fits into
        #   the encoder.
        #
        #   1. First, segment by opening shads.
        #   2. If some segments don't fit, subdivide those segments again by the closing shad.
        #   3. Greedily sweep left to right and append to registers as you go until you run out of space, after which
        #       start a new register.
        #
        # Note that this algorithm does not take any maximum number of registers. Also, after the closing shad
        #   segmentation in the second step there may still be segments of length longer than the encoder length, in
        #   which case their encoding will fail during translation (or they need to be further segmented downstream).
        if translator is None:
            raise ValueError("SegmenterTargetTokenCount needs to have the translator helper class passed in")
        bo_segments = SegmenterOpeningShad()(bo_text)
        available_space = kwargs.get("max_length", translator.model.encoder.max_length) - \
            translator.tokenizer.num_special_tokens_to_add(pair=False)

        bo_token_lengths = [
            len(translator.tokenizer.encode(bo_segment, add_special_tokens=False))
            for bo_segment in tqdm(bo_segments, desc="Calculating lengths", leave=False)
        ]

        if max(bo_token_lengths) > available_space:
            new_segments = []
            for bo_segment, tkn_length in zip(bo_segments, bo_token_lengths):
                if tkn_length > available_space:
                    new_segments.extend(SegmenterClosingShad(prepend_shad=False)(bo_segment))
                else:
                    new_segments.append(bo_segment)
            bo_segments = new_segments
            bo_token_lengths = [
                len(translator.tokenizer.encode(bo_segment, add_special_tokens=False))
                for bo_segment in tqdm(bo_segments, desc="Re-calculating lengths", leave=False)
            ]

        registers, cur_register, cur_register_length = [], [], 0
        while True:
            if (len(bo_segments) == 0) or (cur_register_length + bo_token_lengths[0] > available_space):
                registers.append(' '.join(cur_register).strip())
                cur_register, cur_register_length = [], 0

            if len(bo_segments) == 0:
                break

            cur_register.append(bo_segments.pop(0))
            cur_register_length += bo_token_lengths.pop(0)

        return registers


class SegmenterModel(SegmenterBase):
    discourage_long_segments = False

    model_name = None
    tokenizer = None
    model_cfg = None
    model = None

    def __init__(self, model_name, translator=None):
        if translator is None:
            raise ValueError("SegmenterModel init needs to have the translator helper class passed in")
        super().__init__(translator=translator)
        if not SegmenterModel.model_name == model_name:
            logger.info(f"Loading segmentation model {model_name}")
            SegmenterModel.model_name = model_name

            local_ckpt = get_local_ckpt(model_name)
            logger.debug(f"  Local model checkpoint {model_name} resolved to {local_ckpt}")

            logger.debug("  Loading CAI segmenter model config")
            cai_segmenter_config = get_cai_config(model_name)
            base_model = cai_segmenter_config['base_model']
            logger.debug(f"  Base model resolved to {base_model}")

            logger.debug("  Loading CAI base model config")
            cai_base_config = get_cai_config(base_model)
            config_name = cai_base_config['hf_base_model_name']
            tokenizer_name = cai_base_config['tokenizer_name']

            logger.info(f"  Loading tokenizer {tokenizer_name}")
            SegmenterModel.tokenizer = CAITokenizer.from_pretrained(CAITokenizer.get_local_model_dir(tokenizer_name))

            logger.info("  Loading model")
            logger.debug("  Loading Huggingface model config")
            SegmenterModel.model_cfg = AutoConfig.from_pretrained(
                config_name, vocab_size=SegmenterModel.tokenizer.vocab_size, num_labels=2)
            SegmenterModel.model = AlbertForSequenceClassification.from_pretrained(local_ckpt, config=SegmenterModel.model_cfg)
            logger.debug("  Configuring model")
            SegmenterModel.model.resize_token_embeddings(len(SegmenterModel.tokenizer))
            SegmenterModel.model.eval()

    def __call__(self, bo_text: str, translator=None, tqdm=tqdm, **kwargs: Any) -> List[str]:       # pylint: disable=redefined-outer-name
        bo_segments = SegmenterClosingShad()(bo_text)
        max_length = kwargs.get("max_length", self.translator.model.encoder.max_length)
        available_space = max_length - self.translator.tokenizer.num_special_tokens_to_add(pair=False)

        res, candidate = [], ""
        for segment in tqdm(bo_segments, desc="Segmenting", leave=False):
            old_candidate = candidate
            candidate = (candidate + ' ' + segment).strip()
            candidate_len = len(translator.tokenizer.encode(candidate, add_special_tokens=False))
            if candidate_len > available_space:
                if not old_candidate == "":
                    res.append(old_candidate)
                candidate = segment
                candidate_len = len(translator.tokenizer.encode(candidate, add_special_tokens=False))
            scores = SegmenterModel.model(**SegmenterModel.tokenizer(candidate, return_tensors="pt")).logits.detach()
            scores = softmax(scores, dim=1)
            model_score = float(scores[0][1])
            needed_score = min(1 - 0.5 * candidate_len / available_space, 0.8) if self.discourage_long_segments else 0.5
            if model_score > needed_score:
                res.append(candidate)
                candidate = ""
        if not candidate == "":
            res.append(candidate)
        return res
