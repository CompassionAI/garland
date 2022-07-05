from tqdm.auto import tqdm


def _prepend_shad_if_needed(bo_text):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return bo_text


def none(bo_text, **kwargs):
    return [bo_text]


def opening_shad_segmenter(bo_text, **kwargs):
    bo_text = _prepend_shad_if_needed(bo_text)
    return ['།' + sent if not sent[0] == '།' else sent for sent in bo_text.strip().split(' །') if len(sent) > 0]


def closing_shad_segmenter(bo_text, **kwargs):
    bo_text = _prepend_shad_if_needed(bo_text)
    return [x.strip() + '།' for x in bo_text.strip().split('། ') if len(x.strip()) > 0]


def double_shad_segmenter(bo_text, **kwargs):
    bo_text = _prepend_shad_if_needed(bo_text)
    return [x.strip() for x in bo_text.strip().split('།།') if len(x.strip()) > 0]


def line_break_segmenter(bo_text, **kwargs):
    bo_text = _prepend_shad_if_needed(bo_text)
    return [x.strip() for x in bo_text.split('\n') if len(x.strip()) > 0]


def target_token_count_segmenter(bo_text, translator=None, tqdm=tqdm, **kwargs):
    # This segmenter packs bo_text into registers. Each register is of the longest possible length that fits into the
    #   encoder.
    #
    #   1. First, segment by opening shads.
    #   2. If some segments don't fit, subdivide those segments again by the closing shad.
    #   3. Greedily sweep left to right and append to registers as you go until you run out of space, after which start
    #       a new register.
    #
    # Note that this algorithm does not take any maximum number of registers. Also, after the closing shad segmentation
    #   in the second step there may still be segments of length longer than the encoder length, in which case their
    #   encoding will fail during translation (or they need to be further segmented downstream).
    if translator is None:
        raise ValueError("target_token_count_segmenter needs to have the translator helper class passed in")
    bo_segments = opening_shad_segmenter(bo_text)
    available_space = translator.model.encoder.max_length - translator.tokenizer.num_special_tokens_to_add(pair=False)

    bo_token_lengths = [
        len(translator.tokenizer.encode(bo_segment, add_special_tokens=False))
        for bo_segment in tqdm(bo_segments, desc="Calculating lengths", leave=False)
    ]

    if max(bo_token_lengths) > available_space:
        new_segments = []
        for bo_segment, tkn_length in zip(bo_segments, bo_token_lengths):
            if tkn_length > available_space:
                new_segments.extend(closing_shad_segmenter(bo_segment))
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
