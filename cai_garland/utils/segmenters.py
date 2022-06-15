def opening_shad_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return ['།' + sent if not sent[0] == '།' else sent for sent in bo_text.strip().split(' །') if len(sent) > 0]


def closing_shad_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return [x.strip() + '།' for x in bo_text.strip().split('། ') if len(x.strip()) > 0]


def double_shad_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return [x.strip() for x in bo_text.strip().split('།།') if len(x.strip()) > 0]


def line_break_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return [x.strip() for x in bo_text.split('\n') if len(x.strip()) > 0]


def target_token_count_segmenter(bo_text, tokenizer, max_register_length=128, num_special_tokens=2, **kwargs):
    bo_segments = opening_shad_segmenter(bo_text)
    available_space = max_register_length - num_special_tokens
    bo_token_lengths = [len(tokenizer.encode(bo_segment, add_special_tokens=False)) for bo_segment in bo_segments]
    if max(bo_token_lengths) > available_space:
        new_segments = []
        for idx, (bo_segment, tkn_length) in enumerate(zip(bo_segments, bo_token_lengths)):
            if tkn_length > available_space:
                new_segments.extend(closing_shad_segmenter(bo_segment))
            else:
                new_segments.append(bo_segment)
        bo_segments = new_segments
    bo_token_lengths = [len(tokenizer.encode(bo_segment, add_special_tokens=False)) for bo_segment in bo_segments]
    if max(bo_token_lengths) > available_space:
        raise ValueError("Tokenized Tibetan text is too long for register encoding")
    bo_registers, register_start, register_idx = [], 0, 0
    while register_idx < len(bo_token_lengths):
        while sum(bo_token_lengths[register_start:register_idx + 1]) <= available_space:
            if register_idx == len(bo_token_lengths):
                break
            register_idx += 1
        if register_idx == register_start:
            continue
        bo_registers.append(' '.join(bo_segments[register_start:register_idx]).strip())
        register_start = register_idx
    return bo_registers
