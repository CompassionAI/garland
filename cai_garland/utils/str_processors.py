import re


def remove_newline(segment):
    return segment.replace('\n', '').strip()


def replace_newline_with_space(segment):
    return segment.replace('\n', ' ').strip()


def remove_consecutive_spaces(segment):
    return re.sub('\s+', ' ', segment).strip()