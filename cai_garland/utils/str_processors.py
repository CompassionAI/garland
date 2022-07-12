import re
import unicodedata
from typing import List


class ProcessorStrip:
    def __call__(self, segment: str) -> str:
        return segment.strip()


class ProcessorRemoveNewLine:
    def __call__(self, segment: str) -> str:
        return segment.replace('\n', '').strip()


class ProcessorReplaceNewLineWithSpace:
    def __call__(self, segment: str) -> str:
       return segment.replace('\n', ' ').strip()


class ProcessorRemoveConsecutiveSpaces:
    def __call__(self, segment: str) -> str:
        return re.sub('\s+', ' ', segment).strip()


class ProcessorLowerCase:
    def __call__(self, segment: str) -> str:
        return segment.lower()


class ProcessorRemoveAccents:
    def __call__(self, segment: str) -> str:
        nfkd_form = unicodedata.normalize('NFKD', segment)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


class ProcessorRemoveDanglingShads:
    def __call__(self, segment: str) -> str:
        segment = segment.replace(" ། ", "").strip()
        if segment[:2] == "། ":
            segment = segment[2:]
        if segment[-2:] == " །":
            segment = segment[:-2]
        return segment


class ProcessorRemoveBracketed:
    def __init__(self, opening: str, closing: str) -> None:
        self.opening = opening
        self.closing = closing
        self._re = re.compile(f"[{opening}].*?[{closing}]")

    def __call__(self, segment: str) -> str:
        return self._re.sub("", segment)


class ProcessorRemoveCharacters:
    def __init__(self, to_remove: str) -> None:
        self.to_remove = to_remove
        self._re = re.compile(f"[{to_remove}]")

    def __call__(self, segment: str) -> str:
        return self._re.sub("", segment)


class ProcessorBlankOutAllWithCharacters:
    def __init__(self, to_blank_out: str) -> None:
        self.to_blank_out = set(to_blank_out)

    def __call__(self, segment: str) -> str:
        if len(self.to_blank_out.intersection(segment)) > 0:
            return ""
        return segment


class ProcessorBlankOutDanglingBrackets:
    def __init__(self, bracket_types: str) -> None:
        self.bracket_types = bracket_types

    def __call__(self, segment: str) -> str:
        for bracket in self.bracket_types:
            if segment.count(bracket) % 2 == 1:
                return ""
        return segment


class ProcessorReplaceCharacters:
    def __init__(self, from_: str, to: str) -> None:
        self.from_ = from_
        self.to = to

    def __call__(self, segment: str) -> str:
        return segment.replace(self.from_, self.to)