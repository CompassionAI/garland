import os
import re
import json
import unicodedata

from cai_common.dict import tibetan_digits, tibetan_halves


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
        return re.sub(r'\s+', ' ', segment).strip()


class ProcessorRemoveLineNumbers:
    def __call__(self, segment: str) -> str:
        return re.sub(f"[{''.join(tibetan_digits)}{''.join(tibetan_halves)}]+༽", '', segment).strip()


class ProcessorLowerCase:
    def __call__(self, segment: str) -> str:
        return segment.lower()


class ProcessorRemoveAccents:
    def __call__(self, segment: str) -> str:
        nfkd_form = unicodedata.normalize('NFKD', segment)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


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
        self.opening = opening.replace("[", r"\[").replace("]", r"\]")
        self.closing = closing.replace("[", r"\[").replace("]", r"\]")
        self._re = re.compile(f"[{self.opening}].*?[{self.closing}]")

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


class ProcessorSymbolCleaningJSON:
    base_dir = None

    def __init__(self, symbol_cleaning_json_file) -> None:
        if self.base_dir is None:
            raise ValueError("Set base_dir before instantiating ProcessorSymbolCleaningJSON")
        with open(os.path.join(self.base_dir, symbol_cleaning_json_file), 'r') as f:
            self.symbol_cleaning_map = json.load(f)
        self.skip_lines = set(self.symbol_cleaning_map.pop("skip_this_line", []))

    def __call__(self, segment: str) -> str:
        if any([skip_char in segment for skip_char in self.skip_lines]):
            return ""
        for bad_c, good_c in self.symbol_cleaning_map.items():
            segment = segment.replace(bad_c, good_c)
        return segment
