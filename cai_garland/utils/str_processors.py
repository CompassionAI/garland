import re
import unicodedata


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
