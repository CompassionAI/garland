import re


class ProcessorRemoveNewLine:
    def __call__(self, segment: str) -> str:
        return segment.replace('\n', '').strip()


class ProcessorReplaceNewLineWithSpace:
    def __call__(self, segment: str) -> str:
       return segment.replace('\n', ' ').strip()


class ProcessorRemoveConsecutiveSpaces:
    def __call__(self, segment: str) -> str:
        return re.sub('\s+', ' ', segment).strip()