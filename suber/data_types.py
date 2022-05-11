from enum import Enum
from dataclasses import dataclass
from typing import List


class LineBreak(Enum):
    NONE = 0
    END_OF_LINE = 1  # represented as '<eol>' in plain text files
    END_OF_BLOCK = 2  # represented as '<eob>' in plain text files


@dataclass
class Word:
    string: str
    line_break: LineBreak = LineBreak.NONE  # the line break after the word, if any


@dataclass(unsafe_hash=True)
# Needs to be hashable for cached edit distance in lib_ter.py, but 'approximate_word_time' is currently set after
# creation within SRTFileReader, so cannot set frozen=True.
# TODO: find clean solution
class TimedWord(Word):
    subtitle_start_time: float = None
    subtitle_end_time: float = None
    approximate_word_time: float = None  # usually interpolated from subtitle start and end time; for t-BLEU calculation


@dataclass
class Segment:
    word_list: List[Word]


@dataclass
class Subtitle(Segment):
    index: int
    start_time: float
    end_time: float
