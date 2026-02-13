import numpy
from typing import List

from suber.constants import END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL, MASK_SYMBOL
from suber.data_types import LineBreak, Segment, TimedWord


def segment_to_string(segment: Segment, include_line_breaks=False, include_last_break=True,
                      mask_all_words=False) -> str:
    if not include_line_breaks:
        assert not mask_all_words, (
            "Refusing to mask all words when not printing breaks, output would contain only mask symbols.")
        return " ".join(word.string for word in segment.word_list)

    word_list_with_breaks = []
    for word in segment.word_list:
        word_list_with_breaks.append(MASK_SYMBOL if mask_all_words else word.string)

        if word.line_break == LineBreak.END_OF_LINE:
            word_list_with_breaks.append(END_OF_LINE_SYMBOL)
        elif word.line_break == LineBreak.END_OF_BLOCK:
            word_list_with_breaks.append(END_OF_BLOCK_SYMBOL)

    if not include_last_break and word_list_with_breaks and word_list_with_breaks[-1] == END_OF_BLOCK_SYMBOL:
        word_list_with_breaks.pop()

    return " ".join(word_list_with_breaks)


def get_segment_to_string_opts_from_metric(metric: str):
    include_breaks = False
    mask_words = False
    if metric.endswith("-br"):
        include_breaks = True
        mask_words = True
        metric = metric[:-len("-br")]
    elif metric.endswith("-seg"):
        include_breaks = True
        metric = metric[:-len("-seg")]

    return include_breaks, mask_words, metric


def set_approximate_word_times(word_list: List[TimedWord], subtitle_start_time: float, subtitle_end_time: float):
    """
    Linearly interpolates word times from the subtitle start and end time as described in
    https://www.isca-archive.org/interspeech_2021/cherry21_interspeech.pdf
    """
    # Remove small margin to guarantee the first and last word will always be counted as within the subtitle.
    epsilon = 1e-8
    subtitle_start_time = subtitle_start_time + epsilon
    subtitle_end_time = subtitle_end_time - epsilon

    num_words = len(word_list)
    duration = subtitle_end_time - subtitle_start_time
    assert duration >= 0

    approximate_word_times = numpy.linspace(start=subtitle_start_time, stop=subtitle_end_time, num=num_words)
    for word_time, word in zip(approximate_word_times, word_list):
        word.approximate_word_time = word_time
