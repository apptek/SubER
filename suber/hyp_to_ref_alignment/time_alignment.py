import numpy
from typing import List, Optional

from suber.constants import ASIAN_LANGUAGE_CODES
from suber.data_types import Segment, Subtitle
from suber.tokenizers import reversibly_tokenize_segments, detokenize_segments


def time_align_hypothesis_to_reference(
        hypothesis: List[Segment], reference: List[Subtitle], language: Optional[str] = None) -> List[Subtitle]:
    """
    Re-segments the hypothesis segments according to the reference subtitle timings. The output hypothesis subtitles
    will have the same time stamps as the reference, and each will contain the words whose approximate times falls into
    these intervals, i.e. reference_subtitle.start_time < word.approximate_word_time < reference_subtitle.end_time.
    Hypothesis words that do not fall into any subtitle will be dropped.
    """

    if language in ASIAN_LANGUAGE_CODES:
        hypothesis = reversibly_tokenize_segments(hypothesis, language)

    aligned_hypothesis_word_lists = [[] for _ in reference]

    reference_start_times = numpy.array([subtitle.start_time for subtitle in reference])
    reference_end_times = numpy.array([subtitle.end_time for subtitle in reference])

    for segment in hypothesis:
        for word in segment.word_list:
            assert word.approximate_word_time is not None, "Should have been set by SRTFileReader. Is plain file used?"
            reference_subtitle_index = numpy.searchsorted(reference_start_times, word.approximate_word_time) - 1

            if reference_subtitle_index < 0:
                # Word is before first subtitle, drop it.
                continue

            if word.approximate_word_time < reference_end_times[reference_subtitle_index]:
                aligned_hypothesis_word_lists[reference_subtitle_index].append(word)

    aligned_hypothesis = []

    for index, word_list in enumerate(aligned_hypothesis_word_lists):
        reference_subtitle = reference[index]
        subtitle = Subtitle(
            word_list=word_list,
            index=reference_subtitle.index,
            start_time=reference_subtitle.start_time,
            end_time=reference_subtitle.end_time)

        aligned_hypothesis.append(subtitle)

    if language in ASIAN_LANGUAGE_CODES:
        aligned_hypothesis = detokenize_segments(aligned_hypothesis)

    return aligned_hypothesis
