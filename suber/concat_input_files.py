from typing import List, Tuple

from suber.file_readers import read_input_file
from suber.data_types import Segment, Subtitle


def create_concatenated_segments(hypothesis_files: List[str], reference_files: List[str], hypothesis_format="SRT",
                                 reference_format="SRT") -> Tuple[List[Segment], List[Segment]]:
    """
    Reads all pairs of hypothesis and reference files and creates two concatenated lists containing all hypothesis
    segments and all reference segments, respectively. This can be used to score test corpora available in form of many
    smaller audio / video files without concatenating the files manually.
    In case of SRT input the segments are subtitles with timing information. We adjust the subtitle timings such that
    all files are placed one after the other on the time axis, which corresponds to concatenating the corresponding
    audio / video files.
    """
    if len(hypothesis_files) != len(reference_files):
        raise ValueError("Number of hypothesis and reference files must match.")

    all_hypothesis_segments = []
    all_reference_segments = []

    seconds_to_shift = 0.0
    total_hypothesis_duration = 0.0
    total_reference_duration = 0.0

    for hypothesis_file, reference_file in zip(hypothesis_files, reference_files):
        hypothesis_segments = read_input_file(hypothesis_file, file_format=hypothesis_format)
        reference_segments = read_input_file(reference_file, file_format=reference_format)

        if hypothesis_segments and isinstance(hypothesis_segments[0], Subtitle):
            total_hypothesis_duration = _shift_subtitles_in_time(hypothesis_segments, seconds_to_shift)
        if reference_segments and isinstance(reference_segments[0], Subtitle):
            total_reference_duration = _shift_subtitles_in_time(reference_segments, seconds_to_shift)

        seconds_to_shift = max(total_hypothesis_duration, total_reference_duration)

        all_hypothesis_segments += hypothesis_segments
        all_reference_segments += reference_segments

    return all_hypothesis_segments, all_reference_segments


def _shift_subtitles_in_time(subtitles: List[Subtitle], seconds) -> float:
    """
    Returns new total duration after shift.
    """

    for subtitle in subtitles:
        _shift_subtitle_in_time(subtitle, seconds)

    # There might be audio / video left after the last subtitle end time, but taking this into account is not necessary
    # for metric calculation.
    # We add an epsilon to make sure that subtitles from different files are not counted as overlapping.
    return subtitles[-1].end_time + 1e-8


def _shift_subtitle_in_time(subtitle: Subtitle, seconds):
    subtitle.start_time += seconds
    subtitle.end_time += seconds
    for word in subtitle.word_list:
        word.approximate_word_time += seconds
