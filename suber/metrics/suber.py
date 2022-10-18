import string
from typing import List

from suber.data_types import Subtitle, TimedWord, LineBreak
from suber.constants import END_OF_BLOCK_SYMBOL, END_OF_LINE_SYMBOL
from suber.metrics import lib_ter
from suber.metrics.suber_statistics import SubERStatisticsCollector

from sacrebleu.tokenizers.tokenizer_ter import TercomTokenizer  # only used for "SubER-cased"


def calculate_SubER(hypothesis: List[Subtitle], reference: List[Subtitle], metric="SubER",
                    statistics_collector: SubERStatisticsCollector = None) -> float:
    """
    Main function to caculate the SubER score. It is computed on normalized text, which means case-insensitive and
    without taking punctuation into account, as we observed higher correlation with human judgements and post-edit
    effort in this setting. You can set the 'metric' parameter to "SubER-cased" to calculate a score on cased and
    punctuated text nevertheless. In this case punctuation will be treated as separate words by using a tokenizer.
    We use a modified version of 'lib_ter.py' from sacrebleu for the underlying TER implementation. We altered the
    algorithm by adding a time-overlap condition for word alignments and by disallowing word alignments between real
    words and break tokens.
    """
    assert metric in ["SubER", "SubER-cased"]
    normalize = (metric == "SubER")

    total_num_edits = 0
    total_reference_length = 0

    for part in _get_independent_parts(hypothesis, reference):
        hypothesis_part, reference_part = part

        num_edits, reference_length = _calculate_num_edits_for_part(
            hypothesis_part, reference_part, normalize=normalize, statistics_collector=statistics_collector)

        total_num_edits += num_edits
        total_reference_length += reference_length

    if total_reference_length:
        SubER_score = (total_num_edits / total_reference_length) * 100

    elif not total_num_edits:
        SubER_score = 0.0
    else:
        SubER_score = 100.0

    return round(SubER_score, 3)


def _calculate_num_edits_for_part(hypothesis_part: List[Subtitle], reference_part: List[Subtitle], normalize=True,
                                  statistics_collector: SubERStatisticsCollector = None):
    """
    Returns number of edits (word or break edits and shifts) and the total number of reference tokens (words + breaks)
    for the current part.
    """
    all_hypothesis_words = [word for segment in hypothesis_part for word in segment.word_list]
    all_reference_words = [word for segment in reference_part for word in segment.word_list]

    if normalize:
        # Although casing and punctuation are important aspects of subtitle quality, we observe higher correlation with
        # human post edit effort when normalizing the words.
        all_hypothesis_words = _normalize_words(all_hypothesis_words)
        all_reference_words = _normalize_words(all_reference_words)
    else:
        # When not normalizing punctuation symbols are kept. We treat them as separate tokens by splitting them off
        # the words using sacrebleu's TercomTokenizer.
        all_hypothesis_words = _tokenize_words(all_hypothesis_words)
        all_reference_words = _tokenize_words(all_reference_words)

    all_hypothesis_words = _add_breaks_as_words(all_hypothesis_words)
    all_reference_words = _add_breaks_as_words(all_reference_words)

    num_edits, reference_length = lib_ter.translation_edit_rate(
        all_hypothesis_words, all_reference_words, statistics_collector)

    assert reference_length == len(all_reference_words)

    return num_edits, reference_length


def _add_breaks_as_words(words: List[TimedWord]) -> List[TimedWord]:
    """
    Converts breaks from being an attribute of the previous Word to being a separate Word in the list. Needed because
    TER algorithm should handle breaks as normal tokens.
    """
    output_words = []
    for word in words:
        output_words.append(
            TimedWord(
                string=word.string,
                line_break=LineBreak.NONE,
                subtitle_start_time=word.subtitle_start_time,
                subtitle_end_time=word.subtitle_end_time,
                approximate_word_time=word.approximate_word_time))

        if word.line_break is not LineBreak.NONE:
            output_words.append(
                TimedWord(
                    string=END_OF_LINE_SYMBOL if word.line_break is LineBreak.END_OF_LINE else END_OF_BLOCK_SYMBOL,
                    line_break=LineBreak.NONE,
                    subtitle_start_time=word.subtitle_start_time,
                    subtitle_end_time=word.subtitle_end_time,
                    approximate_word_time=word.approximate_word_time))

    return output_words


remove_punctuation_table = str.maketrans('', '', string.punctuation)


def _normalize_words(words: List[TimedWord]) -> List[TimedWord]:
    """
    Lower-cases Words and removes punctuation.
    """
    output_words = []
    for word in words:
        normalized_string = word.string.lower()
        normalized_string_without_punctuation = normalized_string.translate(remove_punctuation_table)
        normalized_string_without_punctuation = normalized_string_without_punctuation.replace('…', '')

        if normalized_string_without_punctuation:  # keep tokens that are purely punctuation
            normalized_string = normalized_string_without_punctuation

        output_words.append(
            TimedWord(
                string=normalized_string,
                line_break=word.line_break,
                subtitle_start_time=word.subtitle_start_time,
                subtitle_end_time=word.subtitle_end_time,
                approximate_word_time=word.approximate_word_time))

    return output_words


_tokenizer = None  # created if needed in _tokenize_words(), has to be cached...


def _tokenize_words(words: List[TimedWord]) -> List[TimedWord]:
    """
    Not used for the main SubER metric, only for the "SubER-cased" variant. Applies sacrebleu's TercomTokenizer to all
    words in the input, which will create a new list of words containing punctuation symbols as separate elements.
    """
    global _tokenizer
    if not _tokenizer:
        _tokenizer = TercomTokenizer(normalized=True, no_punct=False, case_sensitive=True)

    output_words = []
    for word in words:
        tokenized_word_string = _tokenizer(word.string)
        tokens = tokenized_word_string.split()

        if len(tokens) == 1:
            assert tokenized_word_string == word.string
            output_words.append(word)
            continue

        for token_index, token in enumerate(tokens):
            output_words.append(
                TimedWord(
                    string=token,
                    # Keep line break after the original token, no line breaks within the original token.
                    line_break=word.line_break if token_index == len(tokens) - 1 else LineBreak.NONE,
                    subtitle_start_time=word.subtitle_start_time,
                    subtitle_end_time=word.subtitle_end_time,
                    approximate_word_time=word.approximate_word_time))

    return output_words


def _get_independent_parts(hypothesis: List[Subtitle], reference: List[Subtitle]):
    """
    SubER by definition does not require parallel hypothesis-reference segments. We nevertheless split the subtitle file
    content into parts at positions in time where there is no subtitle in both hypothesis and reference. This makes
    calculation more efficient as Levenshtein distances are computed on shorter sequences, while not changing the
    metric score.

    Note, that in the worst case there are no such split points. In practice, this is unrealistic and subtitle files
    are usually limited to a few hours of speech, such that the current SubER calculation should be efficient enough.

    This function yields Tuple[List[Subtitle],List[Subtitle]] containing the hypothesis and reference subtitles for each
    part.
    """
    hypothesis_part = []
    reference_part = []

    # We sweep the time axis from low to high and handle hypothesis and reference subtitles as soon as we reach them.
    hypothesis_subtitle_index = 0  # index of hypothesis subtitle to handle next
    reference_subtitle_index = 0  # index of reference subtitle to handle next
    latest_observed_time = - float('inf')  # highest time observed so far (end time of a previously handled subtitle)

    while hypothesis_subtitle_index < len(hypothesis) or reference_subtitle_index < len(reference):
        if (hypothesis_subtitle_index < len(hypothesis) and (
                reference_subtitle_index == len(reference) or
                hypothesis[hypothesis_subtitle_index].start_time < reference[reference_subtitle_index].start_time)):
            # We found the next subtitle on the time axis, it is from the hypothesis.

            if ((hypothesis_part or reference_part)
                    and hypothesis[hypothesis_subtitle_index].start_time >= latest_observed_time):
                # The subtitle starts after the latest observed time, meaning there is a gap where no subtitle exists.
                # This concludes the current part, yield it.
                yield (hypothesis_part, reference_part)
                hypothesis_part, reference_part = [], []

            hypothesis_part.append(hypothesis[hypothesis_subtitle_index])
            latest_observed_time = max(latest_observed_time, hypothesis[hypothesis_subtitle_index].end_time)
            hypothesis_subtitle_index += 1

        else:  # Next subtitle to handle is from the reference.
            if ((hypothesis_part or reference_part)
                    and reference[reference_subtitle_index].start_time >= latest_observed_time):
                # The subtitle starts after the latest observed time, meaning there is a gap where no subtitle exists.
                # This concludes the current part, yield it.
                yield (hypothesis_part, reference_part)
                hypothesis_part, reference_part = [], []

            reference_part.append(reference[reference_subtitle_index])
            latest_observed_time = max(latest_observed_time, reference[reference_subtitle_index].end_time)
            reference_subtitle_index += 1

    assert hypothesis_subtitle_index == len(hypothesis) and reference_subtitle_index == len(reference)
    if hypothesis_part or reference_part:
        yield (hypothesis_part, reference_part)
