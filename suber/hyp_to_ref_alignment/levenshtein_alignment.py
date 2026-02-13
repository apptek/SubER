import numpy
import regex
import string
from itertools import zip_longest
from typing import List, Optional, Tuple

from suber import lib_levenshtein
from suber.constants import EAST_ASIAN_LANGUAGE_CODES, SPACE_ESCAPE
from suber.data_types import Segment
from suber.tokenizers import reversibly_tokenize_segments, detokenize_segments


def levenshtein_align_hypothesis_to_reference(
        hypothesis: List[Segment], reference: List[Segment], language: Optional[str] = None) -> List[Segment]:
    """
    Runs the Levenshtein algorithm to get the minimal set of edit operations to convert the full list of hypothesis
    words into the full list of reference words. The edit operations implicitly define an alignment between hypothesis
    and reference words. Using this alignment, the hypotheses are re-segmented to match the reference segmentation.
    """

    if language in EAST_ASIAN_LANGUAGE_CODES:
        # Punctuation kept attached because we want to remove it below to normalize the tokens before alignment, but
        # there we cannot change the number of tokens (and must not create empty tokens).
        hypothesis = reversibly_tokenize_segments(hypothesis, language, keep_punctuation_attached=True)
        reference = reversibly_tokenize_segments(reference, language, keep_punctuation_attached=True)

    remove_punctuation_table = str.maketrans('', '', string.punctuation)

    def normalize_word(word):
        """
        Lower-cases and removes punctuation as this increases the alignment accuracy.
        """
        word = word.lower()

        if language in EAST_ASIAN_LANGUAGE_CODES:
            # Space escape needed for detokenization, but we don't want it to influence the alignment.
            if word.startswith(SPACE_ESCAPE):
                word = word[1:]
                assert word, "Word should not be only space escape character."
            word_without_punctuation = regex.sub(r"\p{P}", "", word)
        else:
            # Backwards compatibility: keep old behavior for other languages, even though removing non-ASCII punctuation
            # would also make sense here.
            word_without_punctuation = word.translate(remove_punctuation_table)

        if not word_without_punctuation:
            return word  # keep tokens that are purely punctuation

        return word_without_punctuation

    all_reference_word_strings = [normalize_word(word.string) for segment in reference for word in segment.word_list]
    all_hypothesis_word_strings = [normalize_word(word.string) for segment in hypothesis for word in segment.word_list]

    all_hypothesis_words = [word for segment in hypothesis for word in segment.word_list]

    reference_string, hypothesis_string = _map_words_to_characters(
        all_reference_word_strings, all_hypothesis_word_strings)

    opcodes = lib_levenshtein.opcodes(reference_string, hypothesis_string)

    reference_segment_lengths = [len(segment.word_list) for segment in reference]
    reference_segment_boundary_indices = numpy.cumsum(reference_segment_lengths)
    current_segment_index = 0
    aligned_hypothesis_word_lists = [[] for _ in reference]

    for opcode_tuple in opcodes:
        edit_operation = opcode_tuple[0]
        hypothesis_position_range = range(opcode_tuple[3], opcode_tuple[4])
        reference_position_range = range(opcode_tuple[1], opcode_tuple[2])

        if edit_operation in ("equal", "replace"):
            assert len(hypothesis_position_range) == len(reference_position_range)
        elif edit_operation == "insert":
            assert len(reference_position_range) == 0
        elif edit_operation == "delete":
            assert len(hypothesis_position_range) == 0
        else:
            assert False, f"Invalid edit operation '{edit_operation}'."

        # 'zip_longest' is a "clever" way to unify the different cases: for 'equal' and 'replace' we indeed have to
        # iterate through hypothesis and reference position in parallel, for 'insert' and 'delete' either
        # 'hypothesis_position' or 'reference_position' will be None in the loop.
        for hypothesis_position, reference_position in zip_longest(hypothesis_position_range, reference_position_range):

            # Update current segment index depending on current reference position.
            if (reference_position is not None
                    and reference_position >= reference_segment_boundary_indices[current_segment_index]):

                assert reference_position == reference_segment_boundary_indices[current_segment_index], (
                    "Bug: missing reference position in edit operations.")
                current_segment_index += 1

                # If there are empty segments in the reference, we get double entries in
                # 'reference_segment_boundary_indices' (because the empty segment ends at the same word index as the
                # previous segment). Skip these empty segments, we don't want to assign any hypothesis words to them.
                while (current_segment_index < len(reference_segment_boundary_indices)
                       and reference_segment_boundary_indices[current_segment_index]
                       == reference_segment_boundary_indices[current_segment_index - 1]):
                    current_segment_index += 1

            # Add hypothesis word to current segment in case of 'equal', 'replace' or 'insert' operation.
            if hypothesis_position is not None:
                word = all_hypothesis_words[hypothesis_position]
                aligned_hypothesis_word_lists[current_segment_index].append(word)

    aligned_hypothesis = [Segment(word_list=word_list) for word_list in aligned_hypothesis_word_lists]

    if language in EAST_ASIAN_LANGUAGE_CODES:
        aligned_hypothesis = detokenize_segments(aligned_hypothesis)

    return aligned_hypothesis


def _map_words_to_characters(reference_words: List[str], hypothesis_words: List[str]) -> Tuple[str, str]:
    """
    The Levenshtein module operates on strings, not list of strings. Therefore we map words to characters here.
    Inspired by https://github.com/jitsi/jiwer/blob/master/jiwer/measures.py.
    """
    unique_words = set(reference_words + hypothesis_words)
    vocabulary = dict(zip(unique_words, range(len(unique_words))))

    reference_string = "".join(chr(vocabulary[word] + 32) for word in reference_words)
    hypothesis_string = "".join(chr(vocabulary[word] + 32) for word in hypothesis_words)

    return reference_string, hypothesis_string
