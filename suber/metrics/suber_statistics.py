from typing import Dict, List, Any
from collections import OrderedDict

from suber.data_types import Word
from suber.constants import END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL


class SubERStatisticsCollector:
    """
    Collects number of different SubER edit operations necessary to turn the reference into the hypothesis.
    (The TER code and paper uses the hypothesis to reference direction, but calling a word occurring only in the
    hypothesis an insertion and a word missing in the hypothesis a deletion seems to be far more common.)
    """

    def __init__(self):
        self._num_reference_words = 0
        self._num_reference_breaks = 0
        self._num_shifts = 0
        self._num_word_deletions = 0
        self._num_break_deletions = 0
        self._num_word_insertions = 0
        self._num_break_insertions = 0
        self._num_word_substitutions = 0
        self._num_break_substitutions = 0

    def add_data(self, trace: str, words_ref: List[Word], words_hyp_shifted: List[Word], num_shifts: int):
        """
        Called inside lib_ter.translation_edit_rate(). 'trace' contains characters 'i', 'd', 's' and ' ' representing
        different edit operations.
        """
        reference_position = -1
        hypothesis_position = -1

        for edit_operation in trace:
            if edit_operation != "i":
                reference_position += 1

            if edit_operation != "d":
                hypothesis_position += 1

            if edit_operation == "i":
                if words_hyp_shifted[hypothesis_position].string in [END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL]:
                    self._num_break_insertions += 1
                else:
                    self._num_word_insertions += 1
            else:
                is_break_edit = words_ref[reference_position].string in [END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL]

                if is_break_edit:
                    self._num_reference_breaks += 1
                else:
                    self._num_reference_words += 1

                if edit_operation == "d":
                    if is_break_edit:
                        self._num_break_deletions += 1
                    else:
                        self._num_word_deletions += 1

                elif edit_operation == "s":
                    if is_break_edit:
                        self._num_break_substitutions += 1
                    else:
                        self._num_word_substitutions += 1

        assert reference_position == len(words_ref) - 1
        assert hypothesis_position == len(words_hyp_shifted) - 1

        self._num_shifts += num_shifts

    def get_statistics(self) -> Dict[str, Any]:
        return OrderedDict(
            num_reference_words=self._num_reference_words,
            num_reference_breaks=self._num_reference_breaks,
            num_shifts=self._num_shifts,
            num_word_deletions=self._num_word_deletions,
            num_break_deletions=self._num_break_deletions,
            num_word_insertions=self._num_word_insertions,
            num_break_insertions=self._num_break_insertions,
            num_word_substitutions=self._num_word_substitutions,
            num_break_substitutions=self._num_break_substitutions,
        )
