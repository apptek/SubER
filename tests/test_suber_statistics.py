import unittest

from typing import Dict, Any

from suber.metrics.suber import calculate_SubER
from suber.metrics.suber_statistics import SubERStatisticsCollector
from .utilities import create_temporary_file_and_read_it


class SubERStatisticsTests(unittest.TestCase):
    """
    Mostly copied from SubERMetricsTests, but now testing the statistics collection.
    """

    def setUp(self):
        self._reference1 = """
            1
            0:00:01.000 --> 0:00:02.000
            This is a subtitle."""

        self._reference2 = """
            1
            0:00:01.000 --> 0:00:02.000
            This is a subtitle.

            2
            0:00:03.000 --> 0:00:04.000
            And another one!"""

        self._statistics_template = {
        }

    def _run_test(self, hypothesis, reference, expected_statistics: Dict[str, Any]):
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis)
        reference_subtitles = create_temporary_file_and_read_it(reference)

        statistics_collector = SubERStatisticsCollector()

        _ = calculate_SubER(
            hypothesis_subtitles,
            reference_subtitles,
            statistics_collector=statistics_collector)

        statistics = statistics_collector.get_statistics()

        for key, value in expected_statistics.items():
            self.assertIn(key, statistics)
            self.assertEqual(value, statistics[key], msg=f"key: {key}")

    def test_empty(self):
        expected_statistics = {
            "num_reference_words": 0,
            "num_reference_breaks": 0,
            "num_shifts": 0,
            "num_word_deletions": 0,
            "num_break_deletions": 0,
            "num_word_insertions": 0,
            "num_break_insertions": 0,
            "num_word_substitutions": 0,
            "num_break_substitutions": 0,
        }

        self._run_test(hypothesis="", reference="", expected_statistics=expected_statistics)

    def test_split_subtitle_no_overlap(self):
        hypothesis = """
            1
            0:00:00.000 --> 0:00:00.500
            This is

            2
            0:00:02.500 --> 0:00:03.000
            a subtitle."""

        expected_statistics = {
            "num_reference_words": 4,
            "num_reference_breaks": 1,
            "num_shifts": 0,
            "num_word_deletions": 4,
            "num_break_deletions": 1,
            "num_word_insertions": 4,
            "num_break_insertions": 2,
            "num_word_substitutions": 0,
            "num_break_substitutions": 0,
        }

        self._run_test(hypothesis, self._reference1, expected_statistics=expected_statistics)

    def test_split_subtitle_one_overlap(self):
        hypothesis = """
            1
            0:00:00.000 --> 0:00:00.500
            This is

            2
            0:00:01.500 --> 0:00:02.000
            a subtitle."""

        expected_statistics = {
            "num_reference_words": 4,
            "num_reference_breaks": 1,
            "num_shifts": 0,
            "num_word_deletions": 2,
            "num_break_deletions": 0,
            "num_word_insertions": 2,
            "num_break_insertions": 1,
            "num_word_substitutions": 0,
            "num_break_substitutions": 0,
        }

        self._run_test(hypothesis, self._reference1, expected_statistics=expected_statistics)

    def test_merged_subtitle_with_shift_and_substitution(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:04.000
            That is a another one! subtitle.
            And"""

        expected_statistics = {
            "num_reference_words": 7,
            "num_reference_breaks": 2,
            "num_shifts": 1,
            "num_word_deletions": 0,
            "num_break_deletions": 0,
            "num_word_insertions": 0,
            "num_break_insertions": 0,
            "num_word_substitutions": 1,
            "num_break_substitutions": 1,
        }
        self._run_test(hypothesis, self._reference2, expected_statistics=expected_statistics)

    def test_split_into_three_with_one_shift(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:01.500
            This is a

            2
            0:00:01.500 --> 0:00:03.500
            another
            subtitle.

            2
            0:00:03.500 --> 0:00:04.000
            And one!"""

        expected_statistics = {
            "num_reference_words": 7,
            "num_reference_breaks": 2,
            "num_shifts": 1,
            "num_word_deletions": 0,
            "num_break_deletions": 0,
            "num_word_insertions": 0,
            "num_break_insertions": 2,
            "num_word_substitutions": 0,
            "num_break_substitutions": 0,
        }

        self._run_test(hypothesis, self._reference2, expected_statistics=expected_statistics)


if __name__ == '__main__':
    unittest.main()
