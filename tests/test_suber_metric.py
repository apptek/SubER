import unittest

from suber.data_types import Subtitle
from suber.metrics.suber import calculate_SubER, _get_independent_parts
from .utilities import create_temporary_file_and_read_it


class SubERMetricTests(unittest.TestCase):
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

    def _run_test(self, hypothesis, reference, expected_score):
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis)
        reference_subtitles = create_temporary_file_and_read_it(reference)

        SubER_score = calculate_SubER(hypothesis_subtitles, reference_subtitles)

        self.assertAlmostEqual(SubER_score, expected_score)

    def test_empty(self):
        self._run_test(hypothesis="", reference="", expected_score=0.0)
        self._run_test(hypothesis="", reference=self._reference1, expected_score=100.0)
        self._run_test(hypothesis=self._reference1, reference="", expected_score=100.0)

    def test_identical(self):
        self._run_test(hypothesis=self._reference1, reference=self._reference1, expected_score=0.0)
        self._run_test(hypothesis=self._reference2, reference=self._reference2, expected_score=0.0)

    def test_one_shift(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:02.000
            a subtitle. This is"""
        # 1 shift / (4 words + 1 break)
        self._run_test(hypothesis, self._reference1, expected_score=20.0)

    def test_no_overlap(self):
        hypothesis = """
            1
            0:00:00.000 --> 0:00:01.000
            This is a subtitle."""
        # All words + breaks count as deletion + insertion.
        self._run_test(hypothesis, self._reference1, expected_score=200.0)

    def test_with_overlap(self):
        hypothesis = """
            1
            0:00:00.500 --> 0:00:01.500
            This is a subtitle."""
        self._run_test(hypothesis, self._reference1, expected_score=0.0)

    def test_split_subtitle(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:01.500
            This is

            2
            0:00:01.500 --> 0:00:02.000
            a subtitle."""
        # 1 break insertion / (4 words + 1 break)
        self._run_test(hypothesis, self._reference1, expected_score=20.0)

    def test_split_subtitle_with_shift(self):
        hypothesis = """
            1
            0:00:00.100 --> 0:00:01.500
            This is

            2
            0:00:01.500 --> 0:00:02.000
            subtitle. a"""
        # (1 shift + 1 break insertion) / (4 words + 1 break)
        self._run_test(hypothesis, self._reference1, expected_score=40.0)

    def test_split_subtitle_no_overlap(self):
        hypothesis = """
            1
            0:00:00.000 --> 0:00:00.500
            This is

            2
            0:00:02.500 --> 0:00:03.000
            a subtitle."""
        # (4 word insertions + 2 break insertions + 4 word deletions + 1 break deletion) / (4 words + 1 break)
        self._run_test(hypothesis, self._reference1, expected_score=220.0)

    def test_split_subtitle_one_overlap(self):
        hypothesis = """
            1
            0:00:00.000 --> 0:00:00.500
            This is

            2
            0:00:01.500 --> 0:00:02.000
            a subtitle."""
        # (2 word insertions + 1 break insertions + 2 word deletions) / (4 words + 1 break)
        self._run_test(hypothesis, self._reference1, expected_score=100.0)

    def test_merged_subtitle(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:04.000
            This is a subtitle.
            And another one!"""
        # 1 break substitution / (7 words + 2 breaks)
        self._run_test(hypothesis, self._reference2, expected_score=11.111)

    def test_merged_subtitle_with_shift(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:04.000
            This is a another one! subtitle.
            And"""
        # (1 shift + 1 break substitution) / (7 words + 2 breaks)
        self._run_test(hypothesis, self._reference2, expected_score=22.222)

    def test_split_into_three(self):
        hypothesis = """
            1
            0:00:01.000 --> 0:00:01.500
            This is a

            2
            0:00:01.500 --> 0:00:03.500
            subtitle.
            And

            2
            0:00:03.500 --> 0:00:04.000
            another one!"""
        # (2 break insertions + 1 break substitution) / (7 words + 2 breaks)
        self._run_test(hypothesis, self._reference2, expected_score=33.333)

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
        # (1 shift + 2 break insertions) / (7 words + 2 breaks)
        self._run_test(hypothesis, self._reference2, expected_score=33.333)


class SubERCasedMetricTests(unittest.TestCase):
    def test_SubER_cased(self):
        reference = """
            1
            0:00:01.000 --> 0:00:02.000
            This is a subtitle.

            2
            0:00:03.000 --> 0:00:04.000
            And another one!"""

        hypothesis = """
            1
            0:00:01.000 --> 0:00:01.500
            This is a

            2
            0:00:01.500 --> 0:00:03.500
            another
            subtitle,

            2
            0:00:03.500 --> 0:00:04.000
            and one!"""

        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis)
        reference_subtitles = create_temporary_file_and_read_it(reference)

        SubER_score = calculate_SubER(hypothesis_subtitles, reference_subtitles, metric="SubER-cased")

        # After tokenization there should be 9 reference words + 2 reference break tokens.
        # 1 shift and 2 break deletions as above for SubER, plus 2 substitutions: "," -> "."; "and" -> "And"
        self.assertAlmostEqual(SubER_score, 45.455)


class SubERHelperFunctionTests(unittest.TestCase):

    def test_get_independent_parts_empty_input(self):
        parts = list(_get_independent_parts(hypothesis=[], reference=[]))
        self.assertFalse(parts)

    def test_get_independent_parts_only_hypothesis(self):
        hypothesis = [
            Subtitle(word_list=[], index=1, start_time=0, end_time=1),
            Subtitle(word_list=[], index=2, start_time=1, end_time=2),
            Subtitle(word_list=[], index=3, start_time=3, end_time=4)]

        parts = list(_get_independent_parts(hypothesis=hypothesis, reference=[]))
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], ([hypothesis[0]], []))
        self.assertEqual(parts[1], ([hypothesis[1]], []))
        self.assertEqual(parts[2], ([hypothesis[2]], []))

    def test_get_independent_parts_only_reference(self):
        reference = [
            Subtitle(word_list=[], index=1, start_time=0, end_time=1),
            Subtitle(word_list=[], index=2, start_time=1, end_time=2),
            Subtitle(word_list=[], index=3, start_time=3, end_time=4)]

        parts = list(_get_independent_parts(hypothesis=[], reference=reference))
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], ([], [reference[0]]))
        self.assertEqual(parts[1], ([], [reference[1]]))
        self.assertEqual(parts[2], ([], [reference[2]]))

    def test_get_independent_parts_all_overlaps(self):
        hypothesis = [
            Subtitle(word_list=[], index=1, start_time=0, end_time=1),
            Subtitle(word_list=[], index=2, start_time=1, end_time=2),
            Subtitle(word_list=[], index=3, start_time=3, end_time=4)]

        parts = list(_get_independent_parts(hypothesis=hypothesis, reference=hypothesis))
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], ([hypothesis[0]], [hypothesis[0]]))
        self.assertEqual(parts[1], ([hypothesis[1]], [hypothesis[1]]))
        self.assertEqual(parts[2], ([hypothesis[2]], [hypothesis[2]]))

    def test_get_independent_parts_overlap_with_one_big(self):
        hypothesis = [
            Subtitle(word_list=[], index=1, start_time=0.25, end_time=1),
            Subtitle(word_list=[], index=2, start_time=1, end_time=2),
            Subtitle(word_list=[], index=3, start_time=3, end_time=4)]

        reference = [
            Subtitle(word_list=[], index=1, start_time=0.5, end_time=3.5)]

        parts = list(_get_independent_parts(hypothesis=hypothesis, reference=reference))
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], (hypothesis, reference))

        reference = [
            Subtitle(word_list=[], index=1, start_time=0, end_time=4.5)]

        parts = list(_get_independent_parts(hypothesis=hypothesis, reference=reference))
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], (hypothesis, reference))

    def test_get_independent_parts(self):
        hypothesis = [
            Subtitle(word_list=[], index=1, start_time=0, end_time=0.25),
            Subtitle(word_list=[], index=2, start_time=0.25, end_time=0.5),
            Subtitle(word_list=[], index=3, start_time=0.5, end_time=1),
            Subtitle(word_list=[], index=4, start_time=1, end_time=1.5),
            Subtitle(word_list=[], index=5, start_time=1.75, end_time=2),
            Subtitle(word_list=[], index=6, start_time=4.1, end_time=4.9),
            Subtitle(word_list=[], index=7, start_time=5.1, end_time=6),
            Subtitle(word_list=[], index=8, start_time=6, end_time=7),
            Subtitle(word_list=[], index=9, start_time=7, end_time=8)]

        reference = [
            Subtitle(word_list=[], index=1, start_time=0.75, end_time=1.1),
            Subtitle(word_list=[], index=2, start_time=1.4, end_time=2.2),
            Subtitle(word_list=[], index=3, start_time=3, end_time=3.5),
            Subtitle(word_list=[], index=4, start_time=3.5, end_time=4),
            Subtitle(word_list=[], index=5, start_time=4, end_time=5),
            Subtitle(word_list=[], index=6, start_time=6, end_time=6.5),
            Subtitle(word_list=[], index=7, start_time=6.5, end_time=7.5),
            Subtitle(word_list=[], index=8, start_time=8, end_time=9)]

        parts = list(_get_independent_parts(hypothesis=hypothesis, reference=reference))
        self.assertEqual(len(parts), 9)
        self.assertEqual(parts[0], (hypothesis[0:1], []))
        self.assertEqual(parts[1], (hypothesis[1:2], []))
        self.assertEqual(parts[2], (hypothesis[2:5], reference[:2]))
        self.assertEqual(parts[3], ([], reference[2:3]))
        self.assertEqual(parts[4], ([], reference[3:4]))
        self.assertEqual(parts[5], (hypothesis[5:6], reference[4:5]))
        self.assertEqual(parts[6], (hypothesis[6:7], []))
        self.assertEqual(parts[7], (hypothesis[7:9], reference[5:7]))
        self.assertEqual(parts[8], ([], reference[7:8]))


if __name__ == '__main__':
    unittest.main()
