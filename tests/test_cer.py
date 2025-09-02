import unittest

from suber.metrics.cer import calculate_character_error_rate
from .utilities import create_temporary_file_and_read_it


class CERTest(unittest.TestCase):

    def test_cer(self):
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            This is a simple first frame.

            2
            00:00:01,000 --> 00:00:02,000
            This is another frame
            having two lines."""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            This is a simple first frame,

            2
            00:00:01,000 --> 00:00:02,000
            this is another
            frame having two lines."""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        cer_score = calculate_character_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="CER")

        # Lower-case and without punctuation by default, so no edits.
        self.assertAlmostEqual(cer_score, 0.0)

        cer_cased_score = calculate_character_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="CER-cased")

        # 2 edits / 68 characters
        self.assertAlmostEqual(cer_cased_score, 2.941)

    def test_cer_japanese(self):
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            これは簡単な最初のブロックです

            2
            00:00:01,000 --> 00:00:02,000
            これは二つの行を持つ
            別のブロックです。"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            「これは簡単な最初のブロックです」

            2
            00:00:01,000 --> 00:00:02,000
            これは二つの行を
            持つ別のブロックです"""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        cer_score = calculate_character_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="CER")

        # TODO: this is not 0 because a space is added at line breaks and thus the second blocks differ in the position
        # of this space. We might want to not add such a space for Japanese?
        # 1 space insertion, 1 space deletion, 34 reference characters (including the spaces).
        self.assertAlmostEqual(cer_score, 5.882)

        cer_cased_score = calculate_character_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="CER-cased")

        # 3 punctuation character errors, 2 space edits as above, now 35 reference characters.
        self.assertAlmostEqual(cer_cased_score, 14.286)


if __name__ == '__main__':
    unittest.main()
