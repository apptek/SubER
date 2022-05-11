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


if __name__ == '__main__':
    unittest.main()
