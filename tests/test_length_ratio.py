import unittest

from suber.metrics.length_ratio import calculate_length_ratio
from .utilities import create_temporary_file_and_read_it


class LengthRatioTest(unittest.TestCase):
    def setUp(self):
        # Punctuation marks should count as separate tokens.
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            One two three.

            2
            00:00:01,000 --> 00:00:02,000
            Five six
            seven eight?"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            One two.

            2
            00:00:01,000 --> 00:00:01,500
            Four five

            3
            00:00:01,500 --> 00:00:02,000
            six?"""

        self._reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        self._hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

    def test_length_ratio(self):
        length_ratio = calculate_length_ratio(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles)

        self.assertAlmostEqual(length_ratio, 7 / 9 * 100, places=3)
