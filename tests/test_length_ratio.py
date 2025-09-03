import unittest

from suber.metrics.length_ratio import calculate_length_ratio
from .utilities import create_temporary_file_and_read_it


class LengthRatioTest(unittest.TestCase):
    def test_length_ratio(self):
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

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        length_ratio = calculate_length_ratio(hypothesis=hypothesis_subtitles, reference=reference_subtitles)

        self.assertAlmostEqual(length_ratio, 7 / 9 * 100, places=3)

    def test_length_ratio_japanese(self):
        # TODO: Not sure what to expect here, some numbers are split into characters, others not. (Without commas it
        # looks even less consistent to me.) Need language expertise. :D
        # Could also test kanji, but then it would be the same as Chinese, i.e. characters?
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            いち、に、さん。

            2
            00:00:01,000 --> 00:00:02,000
            ご、ろく
            しち、はち？"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            いち、に。

            2
            00:00:01,000 --> 00:00:02,000
            し、ご

            3
            00:00:01,500 --> 00:00:02,000
            ろく？"""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        length_ratio = calculate_length_ratio(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, language="ja")

        self.assertAlmostEqual(length_ratio, 10 / 16 * 100, places=3)

    def test_length_ratio_chinese(self):
        # Should be split into characters, including punctuation, except for the English words, which are handled
        # separately by the tokenizer.
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            一二三。

            2
            00:00:01,000 --> 00:00:02,000
            五六
            七八？Plus three!"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            一二。

            2
            00:00:01,000 --> 00:00:02,000
            四五

            3
            00:00:01,500 --> 00:00:02,000
            六？
            Plus three!"""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        length_ratio = calculate_length_ratio(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, language="zh")

        # As in English test_length_ratio(), but both reference and hypothesis "plus three" tokens.
        self.assertAlmostEqual(length_ratio, 10 / 12 * 100, places=3)

    def test_length_ratio_korean(self):
        # Tokenizer expected to split into numbers.
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            하나둘셋.

            2
            00:00:01,000 --> 00:00:02,000
            다섯여섯
            일곱여덟?"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            하나둘.

            2
            00:00:01,000 --> 00:00:02,000
            넷다섯

            3
            00:00:01,500 --> 00:00:02,000
            여섯?"""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        length_ratio = calculate_length_ratio(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, language="ko")

        # As in English test_length_ratio().
        self.assertAlmostEqual(length_ratio, 7 / 9 * 100, places=3)


if __name__ == '__main__':
    unittest.main()
