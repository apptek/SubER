import unittest

from suber.metrics.jiwer_interface import calculate_word_error_rate
from .utilities import create_temporary_file_and_read_it


class JiWERInterfaceTest(unittest.TestCase):

    def test_wer(self):
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

        wer_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER")

        self.assertAlmostEqual(wer_score, 0.0)

        wer_cased_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-cased")

        # 2 substitutions (casing and punctuation error) / 15 tokenized words
        self.assertAlmostEqual(wer_cased_score, 13.333)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg")

        # (1 break deletion + 1 break insertion) / (13 words + 3 breaks)
        self.assertAlmostEqual(wer_seg_score, 12.5)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg",
            score_break_at_segment_end=False)

        # (1 break deletion + 1 break insertion) / (13 words + 1 breaks)
        self.assertAlmostEqual(wer_seg_score, 14.286)


    def test_wer_japanese(self):
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            これは簡単な最初のブロックです

            2
            00:00:01,000 --> 00:00:02,000
            これは二つの行を持つ
            別のブロックです"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            これは簡単な最初のブロックです。

            2
            00:00:01,000 --> 00:00:02,000
            これは二つの行を
            持つ別のブロックです。"""

        # TercomTokenizer(normalized=True, asian_support=True) used for TER expected to tokenize into this:
        # "これは 簡 単 な 最 初 のブロックです"
        # "これは 二 つの 行 を 持 つ 別 のブロックです"

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        wer_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER", language="ja")

        self.assertAlmostEqual(wer_score, 0.0)

        wer_cased_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-cased", language="ja")

        # 2 punctuation errors / 16 tokenized words
        self.assertAlmostEqual(wer_cased_score, 12.5)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg", language="ja")

        # (1 break deletion + 1 break insertion) / (16 tokenized words + 3 breaks)
        self.assertAlmostEqual(wer_seg_score, 10.526)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg",
            score_break_at_segment_end=False, language="ja")

        # (1 break deletion + 1 break insertion) / (16 tokenized words + 1 breaks)
        self.assertAlmostEqual(wer_seg_score, 11.765)



if __name__ == '__main__':
    unittest.main()
