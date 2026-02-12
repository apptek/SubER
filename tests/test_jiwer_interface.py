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

    def test_wer_chinese(self):
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            这是一个简单的第一帧

            2
            00:00:01,000 --> 00:00:02,000
            这是另一个有
            两条线的帧"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            这是一个简单的第一帧。

            2
            00:00:01,000 --> 00:00:02,000
            这是另一个
            有两条线的帧。"""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        wer_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER", language="zh")

        self.assertAlmostEqual(wer_score, 0.0)

        wer_cased_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-cased", language="zh")

        # TokenizerZh expected to split all characters.
        # 2 punctuation errors / 21 tokenized characters
        self.assertAlmostEqual(wer_cased_score, 9.524)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg", language="zh")

        # (1 break deletion + 1 break insertion) / (21 tokenized characters + 3 breaks)
        self.assertAlmostEqual(wer_seg_score, 8.333)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg",
            score_break_at_segment_end=False, language="zh")

        # (1 break deletion + 1 break insertion) / (21 tokenized characters + 1 breaks)
        self.assertAlmostEqual(wer_seg_score, 9.091)

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

        # TokenizerJaMecab expected to tokenize into this:
        # "これ は 簡単 な 最初 の ブロック です"
        # "これ は 二つ の 行 を 持つ 別 の ブロック です"

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        wer_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER", language="ja")

        self.assertAlmostEqual(wer_score, 0.0)

        wer_cased_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-cased", language="ja")

        # 2 punctuation errors / 19 tokenized words
        self.assertAlmostEqual(wer_cased_score, 10.526)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg", language="ja")

        # (1 break deletion + 1 break insertion) / (19 tokenized words + 3 breaks)
        self.assertAlmostEqual(wer_seg_score, 9.091)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg",
            score_break_at_segment_end=False, language="ja")

        # (1 break deletion + 1 break insertion) / (19 tokenized words + 1 breaks)
        self.assertAlmostEqual(wer_seg_score, 10.0)

    def test_wer_korean(self):
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            이것은 간단한 첫 번째 프레임입니다

            2
            00:00:01,000 --> 00:00:02,000
            이것은 두 줄로 이루어진 또
            다른 프레임입니다"""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            이것은 간단한 첫 번째 프레임입니다.

            2
            00:00:01,000 --> 00:00:02,000
            이것은 두 줄로 이루어진
            또 다른 프레임입니다."""

        # TokenizerKoMecab expected to tokenize into this:
        # "이것 은 간단 한 첫 번 째 프레임 입니다"
        # "이것 은 두 줄 로 이루어진 또 다른 프레임 입니다"

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        wer_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER", language="ko")

        self.assertAlmostEqual(wer_score, 0.0)

        wer_cased_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-cased", language="ko")

        # 2 punctuation errors / 19 tokenized words
        self.assertAlmostEqual(wer_cased_score, 10.526)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg", language="ko")

        # (1 break deletion + 1 break insertion) / (19 tokenized characters + 3 breaks)
        self.assertAlmostEqual(wer_seg_score, 9.091)

        wer_seg_score = calculate_word_error_rate(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="WER-seg",
            score_break_at_segment_end=False, language="ko")

        # (1 break deletion + 1 break insertion) / (19 tokenized characters + 1 breaks)
        self.assertAlmostEqual(wer_seg_score, 10.0)


if __name__ == '__main__':
    unittest.main()
