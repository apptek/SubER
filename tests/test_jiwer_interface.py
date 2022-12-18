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


if __name__ == '__main__':
    unittest.main()
