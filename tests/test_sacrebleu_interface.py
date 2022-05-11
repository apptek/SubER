import unittest

from suber.metrics.sacrebleu_interface import calculate_sacrebleu_metric
from .utilities import create_temporary_file_and_read_it


class SacreBleuInterfaceTest(unittest.TestCase):
    def setUp(self):
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
            This is a simple first frame.

            2
            00:00:01,000 --> 00:00:02,000
            This is another
            frame having two lines."""

        self._reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        self._hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

    def test_bleu(self):
        bleu_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="BLEU")

        self.assertAlmostEqual(bleu_score, 100.0)

        bleu_seg_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="BLEU-seg")

        self.assertAlmostEqual(bleu_seg_score, 76.279)

        bleu_seg_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="BLEU-seg",
            score_break_at_segment_end=False)

        self.assertAlmostEqual(bleu_seg_score, 71.538)

    def test_TER(self):
        ter_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="TER")

        self.assertAlmostEqual(ter_score, 0.0)

        ter_seg_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="TER-seg")

        # 1 break shift / (13 words + 3 breaks)
        self.assertAlmostEqual(ter_seg_score, 6.25)

        ter_seg_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="TER-seg",
            score_break_at_segment_end=False)

        # 1 break shift / (13 words + 1 breaks)
        self.assertAlmostEqual(ter_seg_score, 7.143)

    def test_TER_br(self):
        reference_file_content = "This is one sentence <eol> with a line break <eob> and a frame break. <eob>"
        hypothesis_file_content = "This <eol> is a sentence with <eol> a line <eob> break and a block break. <eob>"

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content, file_format="plain")
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content, file_format="plain")

        ter_br_score = calculate_sacrebleu_metric(
            hypothesis=hypothesis_subtitles, reference=reference_subtitles, metric="TER-br")

        # 12 real words, 3 line breaks, so reference length is 15. Edit operations should be 2 shifts and 1 insertion
        # of break symbols.
        expected_ter_br_score = round(3 / (12 + 3) * 100, 3)
        self.assertAlmostEqual(ter_br_score, expected_ter_br_score)

    def test_chrF(self):
        chrF_score = calculate_sacrebleu_metric(
            hypothesis=self._hypothesis_subtitles, reference=self._reference_subtitles, metric="chrF")

        self.assertAlmostEqual(chrF_score, 100.0)


if __name__ == '__main__':
    unittest.main()
