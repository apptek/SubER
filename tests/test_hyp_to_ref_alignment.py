import unittest

from suber.hyp_to_ref_alignment import time_align_hypothesis_to_reference
from suber.hyp_to_ref_alignment import levenshtein_align_hypothesis_to_reference
from .utilities import create_temporary_file_and_read_it


class TimeAlignmentTests(unittest.TestCase):

    def test_full_overlap(self):
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
            00:00:01,000 --> 00:00:01,500
            This is another frame

            3
            00:00:01,500 --> 00:00:02,000
            having two lines."""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        hypothesis_subtitles = time_align_hypothesis_to_reference(hypothesis_subtitles, reference_subtitles)

        self.assertEqual(len(hypothesis_subtitles), 2)

        self.assertEqual(hypothesis_subtitles[0].index, 1)
        self.assertEqual(hypothesis_subtitles[1].index, 2)

        self.assertAlmostEqual(hypothesis_subtitles[0].start_time, 0.0)
        self.assertAlmostEqual(hypothesis_subtitles[0].end_time, 1.0)
        self.assertAlmostEqual(hypothesis_subtitles[1].start_time, 1.0)
        self.assertAlmostEqual(hypothesis_subtitles[1].end_time, 2.0)

        first_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[0].word_list)
        self.assertEqual(first_subtititle_text, "This is a simple first frame.")

        second_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[1].word_list)
        self.assertEqual(second_subtititle_text, "This is another frame having two lines.")

    def test_dropped_words(self):
        reference_file_content = """
            1
            00:00:01,000 --> 00:00:02,000
            This is a simple first frame.

            2
            00:00:03,000 --> 00:00:04,000
            This is another frame
            having two lines."""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            Should be dropped.

            2
            00:00:01,000 --> 00:00:02,000
            This is a simple first frame.

            3
            00:00:02,000 --> 00:00:03,000
            Should be dropped.

            4
            00:00:03,000 --> 00:00:04,000
            This is another frame
            having two lines.

            5
            00:00:04,000 --> 00:00:05,000
            Should be dropped.
            """

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        hypothesis_subtitles = time_align_hypothesis_to_reference(hypothesis_subtitles, reference_subtitles)

        self.assertEqual(len(hypothesis_subtitles), 2)

        first_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[0].word_list)
        self.assertEqual(first_subtititle_text, "This is a simple first frame.")

        second_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[1].word_list)
        self.assertEqual(second_subtititle_text, "This is another frame having two lines.")

    def test_partial_overlap(self):
        reference_file_content = """
            1
            00:00:01,000 --> 00:00:02,000
            This is a simple first frame.

            2
            00:00:03,000 --> 00:00:04,000
            This is another frame
            having two lines."""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:02,000
            This is a simple first frame.

            2
            00:00:02,500 --> 00:00:03,500
            This is another frame

            3
            00:00:03,500 --> 00:00:04,500
            having two lines."""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        hypothesis_subtitles = time_align_hypothesis_to_reference(hypothesis_subtitles, reference_subtitles)

        self.assertEqual(len(hypothesis_subtitles), 2)

        first_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[0].word_list)
        self.assertEqual(first_subtititle_text, "simple first frame.")

        second_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[1].word_list)
        self.assertEqual(second_subtititle_text, "another frame having")

    def test_gap_in_overlap(self):
        reference_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            This is a simple first frame.

            2
            00:00:02,000 --> 00:00:03,000
            This is another frame
            having two lines."""

        hypothesis_file_content = """
            1
            00:00:00,000 --> 00:00:03,000
            This is a simple first frame.
            This is another frame
            having two lines."""

        reference_subtitles = create_temporary_file_and_read_it(reference_file_content)
        hypothesis_subtitles = create_temporary_file_and_read_it(hypothesis_file_content)

        hypothesis_subtitles = time_align_hypothesis_to_reference(hypothesis_subtitles, reference_subtitles)

        self.assertEqual(len(hypothesis_subtitles), 2)

        first_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[0].word_list)
        self.assertEqual(first_subtititle_text, "This is a simple")

        second_subtititle_text = " ".join(word.string for word in hypothesis_subtitles[1].word_list)
        self.assertEqual(second_subtititle_text, "frame having two lines.")


class LevenshteinAlignmentTests(unittest.TestCase):
    def test_identical_files(self):
        file_content = """This is a line.
                          That is another one."""

        segments = create_temporary_file_and_read_it(file_content, file_format="plain")

        aligned_segments = levenshtein_align_hypothesis_to_reference(hypothesis=segments, reference=segments)

        self.assertEqual(segments, aligned_segments)

    def test_identical_words(self):
        reference_file_content = """This is a line.
                                    That is another one.
                                    And a third segment."""

        hypothesis_file_content = """This is a line. That
                                     is another
                                     one. And a third segment."""

        reference_segments = create_temporary_file_and_read_it(reference_file_content, file_format="plain")
        hypothesis_segments = create_temporary_file_and_read_it(hypothesis_file_content, file_format="plain")

        hypothesis_segments = levenshtein_align_hypothesis_to_reference(hypothesis_segments, reference_segments)

        self.assertEqual(len(hypothesis_segments), 3)

        first_segment_text = " ".join(word.string for word in hypothesis_segments[0].word_list)
        self.assertEqual(first_segment_text, "This is a line.")

        second_segment_text = " ".join(word.string for word in hypothesis_segments[1].word_list)
        self.assertEqual(second_segment_text, "That is another one.")

        third_segment_text = " ".join(word.string for word in hypothesis_segments[2].word_list)
        self.assertEqual(third_segment_text, "And a third segment.")

    def test_with_edits(self):
        reference_file_content = """This is a line.
                                    That is another one.
                                    And a third segment."""

        hypothesis_file_content = """This is a lines. That this
                                     is another one. And third segment."""

        reference_segments = create_temporary_file_and_read_it(reference_file_content, file_format="plain")
        hypothesis_segments = create_temporary_file_and_read_it(hypothesis_file_content, file_format="plain")

        hypothesis_segments = levenshtein_align_hypothesis_to_reference(hypothesis_segments, reference_segments)

        self.assertEqual(len(hypothesis_segments), 3)

        first_segment_text = " ".join(word.string for word in hypothesis_segments[0].word_list)
        self.assertEqual(first_segment_text, "This is a lines.")

        second_segment_text = " ".join(word.string for word in hypothesis_segments[1].word_list)
        self.assertEqual(second_segment_text, "That this is another one.")

        third_segment_text = " ".join(word.string for word in hypothesis_segments[2].word_list)
        self.assertEqual(third_segment_text, "And third segment.")

    def test_with_edits_at_segment_boundary(self):
        reference_file_content = """This is a line.
                                    That is another one.
                                    And a third segment."""

        hypothesis_file_content = """Some words at the start. This is a line. Where do these
                                     words belong to? another one.
                                     And a third
                                     segment. Some words in the end."""

        reference_segments = create_temporary_file_and_read_it(reference_file_content, file_format="plain")
        hypothesis_segments = create_temporary_file_and_read_it(hypothesis_file_content, file_format="plain")

        hypothesis_segments = levenshtein_align_hypothesis_to_reference(hypothesis_segments, reference_segments)

        self.assertEqual(len(hypothesis_segments), 3)

        first_segment_text = " ".join(word.string for word in hypothesis_segments[0].word_list)
        self.assertEqual(first_segment_text, "Some words at the start. This is a line. Where do these words")

        second_segment_text = " ".join(word.string for word in hypothesis_segments[1].word_list)
        self.assertEqual(second_segment_text, "belong to? another one.")

        third_segment_text = " ".join(word.string for word in hypothesis_segments[2].word_list)
        self.assertEqual(third_segment_text, "And a third segment. Some words in the end.")

    def test_with_very_few_words(self):
        reference_file_content = """This is a line.
                                    That is another one.
                                    And a third segment."""

        hypothesis_file_content = """Very few words."""

        reference_segments = create_temporary_file_and_read_it(reference_file_content, file_format="plain")
        hypothesis_segments = create_temporary_file_and_read_it(hypothesis_file_content, file_format="plain")

        hypothesis_segments = levenshtein_align_hypothesis_to_reference(hypothesis_segments, reference_segments)

        self.assertEqual(len(hypothesis_segments), 3)


if __name__ == '__main__':
    unittest.main()
