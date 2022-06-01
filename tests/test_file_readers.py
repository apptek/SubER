import unittest

from suber.data_types import LineBreak
from suber.file_readers.srt_file_reader import SRTFormatError
from .utilities import create_temporary_file_and_read_it


class PlainFileReaderTests(unittest.TestCase):
    def test_empty_file(self):
        segments = create_temporary_file_and_read_it("", file_format="plain")
        self.assertFalse(segments)

    def test_simple_file(self):
        file_content = """This is a line. <eob>
                          These are <eol> two subtitle lines. <eob>"""

        segments = create_temporary_file_and_read_it(file_content, file_format="plain")

        self.assertEqual(len(segments), 2)

        first_segment_text = " ".join(word.string for word in segments[0].word_list)
        self.assertEqual(first_segment_text, "This is a line.")
        self.assertTrue(all(word.line_break == LineBreak.NONE for word in segments[0].word_list[:-1]))
        self.assertEqual(segments[0].word_list[-1].line_break, LineBreak.END_OF_BLOCK)

        second_segment_text = " ".join(word.string for word in segments[1].word_list)
        self.assertEqual(second_segment_text, "These are two subtitle lines.")
        self.assertEqual(segments[1].word_list[1].line_break, LineBreak.END_OF_LINE)
        self.assertEqual(segments[1].word_list[-1].line_break, LineBreak.END_OF_BLOCK)


class SRTFileReaderTests(unittest.TestCase):
    def test_empty_file(self):
        subtitles = create_temporary_file_and_read_it("")
        self.assertFalse(subtitles)

    def test_simple_file(self):
        file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            This is a simple first frame.

            2
            00:00:01,000 --> 00:00:02,000
            This is another frame
            having two lines."""

        subtitles = create_temporary_file_and_read_it(file_content)

        self.assertEqual(len(subtitles), 2)

        self.assertEqual(subtitles[0].index, 1)
        self.assertEqual(subtitles[1].index, 2)

        self.assertAlmostEqual(subtitles[0].start_time, 0.0)
        self.assertAlmostEqual(subtitles[0].end_time, 1.0)
        self.assertTrue(all(word.line_break == LineBreak.NONE for word in subtitles[0].word_list[:-1]))
        self.assertEqual(subtitles[0].word_list[-1].line_break, LineBreak.END_OF_BLOCK)

        self.assertAlmostEqual(subtitles[1].start_time, 1.0)
        self.assertAlmostEqual(subtitles[1].end_time, 2.0)
        self.assertEqual(subtitles[1].word_list[3].line_break, LineBreak.END_OF_LINE)
        self.assertEqual(subtitles[1].word_list[-1].line_break, LineBreak.END_OF_BLOCK)

        first_subtititle_text = " ".join(word.string for word in subtitles[0].word_list)
        self.assertEqual(first_subtititle_text, "This is a simple first frame.")

        second_subtititle_text = " ".join(word.string for word in subtitles[1].word_list)
        self.assertEqual(second_subtititle_text, "This is another frame having two lines.")

    def test_overlap_in_time(self):
        file_content = """
            1
            00:00:01,000 --> 00:00:02,000
            This is a simple first frame.

            2
            00:00:00,000 --> 00:00:01,000
            This one is before the first one in time."""

        with self.assertRaises(SRTFormatError):
            create_temporary_file_and_read_it(file_content)


if __name__ == '__main__':
    unittest.main()
