import unittest
import tempfile
import subprocess

from suber.constants import END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL


class SentenceSegmentationTests(unittest.TestCase):

    def test_srt_to_plain(self):
        input_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            This is a simple first frame. This

            2
            00:00:01,000 --> 00:00:02,000
            is another frame
            having two lines."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_input_file, \
                tempfile.NamedTemporaryFile(mode="w+", suffix=".srt") as temporary_output_file:
            temporary_input_file.write(input_file_content)
            temporary_input_file.flush()

            subprocess.run(
                f"python3 -m suber.tools.srt_to_plain "
                f"--input-file {temporary_input_file.name} --output-file {temporary_output_file.name}".split(),
                check=True)

            output_file_content = temporary_output_file.readlines()

            self.assertEqual(len(output_file_content), 2)

            self.assertEqual(output_file_content[0].strip(),
                             f"This is a simple first frame. This {END_OF_BLOCK_SYMBOL}")
            self.assertEqual(output_file_content[1].strip(),
                             f"is another frame {END_OF_LINE_SYMBOL} having two lines. {END_OF_BLOCK_SYMBOL}")

            # Again, now with sentence segmentation.
            subprocess.run(
                f"python3 -m suber.tools.srt_to_plain --sentence-segmentation "
                f"--input-file {temporary_input_file.name} --output-file {temporary_output_file.name}".split(),
                check=True)

            temporary_output_file.seek(0)
            output_file_content = temporary_output_file.readlines()

            self.assertEqual(len(output_file_content), 2)

            self.assertEqual(output_file_content[0].strip(),
                             f"This is a simple first frame.")
            self.assertEqual(output_file_content[1].strip(),
                             f"This {END_OF_BLOCK_SYMBOL} is another frame {END_OF_LINE_SYMBOL} "
                             f"having two lines. {END_OF_BLOCK_SYMBOL}")

    def test_levenshtein_align_hyp_to_ref(self):
        reference_file_content = """This is a line.
                                    That is another one.
                                    And a third segment."""

        hypothesis_file_content = """This is a lines. That this
                                     is another one. And third segment."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_reference_file, \
                tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_hypothesis_file, \
                tempfile.NamedTemporaryFile(mode="w+", suffix=".srt") as temporary_output_file:
            temporary_reference_file.write(reference_file_content)
            temporary_reference_file.flush()

            temporary_hypothesis_file.write(hypothesis_file_content)
            temporary_hypothesis_file.flush()

            subprocess.run(
                f"python3 -m suber.tools.align_hyp_to_ref --method levenshtein "
                f"--hypothesis {temporary_hypothesis_file.name} --reference {temporary_reference_file.name} "
                f"--hypothesis-format plain --reference-format plain "
                f"--aligned-hypothesis {temporary_output_file.name}".split(),
                check=True)

            output_file_content = temporary_output_file.readlines()

        self.assertEqual(len(output_file_content), 3)
        self.assertEqual(output_file_content[0].strip(), "This is a lines.")
        self.assertEqual(output_file_content[1].strip(), "That this is another one.")
        self.assertEqual(output_file_content[2].strip(), "And third segment.")

    def test_time_align_hyp_to_ref(self):
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_reference_file, \
                tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_hypothesis_file, \
                tempfile.NamedTemporaryFile(mode="w+", suffix=".srt") as temporary_output_file:
            temporary_reference_file.write(reference_file_content)
            temporary_reference_file.flush()

            temporary_hypothesis_file.write(hypothesis_file_content)
            temporary_hypothesis_file.flush()

            subprocess.run(
                f"python3 -m suber.tools.align_hyp_to_ref --method time "
                f"--hypothesis {temporary_hypothesis_file.name} --reference {temporary_reference_file.name} "
                f"--hypothesis-format SRT --reference-format SRT "
                f"--aligned-hypothesis {temporary_output_file.name}".split(),
                check=True)

            output_file_content = temporary_output_file.readlines()

        self.assertEqual(len(output_file_content), 2)
        self.assertEqual(output_file_content[0].strip(), "simple first frame.")
        self.assertEqual(output_file_content[1].strip(), "another frame having")


if __name__ == '__main__':
    unittest.main()
