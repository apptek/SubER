import unittest
import tempfile
import subprocess
import json


class MainFunctionTests(unittest.TestCase):

    def test_main_function(self):
        input_file_content = """
            1
            00:00:00,000 --> 00:00:01,000
            This is a simple first frame.

            2
            00:00:01,000 --> 00:00:02,000
            This is another frame
            having two lines."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_input_file:
            temporary_input_file.writelines(input_file_content)
            temporary_input_file.flush()

            # For now, just check that all metrics, including hyp-to-ref-alignment, run through.
            completed_process = subprocess.run(
                f"python3 -m suber "
                f"--hypothesis {temporary_input_file.name} --reference {temporary_input_file.name} "
                f"--metrics SubER WER CER BLEU TER chrF TER-br WER-seg BLEU-seg AS-BLEU t-BLEU".split(),
                check=True, stdout=subprocess.PIPE)

            # Also check that output is a valid json.
            metric_scores = json.loads(completed_process.stdout.decode("utf-8"))
            self.assertTrue(metric_scores)


if __name__ == '__main__':
    unittest.main()
