import unittest

from suber.metrics.pyannote_interface import calculate_time_span_accuracy
from .utilities import create_temporary_file_and_read_it


class PyAnnoteInterfaceTest(unittest.TestCase):
    def setUp(self):
        reference = """
            1
            0:00:01.000 --> 0:00:02.000
            This is a subtitle.

            2
            0:00:03.000 --> 0:00:04.000
            And another one!"""

        self._reference = create_temporary_file_and_read_it(reference)

    def test_time_span_accuracy_empty(self):
        for metric in ("time_span_accuracy", "time_span_precision", "time_span_recall", "time_span_recall"):
            score = calculate_time_span_accuracy(hypothesis=[], reference=[], metric=metric)
            self.assertAlmostEqual(score, 1.0)

        accuracy = calculate_time_span_accuracy(hypothesis=[], reference=self._reference, metric="time_span_accuracy")
        self.assertAlmostEqual(accuracy, 0.333)  # Total interval is from 1 to 4 seconds, 1 second gap is true negative.
        precision = calculate_time_span_accuracy(hypothesis=[], reference=self._reference, metric="time_span_precision")
        self.assertAlmostEqual(precision, 1.0)
        recall = calculate_time_span_accuracy(hypothesis=[], reference=self._reference, metric="time_span_recall")
        self.assertAlmostEqual(recall, 0.0)

        accuracy = calculate_time_span_accuracy(hypothesis=self._reference, reference=[], metric="time_span_accuracy")
        self.assertAlmostEqual(accuracy, 0.333)
        precision = calculate_time_span_accuracy(hypothesis=self._reference, reference=[], metric="time_span_precision")
        self.assertAlmostEqual(precision, 0.0)
        recall = calculate_time_span_accuracy(hypothesis=self._reference, reference=[], metric="time_span_recall")
        self.assertAlmostEqual(recall, 1.0)

    def test_time_span_accuracy_perfect(self):
        for metric in ("time_span_accuracy", "time_span_precision", "time_span_recall", "time_span_recall"):
            score = calculate_time_span_accuracy(hypothesis=self._reference, reference=self._reference, metric=metric)
            self.assertAlmostEqual(score, 1.0)

    def test_time_span_accuracy_no_overlap(self):
        hypothesis = """
            1
            0:00:02.000 --> 0:00:03.000
            This is a subtitle.

            2
            0:00:04.000 --> 0:00:05.000
            And another one!"""
        hypothesis = create_temporary_file_and_read_it(hypothesis)

        for metric in ("time_span_accuracy", "time_span_precision", "time_span_recall", "time_span_recall"):
            score = calculate_time_span_accuracy(hypothesis=hypothesis, reference=self._reference, metric=metric)
            self.assertAlmostEqual(score, 0.0)

    def test_time_span_accuracy_some_overlap(self):
        hypothesis = """
            1
            0:00:01.500 --> 0:00:02.500
            The text doesn't matter.

            2
            0:00:03.200 --> 0:00:03.800
            """
        hypothesis = create_temporary_file_and_read_it(hypothesis)

        accuracy = calculate_time_span_accuracy(
            hypothesis=hypothesis, reference=self._reference, metric="time_span_accuracy"
        )
        self.assertAlmostEqual(accuracy, 1.6 / 3, places=3)
        precision = calculate_time_span_accuracy(
            hypothesis=hypothesis, reference=self._reference, metric="time_span_precision"
        )
        self.assertAlmostEqual(precision, 1.1 / 1.6, places=3)
        recall = calculate_time_span_accuracy(
            hypothesis=hypothesis, reference=self._reference, metric="time_span_recall"
        )
        self.assertAlmostEqual(recall, 1.1 / 2, places=3)
        f1 = calculate_time_span_accuracy(hypothesis=hypothesis, reference=self._reference, metric="time_span_f1")
        self.assertAlmostEqual(f1, 2 * precision * recall / (precision + recall), places=3)
