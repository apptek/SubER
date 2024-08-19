from typing import List
from suber.data_types import Subtitle

from pyannote.core import Segment, Annotation
from pyannote.metrics.detection import (
    DetectionAccuracy,
    DetectionPrecision,
    DetectionRecall,
    DetectionPrecisionRecallFMeasure,
)


def calculate_time_span_accuracy(hypothesis: List[Subtitle], reference: List[Subtitle], metric="time_span_accuracy"):

    reference_timings = Annotation()
    for subtitle in reference:
        reference_timings[Segment(subtitle.start_time, subtitle.end_time)] = str(subtitle.index)

    hypothesis_timings = Annotation()
    for subtitle in hypothesis:
        hypothesis_timings[Segment(subtitle.start_time, subtitle.end_time)] = str(subtitle.index)

    if metric == "time_span_accuracy":
        pyannote_metric = DetectionAccuracy()
    elif metric == "time_span_precision":
        pyannote_metric = DetectionPrecision()
    elif metric == "time_span_recall":
        pyannote_metric = DetectionRecall()
    elif metric == "time_span_f1":
        pyannote_metric = DetectionPrecisionRecallFMeasure()

    score = pyannote_metric(reference_timings, hypothesis_timings)

    return round(score, 3)
