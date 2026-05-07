from typing import List

from suber.data_types import Segment
from suber.utilities import segment_to_string


def calculate_bleurt(hypothesis: List[Segment], reference: List[Segment], checkpoint=None) -> float:

    from bleurt import score  # Local import to make dependency optional.

    if not checkpoint:
        raise ValueError(
            "BLEURT checkpoint needs to be downloaded and specified via --bleurt-checkpoint. "
            "See https://github.com/google-research/bleurt/blob/master/README.md")

    score.logging.set_verbosity("INFO")

    hypothesis_strings = [segment_to_string(segment) for segment in hypothesis]
    reference_strings = [segment_to_string(segment) for segment in reference]

    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=reference_strings, candidates=hypothesis_strings)

    average_score = sum(scores) / len(scores)

    return round(average_score, 3)
