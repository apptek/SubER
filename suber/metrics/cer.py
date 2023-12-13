import Levenshtein
import string
from typing import List

from suber.data_types import Segment
from suber.utilities import segment_to_string


def calculate_character_error_rate(hypothesis: List[Segment], reference: List[Segment], metric="CER") -> float:
    assert len(hypothesis) == len(reference), (
        "Number of hypothesis segments does not match reference, alignment step missing?")

    hypothesis_strings = [segment_to_string(segment) for segment in hypothesis]
    reference_strings = [segment_to_string(segment) for segment in reference]

    if metric != "CER-cased":
        remove_punctuation_table = str.maketrans('', '', string.punctuation)

        def normalize_string(string):
            string = string.translate(remove_punctuation_table)
            # Ellipsis is a common character in subtitles which is not included in string.punctuation.
            string = string.replace('…', '')
            string = string.lower()
            return string

        hypothesis_strings = [normalize_string(string) for string in hypothesis_strings]
        reference_strings = [normalize_string(string) for string in reference_strings]

    num_edits = 0
    num_reference_characters = 0
    for hypothesis_string, reference_string, in zip(hypothesis_strings, reference_strings):
        num_edits += Levenshtein.distance(hypothesis_string, reference_string)
        num_reference_characters += len(reference_string)

    if num_reference_characters:
        cer_score = num_edits / num_reference_characters
    else:
        cer_score = 1.0 if num_edits else 0.0

    return round(cer_score * 100, 3)
