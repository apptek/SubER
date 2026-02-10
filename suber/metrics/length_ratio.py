from typing import List

from suber.data_types import Segment
from suber.tokenizers import get_sacrebleu_tokenizer


def calculate_length_ratio(hypothesis: List[Segment], reference: List[Segment], language: str = None) -> float:
    all_hypothesis_words = [word.string for segment in hypothesis for word in segment.word_list]
    all_reference_words = [word.string for segment in reference for word in segment.word_list]

    full_hypothesis_string = " ".join(all_hypothesis_words)
    full_reference_string = " ".join(all_reference_words)

    # Same tokenizer as used by default for BLEU calculation in SacreBLEU depending on the language, so length ratio we
    # calculate here should correspond to the "ratio" printed by SacreBLEU.
    tokenizer = get_sacrebleu_tokenizer(language)

    num_tokens_hypothesis = len(tokenizer(full_hypothesis_string).split())
    num_tokens_reference = len(tokenizer(full_reference_string).split())

    length_ratio = num_tokens_hypothesis / num_tokens_reference if num_tokens_reference else 0.0

    return round(length_ratio * 100, 3)
