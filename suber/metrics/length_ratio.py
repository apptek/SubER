from typing import List
from suber.data_types import Segment

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


def calculate_length_ratio(hypothesis: List[Segment], reference: List[Segment], language: str = None) -> float:
    all_hypothesis_words = [word.string for segment in hypothesis for word in segment.word_list]
    all_reference_words = [word.string for segment in reference for word in segment.word_list]

    full_hypothesis_string = " ".join(all_hypothesis_words)
    full_reference_string = " ".join(all_reference_words)

    # Same tokenizer as used by default for BLEU calculation in SacreBLEU depending on the language, so length ratio we
    # calculate here should correspond to the "ratio" printed by SacreBLEU.
    if language == "ja":
        tokenizer = TokenizerJaMecab()
    elif language == "ko":
        # Import only here to keep compatible with sacrebleu versions < 2.2 for all other languages.
        from sacrebleu.tokenizers.tokenizer_ko_mecab import TokenizerKoMecab

        tokenizer = TokenizerKoMecab()
    elif language == "zh":
        tokenizer = TokenizerZh()
    else:
        tokenizer = Tokenizer13a()

    num_tokens_hypothesis = len(tokenizer(full_hypothesis_string).split())
    num_tokens_reference = len(tokenizer(full_reference_string).split())

    length_ratio = num_tokens_hypothesis / num_tokens_reference if num_tokens_reference else 0.0

    return round(length_ratio * 100, 3)
