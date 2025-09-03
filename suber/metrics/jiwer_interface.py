import jiwer
import functools
from typing import List

from sacrebleu.tokenizers.tokenizer_ter import TercomTokenizer

from suber.data_types import Segment
from suber.constants import ASIAN_LANGUAGE_CODES
from suber.utilities import segment_to_string, get_segment_to_string_opts_from_metric


def calculate_word_error_rate(hypothesis: List[Segment], reference: List[Segment], metric="WER",
                              score_break_at_segment_end=True, language: str = None) -> float:

    assert len(hypothesis) == len(reference), (
        "Number of hypothesis segments does not match reference, alignment step missing?")

    if metric == "WER-cased":
        transformations = jiwer.Compose([
            # Note: the original release used no tokenization here. We find this change to have a minor positive effect
            # on correlation with post-edit effort (-0.657 vs. -0.650 in Table 1, row 2, "Combined" in our paper.)
            TercomTokenize(language),
            jiwer.ReduceToListOfListOfWords(),
        ])
        metric = "WER"

    else:
        transformations = [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            # Ellipsis is a common character in subtitles that older jiwer versions would not remove by default.
            jiwer.RemoveSpecificWords(['…']),
            jiwer.ReduceToListOfListOfWords(),
        ]
        # For most languages no tokenizer needed when punctuation is removed. Not true though for languages that do not
        # use spaces to separate words.
        if language in ASIAN_LANGUAGE_CODES:
            transformations.insert(3, TercomTokenize(language))

        transformations = jiwer.Compose(transformations)

    include_breaks, mask_words, metric = get_segment_to_string_opts_from_metric(metric)
    assert metric == "WER"

    segment_to_string_ = functools.partial(
        segment_to_string, include_line_breaks=include_breaks, mask_all_words=mask_words,
        include_last_break=score_break_at_segment_end)

    hypothesis_strings = [segment_to_string_(segment) for segment in hypothesis]
    reference_strings = [segment_to_string_(segment) for segment in reference]

    wer_score = jiwer.wer(
        reference_strings,
        hypothesis_strings,
        reference_transform=transformations,
        hypothesis_transform=transformations)

    return round(wer_score * 100, 3)


class TercomTokenize(jiwer.AbstractTransform):
    def __init__(self, language: str):
        asian_support = language in ASIAN_LANGUAGE_CODES
        self.tokenizer = TercomTokenizer(
            normalized=True, no_punct=False, case_sensitive=True, asian_support=asian_support)

    def process_string(self, s: str):
        # TercomTokenizer would split "<eol>" into "< eol >"
        s = s.replace("<eol>", "eol").replace("<eob>", "eob")
        return self.tokenizer(s)
