import functools
from typing import List

from sacrebleu.metrics import BLEU, TER, CHRF

from suber.data_types import Segment
from suber.utilities import segment_to_string, get_segment_to_string_opts_from_metric


def calculate_sacrebleu_metric(hypothesis: List[Segment], reference: List[Segment],
                               metric="BLEU", score_break_at_segment_end=True) -> float:

    assert len(hypothesis) == len(reference), (
        "Number of hypothesis segments does not match reference, alignment step missing?")

    include_breaks, mask_words, metric = get_segment_to_string_opts_from_metric(metric)

    if metric == "BLEU":
        sacrebleu_metric = BLEU()
    elif metric == "TER":
        sacrebleu_metric = TER()
    elif metric == "chrF":
        sacrebleu_metric = CHRF()
    else:
        raise ValueError(f"Unsupported sacrebleu metric '{metric}'.")

    # Sacrebleu currently does not allow empty references, just skip empty reference segments as a workaround.
    if not all(segment.word_list for segment in reference):
        empty_reference_indices = [index for index, segment in enumerate(reference) if not segment.word_list]
        reference = [segment for index, segment in enumerate(reference) if index not in empty_reference_indices]
        hypothesis = [segment for index, segment in enumerate(hypothesis) if index not in empty_reference_indices]

    segment_to_string_ = functools.partial(
        segment_to_string, include_line_breaks=include_breaks, mask_all_words=mask_words,
        include_last_break=score_break_at_segment_end)

    hypothesis_strings = [segment_to_string_(segment) for segment in hypothesis]
    reference_strings = [[segment_to_string_(segment) for segment in reference]]  # sacrebleu expects nested list

    if include_breaks:
        # BLEU tokenizer would split "<eol>" into "< eol >".
        hypothesis_strings = [string.replace("<eol>", "eol").replace("<eob>", "eob") for string in hypothesis_strings]
        reference_strings[0] = [
            string.replace("<eol>", "eol").replace("<eob>", "eob") for string in reference_strings[0]]

    sacrebleu_score = sacrebleu_metric.corpus_score(hypotheses=hypothesis_strings, references=reference_strings)

    return round(sacrebleu_score.score, 3)
