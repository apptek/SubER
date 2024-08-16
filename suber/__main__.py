#!/usr/bin/env python3

import argparse
import json

from collections import OrderedDict

from suber.file_readers import read_input_file
from suber.concat_input_files import create_concatenated_segments
from suber.hyp_to_ref_alignment import levenshtein_align_hypothesis_to_reference
from suber.hyp_to_ref_alignment import time_align_hypothesis_to_reference
from suber.metrics.suber import calculate_SubER
from suber.metrics.suber_statistics import SubERStatisticsCollector
from suber.metrics.sacrebleu_interface import calculate_sacrebleu_metric
from suber.metrics.jiwer_interface import calculate_word_error_rate
from suber.metrics.cer import calculate_character_error_rate
from suber.metrics.length_ratio import calculate_length_ratio


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="SubER - Subtitle Edit Rate. An automatic, reference-based, segmentation- and timing-aware "
                    "edit distance metric to measure quality of subtitle files. Basic usage: "
                    "'python -m suber -H hypothesis.srt -R reference.srt'")
    parser.add_argument("-H", "--hypothesis", required=True, nargs="+",
                        help="The input files to score. Usually just one file, but we support test sets consisting of "
                             "multiple files.")
    parser.add_argument("-R", "--reference", required=True, nargs="+",
                        help="The reference files. Usually just one file, but we support test sets consisting of "
                             "multiple files.")
    parser.add_argument("-m", "--metrics", nargs="+", default=["SubER"], help="The metrics to compute.")
    parser.add_argument("-f", "--hypothesis-format", default="SRT", help="Hypothesis file format, 'SRT' or 'plain'.")
    parser.add_argument("-F", "--reference-format", default="SRT", help="Reference file format, 'SRT' or 'plain'.")
    parser.add_argument("--suber-statistics", action="store_true",
                        help="If set, will create an '#info' field in the output containing statistics about the "
                             "different edit operations used to calculate the SubER score.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    check_metrics(args.metrics)
    check_file_formats(args.hypothesis_format, args.reference_format, args.metrics)

    # A "segment" is a subtitle in case of SRT file input, or a line of text in case of plain input.
    if len(args.hypothesis) == 1 and len(args.reference) == 1:
        hypothesis_segments = read_input_file(args.hypothesis[0], file_format=args.hypothesis_format)
        reference_segments = read_input_file(args.reference[0], file_format=args.reference_format)
    else:
        hypothesis_segments, reference_segments = create_concatenated_segments(
            args.hypothesis, args.reference, args.hypothesis_format, args.reference_format)

    # Aligned hypotheses, either by Levenshtein distance or timing, are only needed by some metrics so we create them
    # lazily here.
    levenshtein_aligned_hypothesis_segments = None
    time_aligned_hypothesis_segments = None

    results = OrderedDict()
    additional_outputs = OrderedDict()

    for metric in args.metrics:
        if metric in results:
            continue  # specified multiple times by the user

        if metric == "length_ratio":
            results[metric] = calculate_length_ratio(hypothesis=hypothesis_segments, reference=reference_segments)
            continue

        # When using existing parallel segments there will always be a <eob> word match in the end, don't count it.
        # On the other hand, if hypothesis gets aligned to reference a match is not guaranteed, so count it.
        score_break_at_segment_end = False

        full_metric_name = metric
        hypothesis_segments_to_use = hypothesis_segments

        if metric.startswith("AS-"):
            # "AS" stands for automatic segmentation, in particular re-segmentation of the hypothesis using
            # the Levenshtein alignment to the reference.
            # AS-WER and AS-BLEU were introduced by Matusov et al. https://aclanthology.org/2005.iwslt-1.19.pdf
            if levenshtein_aligned_hypothesis_segments is None:
                levenshtein_aligned_hypothesis_segments = levenshtein_align_hypothesis_to_reference(
                    hypothesis=hypothesis_segments, reference=reference_segments)

            hypothesis_segments_to_use = levenshtein_aligned_hypothesis_segments
            metric = metric[len("AS-"):]
            score_break_at_segment_end = True

        elif metric.startswith("t-"):
            # "t" stands for timed. Subtitle timings will be used to re-segment the hypothesis to match the reference
            # segments. t-BLEU was introduced by Cherry et al.
            # https://www.isca-archive.org/interspeech_2021/cherry21_interspeech.pdf
            if time_aligned_hypothesis_segments is None:
                time_aligned_hypothesis_segments = time_align_hypothesis_to_reference(
                    hypothesis=hypothesis_segments, reference=reference_segments)

            hypothesis_segments_to_use = time_aligned_hypothesis_segments
            metric = metric[len("t-"):]
            score_break_at_segment_end = True

        elif not metric.startswith("SubER") and len(hypothesis_segments_to_use) != len(reference_segments):
            raise ValueError(f"Metric '{metric}' assumes same number of segments in hypothesis and reference, but got "
                             f"{len(hypothesis_segments)} hypothesis and {len(reference_segments)} "
                             f"reference segments.")

        if metric.startswith("SubER"):
            statistics_collector = SubERStatisticsCollector() if args.suber_statistics else None

            metric_score = calculate_SubER(
                hypothesis=hypothesis_segments_to_use, reference=reference_segments, metric=metric,
                statistics_collector=statistics_collector)

            if statistics_collector:
                additional_outputs[full_metric_name] = statistics_collector.get_statistics()

        elif metric.startswith("WER"):
            metric_score = calculate_word_error_rate(
                hypothesis=hypothesis_segments_to_use, reference=reference_segments, metric=metric,
                score_break_at_segment_end=score_break_at_segment_end)

        elif metric.startswith("CER"):
            metric_score = calculate_character_error_rate(
                hypothesis=hypothesis_segments_to_use, reference=reference_segments, metric=metric)

        else:
            metric_score = calculate_sacrebleu_metric(
                hypothesis=hypothesis_segments_to_use, reference=reference_segments, metric=metric,
                score_break_at_segment_end=score_break_at_segment_end)

        results[full_metric_name] = metric_score

    if additional_outputs:
        results["#info"] = additional_outputs

    json_results = json.dumps(results, indent=4)
    print(json_results)


def check_metrics(metrics):
    allowed_metrics = {
        # Our proposed metric:
        "SubER", "SubER-cased",
        # Established ASR and MT metrics, requiring aligned hypothesis-references segments:
        "WER", "CER", "BLEU", "TER", "chrF",
        # Cased and punctuated variants of the above:
        "WER-cased", "CER-cased",
        # Segmentation-aware variants of the above that include line breaks as tokens:
        "WER-seg", "BLEU-seg", "TER-seg",
        # Same as "TER-seg" but all words replaced by a mask token,
        # proposed by Karakanta et al. https://aclanthology.org/2020.iwslt-1.26.pdf
        "TER-br",
        # With an "AS-" prefix, the metric is computed after Levenshtein alignment of hypothesis and reference:
        "AS-WER", "AS-CER", "AS-BLEU", "AS-TER", "AS-chrF", "AS-WER-cased", "AS-CER-cased", "AS-WER-seg",
        "AS-BLEU-seg", "AS-TER-seg", "AS-TER-br",
        # With an "t-" prefix, the metric is computed after time alignment of hypothesis and reference:
        "t-WER", "t-CER", "t-BLEU", "t-TER", "t-chrF", "t-WER-cased", "t-CER-cased", "t-WER-seg", "t-BLEU-seg",
        "t-TER-seg", "t-TER-br",
        # Hypothesis to reference length ratio in terms of number of tokens.
        "length_ratio"}

    invalid_metrics = list(sorted(set(metrics) - allowed_metrics))
    if invalid_metrics:
        raise ValueError(f"Invalid metric(s): {' '.join(invalid_metrics)}")


def check_file_formats(hypothesis_format, reference_format, metrics):
    is_plain_input = (hypothesis_format == "plain" or reference_format == "plain")
    for metric in metrics:
        if ((metric == "SubER" or metric.startswith("t-")) and is_plain_input):
            raise ValueError(f"Metric '{metric}' requires timing information and can only be computed on SRT "
                             f"files (both hypothesis and reference).")


if __name__ == "__main__":
    main()
