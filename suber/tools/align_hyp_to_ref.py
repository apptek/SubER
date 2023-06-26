#!/usr/bin/env python3

import argparse

from suber.file_readers import read_input_file
from suber.hyp_to_ref_alignment import levenshtein_align_hypothesis_to_reference
from suber.hyp_to_ref_alignment import time_align_hypothesis_to_reference
from suber.utilities import segment_to_string


def parse_arguments():
    parser = argparse.ArgumentParser(description="Re-segments the hypothesis file to match the reference. This can "
                                                 "either be done via Levenshtein alignment, or using the subtitle "
                                                 "timings, if available.")
    parser.add_argument("-H", "--hypothesis", required=True, help="The input file.")
    parser.add_argument("-R", "--reference", required=True, help="The reference file.")
    parser.add_argument("-o", "--aligned-hypothesis", required=True,
                        help="The aligned hypothesis output file in plain format.")
    parser.add_argument("-f", "--hypothesis-format", default="SRT", help="Hypothesis file format, 'SRT' or 'plain'.")
    parser.add_argument("-F", "--reference-format", default="SRT", help="Reference file format, 'SRT' or 'plain'.")
    parser.add_argument("-m", "--method", default="levenshtein",
                        help="The alignment method, either 'levenshtein' or 'time'. See the "
                             "'suber.hyp_to_ref_alignment' module. 'time' only supported if both hypothesis and "
                             "reference are given in SRT format.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.method == "time" and not args.hypothesis_format == "SRT" and args.reference_format == "SRT":
        raise ValueError("For time alignment, both hypothesis and reference have to be given in SRT format.")

    hypothesis_segments = read_input_file(args.hypothesis, file_format=args.hypothesis_format)
    reference_segments = read_input_file(args.reference, file_format=args.reference_format)

    if args.method == "levenshtein":
        aligned_hypothesis_segments = levenshtein_align_hypothesis_to_reference(
            hypothesis=hypothesis_segments, reference=reference_segments)
    elif args.method == "time":
        aligned_hypothesis_segments = time_align_hypothesis_to_reference(
            hypothesis=hypothesis_segments, reference=reference_segments)

    with open(args.aligned_hypothesis, "w", encoding="utf-8") as output_file_object:
        for segment in aligned_hypothesis_segments:
            output_file_object.write(segment_to_string(segment) + '\n')


if __name__ == "__main__":
    main()
