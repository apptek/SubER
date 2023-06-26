#!/usr/bin/env python3

import argparse

from suber.file_readers import SRTFileReader
from suber.sentence_segmentation import resegment_based_on_punctuation
from suber.utilities import segment_to_string


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extracts plain text from SRT files.")
    parser.add_argument("-i", "--input-file", required=True, help="The input SRT file.")
    parser.add_argument("-o", "--output-file", required=True, help="The plain output file.")
    parser.add_argument("-s", "--sentence-segmentation", action="store_true",
                        help="If enabled, output sentences instead of subtitle segments.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    segments = SRTFileReader(args.input_file).read()

    if args.sentence_segmentation:
        segments = resegment_based_on_punctuation(segments)

    with open(args.output_file, "w", encoding="utf-8") as output_file_object:
        for segment in segments:
            output_file_object.write(segment_to_string(segment, include_line_breaks=True) + '\n')


if __name__ == "__main__":
    main()
