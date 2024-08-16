import re
import datetime
import numpy


from suber.file_readers.file_reader_base import FileReaderBase
from suber.data_types import LineBreak, TimedWord, Subtitle


class SRTFormatError(Exception):
    pass


class SRTFileReader(FileReaderBase):
    allowed_time_formats = {
        "iso": r"\d+:\d+:\d+\.\d+",
        "iso_with_comma": r"\d+:\d+:\d+,\d+",
        "seconds": r"^\d+(\.\d+)?$",
    }

    def _parse_lines(self, file_object):
        subtitles = []

        subtitle_index = None
        start_time, end_time = None, None
        word_list = None

        for line in file_object:
            line = line.strip()

            if subtitle_index is None:
                # We expect this line to be the subtitle index. Additional empty line is okay too.
                if line:
                    try:
                        subtitle_index = int(line.replace("\ufeff", ""))
                    except ValueError as e:
                        raise SRTFormatError(f"Tried to read subtitle index from '{line}' but failed.") from e

            elif start_time is None:
                # We expect this line to be the time string. Additional empty line is okay too.
                if line:
                    start_time, end_time = self._parse_time_stamp(line)
                    if end_time < start_time:
                        raise SRTFormatError(f"End time {end_time} is before start time {start_time}.")

                    if subtitles and subtitles[-1].end_time > start_time:
                        start_time_string = line.split()[0]
                        if start_time < subtitles[-1].start_time:
                            raise SRTFormatError("Subtitles must appear ordered according to their start time, "
                                                 f"violated by subtitle at '{start_time_string}'.")

                    assert word_list is None
                    word_list = []  # start collecting words

            elif line:
                # We expect this line to contain subtitle text.
                assert subtitle_index is not None
                assert start_time is not None
                assert end_time is not None

                # We don't consider formatting tags <i>, <b>, etc. in the evaluation.
                # TODO: maybe we want this regex to cover more cases
                line = re.sub('</?[^>]>', '', line)

                word_list.extend([
                    TimedWord(
                        string=word,
                        subtitle_start_time=start_time,
                        subtitle_end_time=end_time)
                    for word in line.split()])

                if word_list:
                    word_list[-1].line_break = LineBreak.END_OF_LINE

            else:
                # This is an empty line after lines of subtitle text which ends the current subtitle.
                assert word_list is not None
                assert start_time is not None
                assert end_time is not None

                if word_list:  # might be an empty subtitle
                    word_list[-1].line_break = LineBreak.END_OF_BLOCK

                    self._set_approximate_word_times(word_list, start_time, end_time)

                subtitles.append(
                    Subtitle(word_list=word_list, index=subtitle_index, start_time=start_time, end_time=end_time))

                subtitle_index = None
                start_time, end_time = None, None
                word_list = None

        if word_list is not None:
            # handle last subtitle
            assert subtitle_index is not None
            assert start_time is not None and end_time is not None

            if word_list:  # might be an empty subtitle
                word_list[-1].line_break = LineBreak.END_OF_BLOCK

                self._set_approximate_word_times(word_list, start_time, end_time)

            subtitles.append(
                Subtitle(word_list=word_list, index=subtitle_index, start_time=start_time, end_time=end_time))

        return subtitles

    @classmethod
    def _set_approximate_word_times(cls, word_list, start_time, end_time):
        """
        Linearly interpolates word times from the subtitle start and end time as described in
        https://www.isca-archive.org/interspeech_2021/cherry21_interspeech.pdf
        """
        # Remove small margin to guarantee the first and last word will always be counted as within the subtitle.
        epsilon = 1e-8
        start_time = start_time + epsilon
        end_time = end_time - epsilon

        num_words = len(word_list)
        duration = end_time - start_time
        assert duration >= 0

        approximate_word_times = numpy.linspace(start=start_time, stop=end_time, num=num_words)
        for word_time, word in zip(approximate_word_times, word_list):
            word.approximate_word_time = word_time

    @classmethod
    def _parse_time_stamp(cls, time_stamp):
        time_stamp_tokens = time_stamp.split()
        if len(time_stamp_tokens) != 3 or not time_stamp_tokens[1].endswith("->"):
            raise SRTFormatError(f"Could not parse subtitle times from '{time_stamp}'.")

        start_time = cls._seconds_from_time_code(time_stamp_tokens[0])
        end_time = cls._seconds_from_time_code(time_stamp_tokens[2])

        return start_time, end_time

    @classmethod
    def _seconds_from_time_code(cls, time_code):
        detected_time_format = None
        for time_format_name, time_format_regex in cls.allowed_time_formats.items():
            if re.match(time_format_regex, time_code):
                detected_time_format = time_format_name
                break

        if not detected_time_format:
            raise SRTFormatError(f"Could not detect format of time code '{time_code}'.")

        try:
            if detected_time_format == "seconds":
                seconds = float(time_code)
            else:
                assert detected_time_format in ["iso", "iso_with_comma"]
                datetime_format_string = "%H:%M:%S,%f" if detected_time_format == "iso_with_comma" else "%H:%M:%S.%f"

                datetime_object = datetime.datetime.strptime(time_code, datetime_format_string)
                # Set year to 1 (default is 1900) to set this time in relation to datetime.datetime.min.
                datetime_object = datetime_object.replace(year=1)

                seconds = (datetime_object - datetime.datetime.min).total_seconds()
        except Exception as e:
            raise SRTFormatError(f"Could not convert '{time_code}' to seconds. "
                                 f"Tried to read it as format '{detected_time_format}'.") from e

        return seconds
