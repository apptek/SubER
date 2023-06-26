import gzip

from typing import List
from io import TextIOWrapper

from suber.data_types import Segment


class FileReaderBase:
    """
    Derived classes must implement self._parse_lines().
    """
    def __init__(self, file_name):
        self._file_name = file_name

    def read(self) -> List[Segment]:
        with self._open_file() as file_object:
            return list(self._parse_lines(file_object))

    def _parse_lines(self, file_object: TextIOWrapper) -> List[Segment]:
        raise NotImplementedError

    def _open_file(self):
        if self._file_name.endswith(".gz"):
            return gzip.open(self._file_name, "rt", encoding="utf-8")
        else:
            return open(self._file_name, "r", encoding="utf-8")


def read_input_file(file_name, file_format) -> List[Segment]:
    from suber.file_readers import PlainFileReader, SRTFileReader  # here to avoid circular import

    if file_format == "SRT":
        file_reader = SRTFileReader(file_name)
    elif file_format == "plain":
        file_reader = PlainFileReader(file_name)
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    try:
        segments = file_reader.read()
    except Exception as e:
        raise Exception(f"Error reading file '{file_name}'") from e

    return segments
