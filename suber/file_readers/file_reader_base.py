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
            return gzip.open(self._file_name, 'rt')
        else:
            return open(self._file_name, 'r')
