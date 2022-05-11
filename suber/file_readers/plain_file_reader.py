from suber.file_readers.file_reader_base import FileReaderBase
from suber.data_types import LineBreak, Word, Segment
from suber.constants import END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL


class PlainFileReader(FileReaderBase):
    def _parse_lines(self, file_object):
        segments = []

        is_first_line = True
        for line in file_object:
            if is_first_line:
                if line.startswith('\ufeff'):
                    line = line[len('\ufeff'):]  # remove byte order mark (BOM)
                is_first_line = False
            words = line.split()

            word_list = []
            for word in words:
                if word in (END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL):
                    if not word_list:
                        continue  # ignore line break symbol at the start of the line
                    else:
                        word_list[-1].line_break = (
                            LineBreak.END_OF_BLOCK if word == END_OF_BLOCK_SYMBOL else LineBreak.END_OF_LINE)
                else:
                    word_list.append(Word(string=word))

            segments.append(Segment(word_list=word_list))

        return segments
