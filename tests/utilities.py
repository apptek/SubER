import tempfile
from suber.file_readers import PlainFileReader, SRTFileReader


def create_temporary_file_and_read_it(file_content, file_format="SRT"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt") as temporary_file:
        temporary_file.write(file_content)
        temporary_file.flush()

        if file_format == "SRT":
            file_reader = SRTFileReader(temporary_file.name)
        elif file_format == "plain":
            file_reader = PlainFileReader(temporary_file.name)
        else:
            raise ValueError(f"Invalid file format '{file_format}'")

        segments = file_reader.read()

        return segments
