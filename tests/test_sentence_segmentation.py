import unittest

from suber.sentence_segmentation import resegment_based_on_punctuation
from .utilities import create_temporary_file_and_read_it


class SentenceSegmentationTests(unittest.TestCase):

    def test_sentence_segmentation(self):
        file_content = """This is a first sentence. This is
                          another one… That is a question? 'That is a quoted
                          exclamation!' Ellipsis... that continues. even further. 1. sentence
                          starting with a number... 2. one."""

        segments = create_temporary_file_and_read_it(file_content, file_format="plain")

        sentences = resegment_based_on_punctuation(segments)

        self.assertEqual(len(sentences), 6)

        expected_sentences = [
            "This is a first sentence.",
            "This is another one…",
            "That is a question?",
            "'That is a quoted exclamation!'",
            "Ellipsis... that continues. even further.",
            "1. sentence starting with a number... 2. one."]

        for index in range(len(sentences)):
            sentence_text = " ".join(word.string for word in sentences[index].word_list)
            self.assertEqual(sentence_text, expected_sentences[index])


if __name__ == '__main__':
    unittest.main()
