from typing import List
from suber.data_types import Segment


sentence_final_punctuation = ['.', '?', '!', '！', '？', '。', "…"]
quotation_marks = ["'", '"']
# all combinations of punctuation and quotes, i.e. '."', "?'" etc.
quoted_sentence_final_punctuation = [punct + quote for punct in sentence_final_punctuation for quote in quotation_marks]
ellipses = ["...", "…"]


def resegment_based_on_punctuation(segments: List[Segment]) -> List[Segment]:
    resegmented_segments = []

    all_words = [word for segment in segments for word in segment.word_list]

    word_list = []
    previous_word = None
    for word in all_words:
        if not previous_word or _is_sentence_end(previous_word.string, word.string):
            if word_list:
                resegmented_segments.append(Segment(word_list=word_list))
            word_list = [word]
        else:
            word_list.append(word)
        previous_word = word

    assert word_list
    resegmented_segments.append(Segment(word_list=word_list))

    return resegmented_segments


def _is_sentence_end(current_word: str, next_word: str = None):
    if not next_word:
        # No next word, force sentence end.
        return True

    assert current_word, "'current_word' must not be empty."

    is_sentence_final_punctuation_at_end = (
        current_word[-1] in sentence_final_punctuation
        or (len(current_word) > 1 and current_word[-2:] in quoted_sentence_final_punctuation))

    is_ellipsis_at_end = any(current_word.endswith(ellipsis) for ellipsis in ellipses)

    next_word_is_lower_cased = next_word[0].islower()
    next_word_is_lower_or_digit = next_word_is_lower_cased or next_word[0].isdigit()

    is_sentence_end = (is_sentence_final_punctuation_at_end and not next_word_is_lower_cased
                       and not (is_ellipsis_at_end and next_word_is_lower_or_digit))

    return is_sentence_end
