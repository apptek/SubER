import regex
from typing import Callable, List, Optional

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_ter import TercomTokenizer
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

from suber.constants import SPACE_ESCAPE
from suber.data_types import LineBreak, Segment, Subtitle, Word, TimedWord
from suber.utilities import set_approximate_word_times


def get_sacrebleu_tokenizer(language: str, default_to_tercom: bool = False) -> Callable[[str], str]:
    """
    Returns the default tokenizer as used by sacrebleu for BLEU calculation. If 'default_to_tercom' is set, will return
    (case-sensitive) TercomTokenizer instead, if language is not "ja", "ko", "zh". The reasoning is that for TER-based
    metrics we want to stay close to the original implementation, also we want to keep the behavior of our original
    implementation which always used TercomTokenizer. But the "asian_support" of TercomTokenizer is questionable,
    especially that for Japanese sequences of Hiragana and Katakana characters are never split. So for those languages
    we switch to the dedicated default BLEU tokenizers.
    """
    if language == "ja":
        tokenizer = TokenizerJaMecab()
    elif language == "ko":
        # Import only here to keep compatible with sacrebleu versions < 2.2 for all other languages.
        from sacrebleu.tokenizers.tokenizer_ko_mecab import TokenizerKoMecab

        tokenizer = TokenizerKoMecab()
    elif language == "zh":
        tokenizer = TokenizerZh()
    elif not default_to_tercom:
        tokenizer = Tokenizer13a()
    else:
        tokenizer = TercomTokenizer(normalized=True, no_punct=False, case_sensitive=True)

    return tokenizer


def reversibly_tokenize_segments(
        segments: List[Segment], language: str, keep_punctuation_attached: bool = False) -> List[Segment]:
    """
    For each segment, splits words by applying the tokenizer function to the Word.string attributes. Uses a "▁" prefix
    to represent the original word boundaries which are positions of spaces (similar to SentencePiece). If the input
    Segments are Subtitles, the output will also be Subtitles. If the input contains TimedWords, the output will too.
    For that, subtitle timings are carried over and approximate word times are recomputed.
    If 'keep_punctuation_attached' is set, do not split off tokens from (space-separated) input words which would
    consist of only punctuation. Most useful for Japanese / Chinese to run word segmentation without creating extra
    punctuation tokens.
    """

    if keep_punctuation_attached:
        tokenizer = lambda string: _reattach_punctuation(get_sacrebleu_tokenizer(language)(string))
    else:
        tokenizer = get_sacrebleu_tokenizer(language)

    tokenized_segments = []
    words_are_timed = None

    for segment in segments:
        tokenized_word_list = []

        for word in segment.word_list:
            assert word, "Words must not be empty."

            tokens = tokenizer(word.string).split()
            assert tokens, "Tokenizer deleted word."

            for token_index, token in enumerate(tokens):
                # Prefix the first token to mark original space.
                if token_index == 0:
                    token = SPACE_ESCAPE + token

                # Only the last token inherits the original line break.
                line_break = word.line_break if token_index == len(tokens) - 1 else LineBreak.NONE

                if isinstance(word, TimedWord):
                    assert words_are_timed is None or words_are_timed, "Either all or no words must be timed."
                    words_are_timed = True

                    tokenized_word_list.append(
                        TimedWord(string=token, line_break=line_break,
                                  subtitle_start_time=segment.start_time, subtitle_end_time=segment.end_time))
                else:
                    assert not words_are_timed, "Either all or no words must be timed."
                    words_are_timed = False
                    tokenized_word_list.append(Word(string=token, line_break=line_break))

        if isinstance(segment, Subtitle):
            if words_are_timed:
                set_approximate_word_times(tokenized_word_list, segment.start_time, segment.end_time)
            tokenized_segment = Subtitle(word_list=tokenized_word_list, index=segment.index,
                                         start_time=segment.start_time, end_time=segment.end_time)
        else:
            tokenized_segment = Segment(word_list=tokenized_word_list)

        tokenized_segments.append(tokenized_segment)

    return tokenized_segments


def detokenize_segments(segments: List[Segment]) -> List[Segment]:
    """
    Inverse of 'reversibly_tokenize_segments()'.
    """

    def add_word(current_tokens: List[str], detokenized_word_list: List[Word], words_are_timed: bool,
                 current_line_break: LineBreak, current_subtitle_start_time: Optional[float],
                 current_subtitle_end_time: Optional[float]):
        """
        Helper function. Joins 'current_tokens' into a word and appends to 'detokenized_word_list'. Clears
        'current_tokens' afterwards.
        """
        if not current_tokens:
            return

        detokenized_word_string = "".join(current_tokens)
        if words_are_timed:
            detokenized_word = TimedWord(
                string=detokenized_word_string, line_break=current_line_break,
                subtitle_start_time=current_subtitle_start_time, subtitle_end_time=current_subtitle_end_time)
        else:
            detokenized_word = Word(string=detokenized_word_string, line_break=current_line_break)
        detokenized_word_list.append(detokenized_word)
        current_tokens.clear()

    detokenized_segments = []
    words_are_timed = None

    for segment in segments:
        detokenized_word_list = []

        current_tokens = []
        current_line_break = LineBreak.NONE
        current_subtitle_start_time = None  # Taken from TimedWord.subtitle_start/end_time. All tokens originating from
        current_subtitle_end_time = None  # a given word should have identical timings, we don't check this here.

        for token in segment.word_list:
            if isinstance(token, TimedWord):
                assert words_are_timed is None or words_are_timed, "Either all or no words must be timed."
                words_are_timed = True
            else:
                assert not words_are_timed, "Either all or no words must be timed."
                words_are_timed = False

            token_string = token.string

            if token_string.startswith(SPACE_ESCAPE):
                assert len(token_string) > 1, "Space escape character should not appear as separate word."
                token_string = token_string[1:]  # strip space escape character

                # Flush the previous word if there is one.
                add_word(current_tokens, detokenized_word_list, words_are_timed, current_line_break,
                         current_subtitle_start_time, current_subtitle_end_time)

            current_tokens.append(token_string)
            current_line_break = token.line_break
            if words_are_timed:
                current_subtitle_start_time = token.subtitle_start_time
                current_subtitle_end_time = token.subtitle_end_time

        # Flush remaining tokens.
        add_word(current_tokens, detokenized_word_list, words_are_timed, current_line_break,
                 current_subtitle_start_time, current_subtitle_end_time)

        if isinstance(segment, Subtitle):
            if words_are_timed:
                set_approximate_word_times(detokenized_word_list, segment.start_time, segment.end_time)
            detokenized_segment = Subtitle(word_list=detokenized_word_list, index=segment.index,
                                           start_time=segment.start_time, end_time=segment.end_time)
        else:
            detokenized_segment = Segment(word_list=detokenized_word_list)

        detokenized_segments.append(detokenized_segment)

    return detokenized_segments


def _reattach_punctuation(word: str) -> str:
    """
    To be used on a 'word' string that is the result of applying a tokenizer to a single (space-separated) word.
    'word' is therefore expected to contain spaces that split the word into tokens. This function removes spaces such
    that tokens consisting of punctuation characters only get attached to the token to their left, except for leading
    punctuation which gets attached right.
    """
    word = regex.sub(r" (\p{P}+)(?= |$)", r"\1", word)
    # Now the only possible punctuation token remaining should be at start of the string, remove space after it.
    word = regex.sub(r"^(\p{P}+) ", r"\1", word)
    return word
