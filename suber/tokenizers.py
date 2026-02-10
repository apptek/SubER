from typing import Callable

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_ter import TercomTokenizer
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


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
