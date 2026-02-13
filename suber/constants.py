# As used in MuST-Cinema corpus: https://ict.fbk.eu/must-cinema/
END_OF_LINE_SYMBOL = "<eol>"
END_OF_BLOCK_SYMBOL = "<eob>"

MASK_SYMBOL = "<mask>"

# These are the languages for which we enable "asian_support" for TER computation.
# TODO: Korean included as a precaution, does it make sense? "asian_support=True" should only have an effect in very
# rare cases for Korean text?
# For SubER and WER we actually use sacrebleu's TokenizerZh, TokenizerJaMecab, and TokenizerKoMecab instead of
# TercomTokenizer with "asian_support".
EAST_ASIAN_LANGUAGE_CODES = ["zh", "ja", "ko"]

SPACE_ESCAPE = "▁"
