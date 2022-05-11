from suber.data_types import LineBreak, Segment
from suber.constants import END_OF_LINE_SYMBOL, END_OF_BLOCK_SYMBOL, MASK_SYMBOL


def segment_to_string(segment: Segment, include_line_breaks=False, include_last_break=True,
                      mask_all_words=False) -> str:
    if not include_line_breaks:
        assert not mask_all_words, (
            "Refusing to mask all words when not printing breaks, output would contain only mask symbols.")
        return " ".join(word.string for word in segment.word_list)

    word_list_with_breaks = []
    for word in segment.word_list:
        word_list_with_breaks.append(MASK_SYMBOL if mask_all_words else word.string)

        if word.line_break == LineBreak.END_OF_LINE:
            word_list_with_breaks.append(END_OF_LINE_SYMBOL)
        elif word.line_break == LineBreak.END_OF_BLOCK:
            word_list_with_breaks.append(END_OF_BLOCK_SYMBOL)

    if not include_last_break and word_list_with_breaks and word_list_with_breaks[-1] == END_OF_BLOCK_SYMBOL:
        word_list_with_breaks.pop()

    return " ".join(word_list_with_breaks)


def get_segment_to_string_opts_from_metric(metric: str):
    include_breaks = False
    mask_words = False
    if metric.endswith("-br"):
        include_breaks = True
        mask_words = True
        metric = metric[:-len("-br")]
    elif metric.endswith("-seg"):
        include_breaks = True
        metric = metric[:-len("-seg")]

    return include_breaks, mask_words, metric
