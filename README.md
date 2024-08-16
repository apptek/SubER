# SubER - Subtitle Edit Rate

SubER is an automatic, reference-based, segmentation- and timing-aware edit distance metric to measure quality of subtitle files.
For a detailed description of the metric and a human post-editing evaluation we refer to our [IWSLT 2022 paper](https://aclanthology.org/2022.iwslt-1.1.pdf).
In addition to the SubER metric, this scoring tool calculates a wide range of established speech recognition and machine translation metrics (WER, BLEU, TER, chrF) directly on subtitle files.

## Installation
```console
pip install subtitle-edit-rate
```
will install the `suber` command line tool.
Alternatively, check out this git repository and run the contained `suber` module with `python -m suber`.

## Basic Usage
Currently, we expect subtitle files to come in [SubRip text (SRT)](https://en.wikipedia.org/wiki/SubRip) format. Given a human reference subtitle file `reference.srt` and a hypothesis file `hypothesis.srt` (typically the output of an automatic subtitling system) the SubER score can be calculated by running:

```console
$ suber -H hypothesis.srt -R reference.srt
{
    "SubER": 19.048
}
```
The SubER score is printed to stdout in json format. As SubER is an edit rate, lower scores are better. As a rough rule of thumb from our experience, a score lower than 20(%) is very good quality while a score above 40 to 50(%) is bad.

Make sure that there is no constant time offset between the timestamps in hypothesis and reference as this will lead to incorrect scores.
Also, note that `<i>`, `<b>` and `<u>` formatting tags are ignored if present in the files. All other formatting must be removed from the files before scoring for accurate results.

#### Punctuation and Case-Sensitivity
The main SubER metric is computed on normalized text, which means case-insensitive and without taking punctuation into account, as we observe higher correlation with human judgements and post-edit effort in this setting. We provide an implementation of a case-sensitive variant which also uses a tokenizer to take punctuation into account as separate tokens which you can use "at your own risk" or to reassess our findings. For this, add `--metrics SubER-cased` to the command above. Please do not report results using this variant as "SubER" unless explicitly mentioning the punctuation-/case-sensitivity.

## Other Metrics
The SubER tool supports computing the following other metrics directly on subtitle files:

- word error rate (WER)
- bilingual evaluation understudy (BLEU)
- translation edit rate (TER)
- character n-gram F score (chrF)
- character error rate (CER)

BLEU, TER and chrF calculations are done using [SacreBLEU](https://github.com/mjpost/sacrebleu) with default settings. WER is computed with [JiWER](https://github.com/jitsi/jiwer) on normalized text (lower-cased, punctuation removed).

__Assuming__ `hypothesis.srt` __and__ `reference.srt` __are parallel__, i.e. they contain the same number of subtitles and the contents of the _n_-th subtitle in both files corresponds to each other, the above-mentioned metrics can be computed by running:
```console
$ suber -H hypothesis.srt -R reference.srt --metrics WER BLEU TER chrF CER
{
    "WER": 23.529,
    "BLEU": 39.774,
    "TER": 23.529,
    "chrF": 68.402,
    "CER": 17.857
}
```
In this mode, the text from each parallel subtitle pair is considered to be a sentence pair.

### Scoring Non-Parallel Subtitle Files
In the general case, subtitle files for the same video can have different numbers of subtitles with different time stamps. All metrics - except SubER - usually require to be calculated on parallel segments. To apply these metrics to general subtitle files, the hypothesis file has to be re-segmented to correspond to the reference subtitles. The SubER tool implements two options:

- alignment by minimizing Levenshtein distance ([Matusov et al.](https://aclanthology.org/2005.iwslt-1.19.pdf))
- time alignment method from [Cherry et al.](https://www.isca-archive.org/interspeech_2021/cherry21_interspeech.pdf)

See our [paper](https://aclanthology.org/2022.iwslt-1.1.pdf) for further details.

To use the Levenshtein method add an `AS-` prefix to the metric name, e.g.:
```console
suber -H hypothesis.srt -R reference.srt --metrics AS-BLEU
```
The `AS-` prefix terminology is taken from [Matusov et al.](https://aclanthology.org/2005.iwslt-1.19.pdf) and stands for "automatic segmentation".
To use the time-alignment method instead, add a `t-` prefix. This works for all metrics (except for SubER itself which does not require re-segmentation). In particular, we implement `t-BLEU` from [Cherry et al.](https://www.isca-archive.org/interspeech_2021/cherry21_interspeech.pdf). We encode the segmentation method (or lack thereof) in the metric name to explicitly distinguish the different resulting metric scores!

To inspect the re-segmentation applied to the hypothesis you can use the `align_hyp_to_ref.py` tool (run `python -m suber.tools.align_hyp_to_ref -h` for help).

In case of Levenshtein alignment, there is also the option to give a plain file as the reference. This can be used to provide sentences instead of subtitles as reference segments (each line will be considered a segment):

```console
suber -H hypothesis.srt -R reference.txt --reference-format plain --metrics AS-TER
```

We provide a simple tool to extract sentences from SRT files based on punctuation:

```console
python -m suber.tools.srt_to_plain -i reference.srt -o reference.txt --sentence-segmentation
```

It can be used to create the plain sentence-level reference `reference.txt` for the scoring command above.

### Scoring Line Breaks as Tokens
The line breaks present in the subtitle files can be included into the text segments to be scored as `<eol>` (end of line) and `<eob>` (end of block) tokens. For example:

```
636
00:50:52,200 -> 00:50:57,120
Ladies and gentlemen,
the dance is about to begin.
```
would be represented as
```
Ladies and gentlemen, <eol> the dance is about to begin. <eob>
```
To do so, add a `-seg` ("segmentation-aware") postfix to the metric name, e.g. `BLEU-seg`, `AS-TER-seg` or `t-WER-seg`. Character-level metrics (chrF and CER) do not support this as it is not obvious how to count character edits for `<eol>` tokens.

### TER-br
As a special case, we implement TER-br from [Karakanta et al.](https://aclanthology.org/2020.iwslt-1.26.pdf). It is similar to `TER-seg`, but all (real) words are replaced by a mask token. This would convert the sentence from the example above to:
```
<mask> <mask> <mask> <eol> <mask> <mask> <mask> <mask> <mask> <mask> <eob>
```
Note, that also TER-br has variants for computing it on existing parallel segments (`TER-br`) or on re-aligned segments (`AS-TER-br`/`t-TER-br`). Re-segmentation happens before masking.

## Contributing
If you run into an issue, have a feature request or have questions about the usage or the implementation of SubER, please do not hesitate to open an issue or a thread under "discussions". Pull requests are welcome too, of course!

Things I'm already considering to add in future versions:
- support for other subtitling formats than SRT
- a verbose output that explains the SubER score (list of edit operations)

