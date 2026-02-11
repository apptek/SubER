import unittest

from suber.tokenizers import _reattach_punctuation, detokenize_segments, reversibly_tokenize_segments

from .utilities import create_temporary_file_and_read_it


class ReversibleTokenizationTests(unittest.TestCase):
    def test_reversible_tokenization(self):
        example_srt = """
1
00:00:00,000 --> 00:00:01,380
入れ墨入れたの？

2
00:00:01,380 --> 00:00:03,000
入れ墨入れたよ

3
00:00:03,020 --> 00:00:04,000
新品同様か？

4
00:00:04,000 --> 00:00:05,240
うん

5
00:00:05,240 --> 00:00:06,700
なんでそんな顔してるの？

6
00:00:06,700 --> 00:00:07,960
ユーチューブの皆さん
どうも！

7
00:00:07,960 --> 00:00:09,180
ぼくはジェイソン

8
00:00:09,180 --> 00:00:10,640
二年目の医学生です

9
00:00:10,640 --> 00:00:11,680
ようこそ ぼくのチャンネル

10
00:00:11,680 --> 00:00:12,740
「信じて見据えよう」へ

11
00:00:12,740 --> 00:00:14,500
簡単に背景を説明すると

12
00:00:14,500 --> 00:00:17,640
ずっと胸の入れ墨を入れたい
と思っていて

13
00:00:17,640 --> 00:00:20,020
下調べとかもしてきたんだ

14
00:00:20,020 --> 00:00:24,240
でも頭の中では
入れ墨が全くない状態から

15
00:00:24,240 --> 00:00:26,040
いきなり胸に入れ墨を入れる
つもりはなかったんだ

16
00:00:26,040 --> 00:00:27,240
それでぼくがしたことは

17
00:00:27,240 --> 00:00:29,960
胸に貼るステッカー式の
入れ墨を買ったんだ

18
00:00:29,960 --> 00:00:32,960
胸の入れ墨がどう見えるか
確かめるために

19
00:00:32,960 --> 00:00:35,300
問題は
ぼくが両親に電話して

20
00:00:35,300 --> 00:00:37,160
そのステッカーの入れ墨が
"""
        subtitles = create_temporary_file_and_read_it(example_srt)

        tokenized_subtitles = reversibly_tokenize_segments(subtitles, language="ja", keep_punctuation_attached=False)
        tokenized_subtitles_punct_attached = reversibly_tokenize_segments(
            subtitles, language="ja", keep_punctuation_attached=True)

        num_words = sum(len(subtitle.word_list) for subtitle in subtitles)
        num_tokens = sum(len(subtitle.word_list) for subtitle in tokenized_subtitles)
        num_tokens_punct_attached = sum(len(subtitle.word_list) for subtitle in tokenized_subtitles_punct_attached)

        self.assertTrue(num_words < num_tokens_punct_attached < num_tokens)

        detokenize_subtitles = detokenize_segments(tokenized_subtitles)
        self.assertEqual(subtitles, detokenize_subtitles)
        detokenize_subtitles = detokenize_segments(tokenized_subtitles_punct_attached)
        self.assertEqual(subtitles, detokenize_subtitles)

    def test_reattach_punctuation(self):
        self.assertEqual(_reattach_punctuation("No punctuation"), "No punctuation")
        self.assertEqual(_reattach_punctuation("シンプル な 句読点 。"), "シンプル な 句読点。")
        self.assertEqual(_reattach_punctuation("¿ Esto funciona ?"), "¿Esto funciona?")
        self.assertEqual(_reattach_punctuation("アルバート ・ アインシュタイン"), "アルバート・ アインシュタイン")
        self.assertEqual(_reattach_punctuation("Multiple . .. ... tokens"), "Multiple...... tokens")
        self.assertEqual(_reattach_punctuation(". .. ... Multiple tokens"), "......Multiple tokens")
        self.assertEqual(_reattach_punctuation("Multiple tokens . .. ..."), "Multiple tokens......")


if __name__ == '__main__':
    unittest.main()
