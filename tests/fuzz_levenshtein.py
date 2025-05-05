#!/usr/bin/env python3
"""
Checks that our altered Levenshtein implementation, which preserves behavior of Levenshtein v0.18.0 in terms of
edit ops, returns the same edit distance as whichever (newer) Levenshtein is installed in your environment.
Must match, ambiguity exists only in the alignment / edit ops, not the number of edit ops.
"""

import random
import string

from rapidfuzz.distance import Levenshtein

from suber import lib_levenshtein


for i in range(100000):
    N1 = random.randint(0, 40)
    N2 = random.randint(0, 40)

    s1 = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N1))
    s2 = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N2))

    distance_rapidfuzz = Levenshtein.distance(s1, s2)
    distance = lib_levenshtein.distance(s1, s2)
    assert distance == distance_rapidfuzz, (s1, s2, distance_levenshtein, distance_opcodes, i)
    num_editops = len(lib_levenshtein.editops(s1, s2))
    assert distance == num_editops, (s1, s2, distance_levenshtein, distance_opcodes, i)
