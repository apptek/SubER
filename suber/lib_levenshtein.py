# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

# This code was taken from https://github.com/rapidfuzz/RapidFuzz/blob/main/src/rapidfuzz/distance/Levenshtein_py.py
# https://github.com/rapidfuzz/RapidFuzz/blob/main/src/rapidfuzz/_common_py.py
# and https://github.com/rapidfuzz/RapidFuzz/blob/main/src/rapidfuzz/distance/_initialize_py.py
# and altered to recover the exact behavior of python-Levenshtein v0.12.0, which was our original Levenshtein
# dependency but does not support Python > 3.10. In general, there are several possible alignments resulting in minimal
# Levenshtein distance, and the choice of a particular alignment changed when (python-)Levenshtein started using the
# rapidfuzz implementation in v0.18.0. Upgrading python-Levenshtein would therefore result in slightly different scores
# for the "AS-" metrics on our end. For now, we want perfect backwards compatibility and therefore integrate our own
# version of the Levenshtein code here.


def _matrix(s1, s2):
    if not s1:
        return (len(s2), [], [])

    VP = (1 << len(s1)) - 1
    VN = 0
    currDist = len(s1)
    mask = 1 << (len(s1) - 1)

    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    matrix_VP = []
    matrix_VN = []
    for ch2 in s2:
        # Step 1: Computing D0
        PM_j = block_get(ch2, 0)
        X = PM_j
        D0 = (((X & VP) + VP) ^ VP) | X | VN
        # Step 2: Computing HP and HN
        HP = VN | ~(D0 | VP)
        HN = D0 & VP
        # Step 3: Computing the value D[m,j]
        currDist += (HP & mask) != 0
        currDist -= (HN & mask) != 0
        # Step 4: Computing Vp and VN
        HP = (HP << 1) | 1
        HN = HN << 1
        VP = HN | ~(D0 | HP)
        VN = HP & D0

        matrix_VP.append(VP)
        matrix_VN.append(VN)

    return (currDist, matrix_VP, matrix_VN)


def distance(s1, s2):
    prefix_len, suffix_len = common_affix(s1, s2)
    s1 = s1[prefix_len : len(s1) - suffix_len]
    s2 = s2[prefix_len : len(s2) - suffix_len]
    dist, _, _ = _matrix(s1, s2)
    return dist


def editops(s1, s2):
    """
    Creates editops from the output of the bit-parallel rapidfuzz implementation above (edit distance matrix expressed
    as delta vectors), but makes the exact choices in case of ties as the original python-Levenshtein code:
    https://github.com/rapidfuzz/Levenshtein/blob/v0.17.0/src/Levenshtein-c/_levenshtein.c#L3205
    The rapidfuzz implementation prefers "insert", as the decisions can be made efficiently using the delta vectors
    in that case, see
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=339dec563792acb5bb2feffc53628b62bdc36329
    To prefer "replace" (among other differences) we need to re-calculate the actual elements of the distance matrix
    from the delta vectors, which kind of defeats the purpose as it makes the algorithm less efficient. But here we care
    more about backwards compatibility than efficiency.
    """
    prefix_len, suffix_len = common_affix(s1, s2)
    s1 = s1[prefix_len : len(s1) - suffix_len]
    s2 = s2[prefix_len : len(s2) - suffix_len]
    dist, VP, VN = _matrix(s1, s2)

    if dist == 0:
        return []

    editop_list = [None] * dist
    col = len(s1)
    row = len(s2)
    direction = 0

    while row != 0 and col != 0:
        masked_VP = VP[row - 1] & ((1 << col) - 1)
        masked_VN = VN[row - 1] & ((1 << col) - 1)
        current_distance = masked_VP.bit_count() - masked_VN.bit_count() + row

        masked_VP = VP[row - 1] & ((1 << (col - 1)) - 1)
        masked_VN = VN[row - 1] & ((1 << (col - 1)) - 1)
        deletion_distance = masked_VP.bit_count() - masked_VN.bit_count() + row

        if row > 1:
            masked_VP = VP[row - 2] & ((1 << (col - 1)) - 1)
            masked_VN = VN[row - 2] & ((1 << (col - 1)) - 1)
            replace_distance = masked_VP.bit_count() - masked_VN.bit_count() + row - 1

            masked_VP = VP[row - 2] & ((1 << col) - 1)
            masked_VN = VN[row - 2] & ((1 << col) - 1)
            insertion_distance = masked_VP.bit_count() - masked_VN.bit_count() + row - 1

        else:
            replace_distance = col - 1
            insertion_distance = col

        if direction == -1 and current_distance == insertion_distance + 1:
            dist -= 1
            row -= 1
            direction = -1
            editop_list[dist] = ("insert", col + prefix_len, row + prefix_len)

        elif direction == 1 and current_distance == deletion_distance + 1:
            dist -= 1
            col -= 1
            direction = 1
            editop_list[dist] = ("delete", col + prefix_len, row + prefix_len)

        elif current_distance == replace_distance and s1[col - 1] == s2[row - 1]:
            col -= 1
            row -= 1
            direction = 0

        elif current_distance == replace_distance + 1:
            col -= 1
            row -= 1
            dist -= 1
            direction = 0
            editop_list[dist] = ("replace", col + prefix_len, row + prefix_len)

        elif direction == 0 and current_distance == insertion_distance + 1:
            dist -= 1
            row -= 1
            direction = -1
            editop_list[dist] = ("insert", col + prefix_len, row + prefix_len)

        elif direction == 0 and current_distance == deletion_distance + 1:
            dist -= 1
            col -= 1
            direction = 1
            editop_list[dist] = ("delete", col + prefix_len, row + prefix_len)

        else:
            assert False, "Bug while back-tracing cost matrix."

        assert dist >= 0, "Bug: distance differs from number of edit ops computed during back-tracing."

    while col != 0:
        dist -= 1
        col -= 1
        editop_list[dist] = ("delete", col + prefix_len, row + prefix_len)

    while row != 0:
        dist -= 1
        row -= 1
        editop_list[dist] = ("insert", col + prefix_len, row + prefix_len)

    assert dist == 0, "Bug: distance differs from number of edit ops computed during back-tracing."
    return editop_list


def opcodes(s1, s2):
    editops_ = editops(s1, s2)

    src_len = len(s1)
    dest_len = len(s2)

    blocks = []
    src_pos = 0
    dest_pos = 0
    i = 0
    while i < len(editops_):
        if src_pos < editops_[i][1] or dest_pos < editops_[i][2]:
            blocks.append(
                (
                    "equal",
                    src_pos,
                    editops_[i][1],
                    dest_pos,
                    editops_[i][2],
                )
            )
            src_pos = editops_[i][1]
            dest_pos = editops_[i][2]

        src_begin = src_pos
        dest_begin = dest_pos
        tag = editops_[i][0]
        while (
            i < len(editops_)
            and editops_[i][0] == tag
            and src_pos == editops_[i][1]
            and dest_pos == editops_[i][2]
        ):
            if tag == "replace":
                src_pos += 1
                dest_pos += 1
            elif tag == "insert":
                dest_pos += 1
            elif tag == "delete":
                src_pos += 1

            i += 1

        blocks.append((tag, src_begin, src_pos, dest_begin, dest_pos))

    if src_pos < src_len or dest_pos < dest_len:
        blocks.append(("equal", src_pos, src_len, dest_pos, dest_len))

    return blocks


def common_prefix(s1: str, s2: str) -> int:
    prefix_len = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2:
            break

        prefix_len += 1

    return prefix_len


def common_suffix(s1: str, s2: str) -> int:
    suffix_len = 0
    for ch1, ch2 in zip(reversed(s1), reversed(s2)):
        if ch1 != ch2:
            break

        suffix_len += 1

    return suffix_len


def common_affix(s1: str, s2: str) -> tuple[int, int]:
    prefix_len = common_prefix(s1, s2)
    suffix_len = common_suffix(s1[prefix_len:], s2[prefix_len:])
    return (prefix_len, suffix_len)