def num_cmp(S: int, l: int, d: int) -> int:
    return 0 if S < l else (S - l) // d + 1


def reads(S: int, l: int, d: int, n: int, l_sel: int, w: int) -> int:
    return num_cmp(S, l, d) + n * l_sel + min(w, S)


def test_decode_reads_formula_early_and_normal():
    l, d, l_sel, n, w = 32, 16, 64, 16, 512
    for S in [0, 1, 15, 16, 31, 32, 33, 63, 64, 1000]:
        expected = reads(S, l, d, n, l_sel, w)
        # trivial invariants
        assert expected >= 0
        if S < l:
            assert num_cmp(S, l, d) == 0
        if S < w:
            assert expected == num_cmp(S, l, d) + n * l_sel + S
