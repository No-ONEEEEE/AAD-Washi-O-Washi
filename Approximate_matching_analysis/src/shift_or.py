"""Myers' bit-parallel (Shift-Or style) approximate search.

Functions:
- myers_search(text, pattern, max_dist): returns list of (end_pos, dist)

This implementation follows the bit-parallel algorithm described by Myers.
"""
from typing import List, Tuple


def _build_peq(pattern: str) -> dict:
    peq = {}
    for i, ch in enumerate(pattern):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    return peq


def shiftor_search(text: str, pattern: str, max_dist: int) -> List[Tuple[int, int]]:
    """Return list of (end_index, distance) where matches have distance <= max_dist.

    end_index is the index in `text` of the last character of the best-aligned match.
    """
    if pattern == "":
        return [(i, 0) for i in range(len(text))]

    m = len(pattern)
    n = len(text)
    if m == 0:
        return []

    peq = _build_peq(pattern)

    Peq_default = 0
    Pv = ~0
    Mv = 0
    Matches = []

    highest_bit = 1 << (m - 1)

    for i, ch in enumerate(text):
        Eq = peq.get(ch, Peq_default)

        Xv = Eq | Mv
        Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq
        Ph = Mv | ~(Xh | Pv)
        Mh = Pv & Xh

        # update Pv and Mv for next iteration
        # left shift by 1; Python ints are unbounded so mask isn't necessary
        Pv = (Mh << 1) | ~(Xv | ((Ph << 1) | 1))
        Mv = ((Ph << 1) | 1) & Xv

        # compute current score
        # The high bit of Pv/Mv encodes the sign of the difference
        if Pv & highest_bit:
            curr_dist = ((~Pv) & ((1 << m) - 1)).bit_count()
        else:
            curr_dist = (Mv & ((1 << m) - 1)).bit_count()

        # a conservative check: compute exact edit distance up to max_dist if candidate
        if curr_dist <= max_dist:
            # accept match ending at i with distance curr_dist
            Matches.append((i, curr_dist))

    return Matches


if __name__ == "__main__":
    # small demo
    text = "This is a simple example where we search for sample patterns."
    pattern = "sample"
    print(shiftor_search(text, pattern, 1))
