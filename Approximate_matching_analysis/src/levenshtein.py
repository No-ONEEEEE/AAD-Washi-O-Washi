"""Levenshtein functions and a Ukkonen banded search.

Functions:
- levenshtein_distance(a, b): full DP distance
- ukkonen_search(text, pattern, max_dist): returns list of (end_pos, dist)
"""
from typing import List, Tuple


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def levenshtein_search(text: str, pattern: str, max_dist: int) -> List[Tuple[int, int]]:
    """Return list of (end_index, distance) with distance <= max_dist.

    This uses a banded dynamic programming window (Ukkonen's algorithm) for efficiency.
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return [(i, 0) for i in range(n)]

    matches = []
    # For each possible alignment ending at position i in text, compute banded DP
    for end in range(1, n + 1):
        start = max(0, end - m - max_dist)
        window = text[start:end]
        # compute distance between pattern and window suffixes but we allow partial
        d = _banded_distance(pattern, window, max_dist)
        if d is not None and d <= max_dist:
            matches.append((end - 1, d))
    return matches


def _banded_distance(p: str, t: str, k: int):
    # returns distance if <= k else None
    m = len(p)
    n = len(t)
    if abs(m - n) > k:
        return None
    INF = k + 1
    prev = [j if j <= k else INF for j in range(n + 1)]
    for i in range(1, m + 1):
        curr = [INF] * (n + 1)
        # band limits
        from_j = max(1, i - k)
        to_j = min(n, i + k)
        if from_j > 1:
            curr[from_j - 1] = INF
        for j in range(from_j, to_j + 1):
            cost = 0 if p[i - 1] == t[j - 1] else 1
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + cost
            curr[j] = min(ins, delete, sub)
        prev = curr
    res = prev[n]
    return res if res <= k else None


if __name__ == "__main__":
    print(levenshtein_distance("kitten", "sitting"))
