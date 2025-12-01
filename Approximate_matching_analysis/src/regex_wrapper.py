"""Wrapper around Python built-in regex for baseline comparisons.

Provides exact `re`-based search functions.
"""
import re
from typing import List, Tuple


def regex_search(text: str, pattern: str) -> List[Tuple[int, int]]:
    """Return list of (end_index, 0) for exact matches found by `re`.

    end_index is index of last character matched.
    """
    matches = []
    try:
        for m in re.finditer(re.escape(pattern), text):
            matches.append((m.end() - 1, 0))
    except re.error:
        return []
    return matches


if __name__ == "__main__":
    print(regex_search("hello world hello", "hello"))
