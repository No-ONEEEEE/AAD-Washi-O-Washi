"""
Washi O Washi - String and Pattern Matching Algorithms Package

This package contains implementations of various string and pattern matching algorithms
for exact and approximate matching, with applications in bioinformatics and genomics.
"""

__version__ = "1.0.0"
__authors__ = ["Yogansh", "Chanakya", "Saketh", "Navadeep", "Mahanth"]

from .exact_matching import *
from .approximate_matching import *

__all__ = [
    "KMPMatcher",
    "BoyerMooreMatcher", 
    "SuffixTreeMatcher",
    "NaiveMatcher",
    "LevenshteinMatcher",
    "ShiftOrMatcher"
]
