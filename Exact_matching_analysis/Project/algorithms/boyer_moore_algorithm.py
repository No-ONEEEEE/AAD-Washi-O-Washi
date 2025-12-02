"""
Boyer-Moore String Matching Algorithm
Implements both Bad Character Rule and Good Suffix Rule for optimal performance
"""

def build_bad_character_table(pattern):
    """
    Build the bad character table for Boyer-Moore algorithm.
    For each character, store the rightmost position in pattern (or -1 if not present).
    """
    table = {}
    m = len(pattern)
    
    # Initialize all characters to -1 (not found)
    # For DNA, we only care about A, C, G, T
    for char in 'ACGT':
        table[char] = -1
    
    # For each character in the pattern, store its rightmost position
    for i in range(m):
        table[pattern[i]] = i
    
    return table


def build_good_suffix_table(pattern):
    """
    Build the good suffix table for Boyer-Moore algorithm.
    Returns an array where shift[i] represents the shift distance when mismatch occurs at position i.
    """
    m = len(pattern)
    shift = [0] * m
    border = [0] * (m + 1)
    
    # Preprocessing for case 1: suffix of pattern matches a substring
    i = m
    j = m + 1
    border[i] = j
    
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j - 1] == 0:
                shift[j - 1] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j
    
    # Preprocessing for case 2: prefix of pattern matches a suffix
    j = border[0]
    for i in range(m):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border[j]
    
    return shift


def boyer_moore_search(text, pattern):
    """
    Boyer-Moore string matching algorithm with both bad character and good suffix rules.
    
    Args:
        text: The text to search in
        pattern: The pattern to search for
    
    Returns:
        List of starting positions where pattern is found in text
    """
    n = len(text)
    m = len(pattern)
    
    if m == 0 or m > n:
        return []
    
    # Build preprocessing tables
    bad_char = build_bad_character_table(pattern)
    good_suffix = build_good_suffix_table(pattern)
    
    matches = []
    s = 0  # shift of the pattern with respect to text
    
    while s <= n - m:
        j = m - 1
        
        # Keep reducing index j while characters match
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        
        if j < 0:
            # Pattern found at position s
            matches.append(s)
            # Shift pattern to align with next possible match
            s += good_suffix[0] if s + m < n else 1
        else:
            # Mismatch occurred
            # Calculate shift using bad character rule
            mismatched_char = text[s + j]
            if mismatched_char in bad_char:
                # Shift to align mismatched character with its rightmost occurrence in pattern
                bad_char_shift = max(1, j - bad_char[mismatched_char])
            else:
                # Character not in pattern, shift pattern completely past it
                bad_char_shift = j + 1
            
            # Calculate shift using good suffix rule
            good_suffix_shift = good_suffix[j]
            
            # Use the maximum of both shifts
            s += max(bad_char_shift, good_suffix_shift)
    
    return matches


def search(text, pattern):
    """
    Wrapper function to match the interface of other algorithms.
    Returns the list of match positions.
    """
    return boyer_moore_search(text, pattern)


if __name__ == "__main__":
    # Test cases
    text = "AABAACAADAABAABA"
    pattern = "AABA"
    
    matches = boyer_moore_search(text, pattern)
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print(f"Matches found at positions: {matches}")
    
    # Edge cases
    print("\n--- Edge Cases ---")
    
    # No match
    result = boyer_moore_search("AAAA", "B")
    print(f"No match test: {result}")
    
    # Entire text is pattern
    result = boyer_moore_search("ABC", "ABC")
    print(f"Full match test: {result}")
    
    # Repetitive pattern
    result = boyer_moore_search("AAAAAAA", "AAA")
    print(f"Repetitive pattern: {result}")
    
    # Multiple matches
    result = boyer_moore_search("ABABABABAB", "AB")
    print(f"Multiple matches: {result}")
