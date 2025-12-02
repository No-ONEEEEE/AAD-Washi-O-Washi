"""
Knuth-Morris-Pratt (KMP) Algorithm
Linear-time O(n+m) pattern matching using failure function preprocessing
to avoid redundant comparisons.
"""

def compute_failure_function(pattern):
    """
    Compute the failure function (also called prefix function or LPS array)
    for the KMP algorithm.
    
    Args:
        pattern: The pattern string to search for
        
    Returns:
        List of integers representing the failure function
    """
    m = len(pattern)
    failure = [0] * m
    j = 0
    
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        
        if pattern[i] == pattern[j]:
            j += 1
        
        failure[i] = j
    
    return failure


def kmp_search(text, pattern):
    """
    Search for pattern in text using the KMP algorithm.
    
    Args:
        text: The text string to search in
        pattern: The pattern string to search for
        
    Returns:
        List of starting indices where pattern is found in text
    """
    if not pattern or not text:
        return []
    
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    # Compute failure function
    failure = compute_failure_function(pattern)
    
    matches = []
    j = 0  # Index for pattern
    
    for i in range(n):  # Index for text
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]
        
        if text[i] == pattern[j]:
            j += 1
        
        if j == m:
            # Pattern found at index i - m + 1
            matches.append(i - m + 1)
            j = failure[j - 1]
    
    return matches


def main():
    """Example usage of KMP algorithm"""
    # Example 1: DNA sequence matching
    dna_sequence = "ACGTACGTACGTACGT"
    pattern1 = "ACGT"
    
    print("KMP Algorithm - Pattern Matching")
    print("=" * 50)
    print(f"Text: {dna_sequence}")
    print(f"Pattern: {pattern1}")
    
    matches = kmp_search(dna_sequence, pattern1)
    print(f"Pattern found at indices: {matches}")
    print(f"Number of matches: {len(matches)}")
    print()
    
    # Example 2: Text search
    text = "ABABDABACDABABCABAB"
    pattern2 = "ABABCABAB"
    
    print(f"Text: {text}")
    print(f"Pattern: {pattern2}")
    
    matches = kmp_search(text, pattern2)
    print(f"Pattern found at indices: {matches}")
    print(f"Number of matches: {len(matches)}")
    print()
    
    # Example 3: Failure function demonstration
    pattern3 = "ABABACA"
    failure = compute_failure_function(pattern3)
    print(f"Pattern: {pattern3}")
    print(f"Failure function: {failure}")
    print()
    
    # Time complexity demonstration
    print("Time Complexity:")
    print("- Preprocessing (failure function): O(m)")
    print("- Searching: O(n)")
    print("- Total: O(n + m)")
    print(f"  where n = {len(text)}, m = {len(pattern2)}")


if __name__ == "__main__":
    main()
