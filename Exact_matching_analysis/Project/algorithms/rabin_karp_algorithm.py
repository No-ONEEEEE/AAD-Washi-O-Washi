"""
Rabin-Karp Algorithm
Rolling hash-based approach with O(n+m) expected time.
Handles multiple pattern matching efficiently.

Time Complexity: O(n+m) expected, O(nm) worst case
Space Complexity: O(1)
"""

def rabin_karp_search(text, pattern, prime=101):
    """
    Search for pattern in text using Rabin-Karp algorithm.
    
    Args:
        text: The text to search in
        pattern: The pattern to search for
        prime: Prime number for hashing (default: 101)
        
    Returns:
        List of starting indices where pattern is found
    """
    if not text or not pattern or len(pattern) > len(text):
        return []
    
    n = len(text)
    m = len(pattern)
    d = 256  # Number of characters in alphabet
    matches = []
    
    # Calculate hash value for pattern and first window of text
    pattern_hash = 0
    text_hash = 0
    h = 1
    
    # h = d^(m-1) % prime
    for i in range(m - 1):
        h = (h * d) % prime
    
    # Calculate initial hash values
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime
    
    # Slide the pattern over text
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Verify character by character (spurious hit check)
            if text[i:i + m] == pattern:
                matches.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            
            # Handle negative hash
            if text_hash < 0:
                text_hash += prime
    
    return matches


def rabin_karp_multiple_patterns(text, patterns, prime=101):
    """
    Search for multiple patterns in text using Rabin-Karp algorithm.
    
    Args:
        text: The text to search in
        patterns: List of patterns to search for
        prime: Prime number for hashing
        
    Returns:
        Dictionary mapping pattern to list of match positions
    """
    if not text or not patterns:
        return {}
    
    results = {}
    d = 256
    
    for pattern in patterns:
        if not pattern or len(pattern) > len(text):
            results[pattern] = []
            continue
        
        matches = rabin_karp_search(text, pattern, prime)
        results[pattern] = matches
    
    return results


def rabin_karp_with_hash_info(text, pattern, prime=101):
    """
    Rabin-Karp with detailed hash information for educational purposes.
    
    Returns:
        Tuple of (matches, hash_comparisons, character_comparisons)
    """
    if not text or not pattern or len(pattern) > len(text):
        return [], 0, 0
    
    n = len(text)
    m = len(pattern)
    d = 256
    matches = []
    hash_comparisons = 0
    character_comparisons = 0
    
    pattern_hash = 0
    text_hash = 0
    h = 1
    
    for i in range(m - 1):
        h = (h * d) % prime
    
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime
    
    for i in range(n - m + 1):
        hash_comparisons += 1
        
        if pattern_hash == text_hash:
            # Verify character by character
            match = True
            for j in range(m):
                character_comparisons += 1
                if text[i + j] != pattern[j]:
                    match = False
                    break
            
            if match:
                matches.append(i)
        
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime
    
    return matches, hash_comparisons, character_comparisons


def main():
    """Example usage and demonstration of Rabin-Karp algorithm."""
    print("=" * 70)
    print("Rabin-Karp Algorithm - Rolling Hash Pattern Matching")
    print("=" * 70)
    print()
    
    # Example 1: Basic pattern matching
    print("Example 1: Basic Pattern Matching")
    print("-" * 70)
    text = "AABAACAADAABAABA"
    pattern = "AABA"
    
    matches = rabin_karp_search(text, pattern)
    
    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    print(f"Matches found at positions: {matches}")
    print()
    
    # Example 2: DNA sequence matching
    print("Example 2: DNA Sequence Matching")
    print("-" * 70)
    dna = "ACGTACGTACGTACGT"
    gene = "ACGT"
    
    matches = rabin_karp_search(dna, gene)
    
    print(f"DNA:  {dna}")
    print(f"Gene: {gene}")
    print(f"Matches: {matches}")
    print()
    
    # Example 3: Multiple pattern matching
    print("Example 3: Multiple Pattern Matching")
    print("-" * 70)
    text = "ACGTACGTACGTACGT"
    patterns = ["ACGT", "CGTA", "GTAC", "TACG"]
    
    results = rabin_karp_multiple_patterns(text, patterns)
    
    print(f"Text: {text}")
    print(f"Patterns: {patterns}")
    print()
    for pattern, positions in results.items():
        print(f"  {pattern}: {positions}")
    print()
    
    # Example 4: Hash collision demonstration
    print("Example 4: Algorithm Analysis")
    print("-" * 70)
    text = "ACGTACGTACGTACGT" * 100
    pattern = "ACGTACGT"
    
    matches, hash_comp, char_comp = rabin_karp_with_hash_info(text, pattern)
    
    print(f"Text length: {len(text)}")
    print(f"Pattern length: {len(pattern)}")
    print(f"Matches found: {len(matches)}")
    print(f"Hash comparisons: {hash_comp}")
    print(f"Character comparisons: {char_comp}")
    print(f"Efficiency: {char_comp / hash_comp:.2f} chars/hash")
    print()
    
    # Example 5: Performance comparison
    print("Example 5: Performance Benchmark")
    print("-" * 70)
    import time
    
    large_text = "ACGT" * 25000  # 100K characters
    search_pattern = "ACGTACGT"
    
    start = time.time()
    matches = rabin_karp_search(large_text, search_pattern)
    elapsed = time.time() - start
    
    print(f"Text size: {len(large_text):,} characters")
    print(f"Pattern: {search_pattern}")
    print(f"Matches found: {len(matches)}")
    print(f"Time: {elapsed:.6f} seconds")
    print()
    
    print("=" * 70)
    print("Algorithm Characteristics:")
    print("  • Expected time: O(n+m)")
    print("  • Worst case: O(nm) with many hash collisions")
    print("  • Space: O(1)")
    print("  • Excellent for multiple pattern matching")
    print("  • Uses rolling hash for efficiency")
    print("=" * 70)


if __name__ == "__main__":
    main()
