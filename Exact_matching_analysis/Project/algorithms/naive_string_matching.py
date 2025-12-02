"""
Naive String Matching Algorithm - Modular Implementation
========================================================

This module implements the naive (brute-force) string matching algorithm.
Despite O(nm) complexity, it's simple, requires no preprocessing, and serves
as a baseline for understanding pattern matching fundamentals.

Key Features:
    - Simple and intuitive
    - No preprocessing required
    - O(nm) time complexity
    - Good for teaching and small inputs
    
Author: Algorithm Implementation
Date: 2025
"""


# ============================================================================
# CORE SEARCH MODULE
# ============================================================================

def naive_string_match(text, pattern):
    """
    Search for pattern in text using naive character-by-character comparison.
    
    Algorithm:
        1. Try every possible starting position in text
        2. At each position, compare pattern character by character
        3. If all characters match, record the position
        4. Continue until all positions are checked
    
    Time Complexity: O(nm) where n = len(text), m = len(pattern)
    Space Complexity: O(1) excluding output
    
    Args:
        text (str): The text string to search in
        pattern (str): The pattern string to search for
        
    Returns:
        list[int]: List of starting indices where pattern is found
        
    Example:
        >>> naive_string_match("ABABDABACDABABCABAB", "ABAB")
        [0, 10, 15]
    """
    # Handle edge cases
    if not pattern or not text:
        return []
    
    n = len(text)
    m = len(pattern)
    
    # Pattern cannot be longer than text
    if m > n:
        return []
    
    matches = []
    
    # Try every possible starting position
    for i in range(n - m + 1):
        # Check if pattern matches at position i
        match = True
        
        # Compare pattern character by character
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break  # Mismatch found, stop comparing
        
        # If all characters matched, record this position
        if match:
            matches.append(i)
    
    return matches


# ============================================================================
# VERBOSE SEARCH MODULE (FOR ANALYSIS)
# ============================================================================

def naive_string_match_verbose(text, pattern, show_comparisons=False):
    """
    Naive string matching with detailed comparison tracking.
    
    Useful for:
        - Understanding algorithm behavior
        - Analyzing worst-case scenarios
        - Teaching and demonstration
    
    Args:
        text (str): The text string to search in
        pattern (str): The pattern string to search for
        show_comparisons (bool): If True, print each comparison step
        
    Returns:
        tuple: (matches, total_comparisons)
               matches (list[int]): Indices where pattern found
               total_comparisons (int): Number of character comparisons made
               
    Example:
        >>> matches, comps = naive_string_match_verbose("AAAA", "AA", False)
        >>> print(f"Matches: {matches}, Comparisons: {comps}")
        Matches: [0, 1, 2], Comparisons: 6
    """
    if not pattern or not text:
        return [], 0
    
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return [], 0
    
    matches = []
    comparisons = 0
    
    # Try every possible starting position
    for i in range(n - m + 1):
        if show_comparisons:
            print(f"\n{'='*50}")
            print(f"Checking position {i}:")
            print(f"Text:    {text}")
            print(f"         {' ' * i}{pattern}")
            print(f"{'='*50}")
        
        # Check if pattern matches at position i
        match = True
        for j in range(m):
            comparisons += 1
            
            if show_comparisons:
                print(f"  Compare text[{i+j}]='{text[i+j]}' with pattern[{j}]='{pattern[j]}'", end="")
            
            if text[i + j] != pattern[j]:
                match = False
                if show_comparisons:
                    print(" ✗ Mismatch!")
                break
            else:
                if show_comparisons:
                    print(" ✓ Match")
        
        if match:
            matches.append(i)
            if show_comparisons:
                print(f"  ✓✓✓ Pattern found at position {i}!")
    
    return matches, comparisons


# ============================================================================
# WILDCARD SEARCH MODULE
# ============================================================================

def naive_string_match_with_wildcards(text, pattern, wildcard='?'):
    """
    Naive string matching with wildcard support.
    
    The wildcard character matches any single character in the text.
    This extends the basic algorithm to support flexible pattern matching.
    
    Time Complexity: O(nm)
    Space Complexity: O(1) excluding output
    
    Args:
        text (str): The text string to search in
        pattern (str): The pattern string (may contain wildcards)
        wildcard (str): Character to use as wildcard (default '?')
        
    Returns:
        list[int]: Starting indices where pattern matches
        
    Example:
        >>> naive_string_match_with_wildcards("ABCDEFG", "C?E")
        [2]  # Matches "CDE" where ? matches 'D'
    """
    if not pattern or not text:
        return []
    
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    matches = []
    
    # Try every possible starting position
    for i in range(n - m + 1):
        match = True
        
        # Compare with wildcard support
        for j in range(m):
            # Wildcard matches any character
            if pattern[j] != wildcard and text[i + j] != pattern[j]:
                match = False
                break
        
        if match:
            matches.append(i)
    
    return matches


# ============================================================================
# UTILITY MODULE
# ============================================================================

def count_pattern_occurrences(text, pattern):
    """
    Count the number of times pattern occurs in text.
    
    Args:
        text (str): The text to search in
        pattern (str): The pattern to count
        
    Returns:
        int: Number of occurrences
    """
    matches = naive_string_match(text, pattern)
    return len(matches)


def get_algorithm_info():
    """Get algorithm metadata."""
    return {
        'name': 'Naive String Matching',
        'time_complexity': 'O(nm)',
        'space_complexity': 'O(1)',
        'preprocessing': 'None',
        'best_for': 'Small texts, teaching, baseline comparison',
        'worst_case': 'Highly repetitive text with pattern'
    }


def analyze_complexity(text, pattern):
    """
    Analyze the complexity for given text and pattern.
    
    Args:
        text (str): Text to analyze
        pattern (str): Pattern to analyze
        
    Returns:
        dict: Analysis results
    """
    n = len(text)
    m = len(pattern)
    
    # Perform verbose search to count comparisons
    matches, comparisons = naive_string_match_verbose(text, pattern, False)
    
    # Theoretical worst case
    worst_case_comparisons = (n - m + 1) * m
    
    return {
        'text_length': n,
        'pattern_length': m,
        'matches_found': len(matches),
        'actual_comparisons': comparisons,
        'worst_case_comparisons': worst_case_comparisons,
        'efficiency': f"{(comparisons/worst_case_comparisons)*100:.1f}%"
    }


# ============================================================================
# DEMONSTRATION MODULE
# ============================================================================

def run_example_1():
    """Example 1: Basic DNA sequence matching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Pattern Matching")
    print("=" * 70)
    
    text = "ACGTACGTACGTACGT"
    pattern = "ACGT"
    
    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    
    matches = naive_string_match(text, pattern)
    
    print(f"\nMatches found at indices: {matches}")
    print(f"Total matches: {len(matches)}")
    
    # Visualize
    print("\nVisualization:")
    for idx in matches:
        print(f"  Position {idx}: {text[idx:idx+len(pattern)]}")


def run_example_2():
    """Example 2: Verbose mode demonstration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Verbose Mode (Showing Comparisons)")
    print("=" * 70)
    
    text = "AABAACAAD"
    pattern = "AAC"
    
    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    
    matches, comparisons = naive_string_match_verbose(text, pattern, True)
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total comparisons: {comparisons}")
    print(f"  Matches found: {matches}")


def run_example_3():
    """Example 3: Wildcard matching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Wildcard Matching")
    print("=" * 70)
    
    text = "ABCDEFGHIJK"
    pattern = "C?E"
    
    print(f"Text:    {text}")
    print(f"Pattern: {pattern} (? matches any character)")
    
    matches = naive_string_match_with_wildcards(text, pattern)
    
    print(f"\nMatches at indices: {matches}")
    for idx in matches:
        print(f"  Position {idx}: {text[idx:idx+len(pattern)]}")


def run_example_4():
    """Example 4: Complexity analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Complexity Analysis")
    print("=" * 70)
    
    # Best case: pattern found immediately
    text1 = "ABCDEFGH"
    pattern1 = "ABC"
    
    print("Best Case Scenario:")
    print(f"  Text: {text1}")
    print(f"  Pattern: {pattern1}")
    
    analysis1 = analyze_complexity(text1, pattern1)
    print(f"  Comparisons: {analysis1['actual_comparisons']} / {analysis1['worst_case_comparisons']} (worst)")
    
    # Worst case: many partial matches
    text2 = "AAAAAAAAAA"
    pattern2 = "AAAB"
    
    print("\nWorst Case Scenario:")
    print(f"  Text: {text2}")
    print(f"  Pattern: {pattern2}")
    
    analysis2 = analyze_complexity(text2, pattern2)
    print(f"  Comparisons: {analysis2['actual_comparisons']} / {analysis2['worst_case_comparisons']} (worst)")


def display_algorithm_info():
    """Display algorithm information."""
    print("\n" + "=" * 70)
    print("ALGORITHM INFORMATION")
    print("=" * 70)
    
    info = get_algorithm_info()
    
    print(f"Algorithm:        {info['name']}")
    print(f"Time Complexity:  {info['time_complexity']}")
    print(f"Space Complexity: {info['space_complexity']}")
    print(f"Preprocessing:    {info['preprocessing']}")
    print(f"Best For:         {info['best_for']}")
    
    print("\nCharacteristics:")
    print("  • Simple and intuitive")
    print("  • No preprocessing overhead")
    print("  • Checks every position in text")
    print("  • Inefficient for large texts")
    
    print("\nWhen to Use:")
    print("  • Teaching pattern matching concepts")
    print("  • Very small texts")
    print("  • Baseline for algorithm comparison")
    print("  • When simplicity is more important than speed")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 70)
    print("NAIVE STRING MATCHING ALGORITHM - DEMONSTRATION")
    print("=" * 70)
    
    run_example_1()
    run_example_2()
    run_example_3()
    run_example_4()
    display_algorithm_info()
    
    print("\n" + "=" * 70)
    print("END OF DEMONSTRATION")
    print("=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
