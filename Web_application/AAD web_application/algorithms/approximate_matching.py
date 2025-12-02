"""
Approximate Pattern Matching Algorithms

This module contains implementations of approximate/fuzzy string matching algorithms
including Shift-Or, Levenshtein Distance, and other edit distance based algorithms.
"""

from typing import List, Dict, Tuple, Optional
import time
from abc import ABC, abstractmethod


class ApproximatePatternMatcher(ABC):
    """Abstract base class for approximate pattern matching algorithms."""
    
    @abstractmethod
    def search(self, text: str, pattern: str, max_errors: int = 1) -> List[Tuple[int, int]]:
        """
        Search for approximate matches of pattern in text.
        
        Returns:
            List of tuples (position, edit_distance) where matches are found
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        pass
    
    def benchmark(self, text: str, pattern: str, max_errors: int = 1) -> Dict[str, any]:
        """Benchmark the algorithm and return timing information."""
        start_time = time.perf_counter()
        matches = self.search(text, pattern, max_errors)
        end_time = time.perf_counter()
        
        return {
            "algorithm": self.get_algorithm_name(),
            "execution_time": end_time - start_time,
            "matches_found": len(matches),
            "matches": matches
        }


class LevenshteinMatcher(ApproximatePatternMatcher):
    """
    Levenshtein Distance (Edit Distance) Dynamic Programming Algorithm
    
    Time Complexity: O(nm) where n = len(text), m = len(pattern)
    Space Complexity: O(nm) for DP table
    
    Operations: insertions, deletions, substitutions
    Best for: Finding similar sequences with few edits
    """
    
    def edit_distance(self, str1: str, str2: str) -> int:
        """
        Calculate edit distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Minimum number of edits needed to transform str1 to str2
        """
        m, n = len(str1), len(str2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]
    
    def search(self, text: str, pattern: str, max_errors: int = 1) -> List[Tuple[int, int]]:
        """
        Search for approximate matches using edit distance.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            max_errors: Maximum allowed edit distance
            
        Returns:
            List of tuples (position, edit_distance) where matches are found
        """
        if not pattern or not text:
            return []
        
        matches = []
        m = len(pattern)
        n = len(text)
        
        # Check all possible starting positions
        for i in range(n - m + max_errors + 1):
            best_distance = float('inf')
            best_length = m
            
            # Try different substring lengths around pattern length
            for length in range(max(1, m - max_errors), min(n - i + 1, m + max_errors + 1)):
                substring = text[i:i + length]
                distance = self.edit_distance(pattern, substring)
                
                # Keep only the best match at this position
                if distance < best_distance:
                    best_distance = distance
                    best_length = length
            
            # Only add if within error threshold
            if best_distance <= max_errors:
                matches.append((i, best_distance))
        
        # Remove overlapping matches - keep only non-overlapping best matches
        if not matches:
            return []
        
        # Sort by position and distance
        matches.sort(key=lambda x: (x[0], x[1]))
        
        # Filter to remove duplicates and keep best matches
        filtered_matches = []
        last_pos = -m
        
        for pos, dist in matches:
            if pos >= last_pos + m:  # No overlap with previous match
                filtered_matches.append((pos, dist))
                last_pos = pos
        
        return filtered_matches
    
    def get_algorithm_name(self) -> str:
        return "Levenshtein Distance (Edit Distance)"
    
    def visualize_search(self, text: str, pattern: str, max_errors: int = 1) -> Dict:
        """
        Generate step-by-step visualization data for Levenshtein Distance search.
        
        Returns detailed DP matrix construction and edit operations.
        """
        if not pattern or not text:
            return {"steps": [], "summary": {}}
        
        m = len(pattern)
        n = len(text)
        steps = []
        total_cells_computed = 0
        total_positions_checked = 0
        matches_found = []
        
        # Check all possible starting positions with sliding window
        for start_pos in range(n - m + max_errors + 1):
            # Try different substring lengths around pattern length
            for length in range(max(1, m - max_errors), min(n - start_pos + 1, m + max_errors + 1)):
                substring = text[start_pos:start_pos + length]
                total_positions_checked += 1
                
                # Create DP table for this comparison
                dp = [[0] * (length + 1) for _ in range(m + 1)]
                
                # Initialize base cases
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(length + 1):
                    dp[0][j] = j
                
                # Preprocessing step - show initial DP matrix
                if start_pos == 0 and length == min(m + max_errors, n):
                    preprocessing_step = {
                        'step_number': len(steps),
                        'step_type': 'preprocessing',
                        'start_position': start_pos,
                        'substring': substring,
                        'substring_length': length,
                        'dp_matrix': [row[:] for row in dp],
                        'pattern': pattern,
                        'explanation': f"Initializing DP matrix for pattern '{pattern}' vs text substring '{substring[:20]}{'...' if len(substring) > 20 else ''}'"
                    }
                    steps.append(preprocessing_step)
                
                # Fill DP table and track steps for first few positions
                show_steps = (start_pos < 3 or len(matches_found) < 2)  # Show first few or when matches found
                
                for i in range(1, m + 1):
                    for j in range(1, length + 1):
                        total_cells_computed += 1
                        
                        pattern_char = pattern[i - 1]
                        text_char = substring[j - 1]
                        is_match = pattern_char == text_char
                        
                        # Calculate three options
                        deletion_cost = dp[i - 1][j] + 1
                        insertion_cost = dp[i][j - 1] + 1
                        substitution_cost = dp[i - 1][j - 1] + (0 if is_match else 1)
                        
                        # Choose minimum
                        min_cost = min(deletion_cost, insertion_cost, substitution_cost)
                        dp[i][j] = min_cost
                        
                        # Determine operation
                        if is_match and substitution_cost == min_cost:
                            operation = 'match'
                            operation_symbol = '✓'
                        elif substitution_cost == min_cost:
                            operation = 'substitute'
                            operation_symbol = '⇄'
                        elif deletion_cost == min_cost:
                            operation = 'delete'
                            operation_symbol = '⌫'
                        else:
                            operation = 'insert'
                            operation_symbol = '⊕'
                        
                        # Record step for visualization (limited to keep size manageable)
                        if show_steps and (i <= 3 or j <= 3 or i == m or j == length):
                            step_data = {
                                'step_number': len(steps),
                                'step_type': 'dp_computation',
                                'start_position': start_pos,
                                'substring': substring,
                                'dp_row': i,
                                'dp_col': j,
                                'pattern_char': pattern_char,
                                'text_char': text_char,
                                'is_match': is_match,
                                'deletion_cost': deletion_cost,
                                'insertion_cost': insertion_cost,
                                'substitution_cost': substitution_cost,
                                'chosen_cost': min_cost,
                                'operation': operation,
                                'operation_symbol': operation_symbol,
                                'dp_matrix': [row[:] for row in dp],  # Deep copy
                                'explanation': self._get_levenshtein_explanation(
                                    i, j, pattern_char, text_char, is_match, 
                                    operation, min_cost, deletion_cost, 
                                    insertion_cost, substitution_cost
                                ),
                                'predecessor_cells': {
                                    'diagonal': dp[i-1][j-1],
                                    'left': dp[i][j-1],
                                    'top': dp[i-1][j]
                                }
                            }
                            steps.append(step_data)
                
                # Get final edit distance
                final_distance = dp[m][length]
                
                # If within threshold, it's a match
                if final_distance <= max_errors:
                    # Traceback to get alignment
                    alignment = self._traceback_alignment(pattern, substring, dp)
                    
                    match_step = {
                        'step_number': len(steps),
                        'step_type': 'match_found',
                        'start_position': start_pos,
                        'substring': substring,
                        'edit_distance': final_distance,
                        'dp_matrix': [row[:] for row in dp],
                        'alignment': alignment,
                        'pattern_aligned': alignment['pattern_aligned'],
                        'text_aligned': alignment['text_aligned'],
                        'operations': alignment['operations'],
                        'explanation': f"✓ Match found at position {start_pos} with edit distance {final_distance}! Alignment: {alignment['pattern_aligned']} ↔ {alignment['text_aligned']}"
                    }
                    steps.append(match_step)
                    
                    matches_found.append((start_pos, final_distance))
                    
                    # Break inner loop since we found best match at this position
                    break
            
            # Skip overlapping positions if match found
            if matches_found and matches_found[-1][0] == start_pos:
                # Skip next m-1 positions to avoid overlap
                continue
        
        # Summary statistics
        summary = {
            'algorithm': 'Levenshtein Distance (Edit Distance)',
            'text_length': n,
            'pattern_length': m,
            'max_errors': max_errors,
            'total_cells_computed': total_cells_computed,
            'positions_checked': total_positions_checked,
            'matches_found': len(matches_found),
            'match_positions': [pos for pos, dist in matches_found],
            'match_details': [{'position': pos, 'distance': dist} for pos, dist in matches_found],
            'time_complexity': 'O(nm) per position',
            'space_complexity': 'O(nm)',
            'total_comparisons': total_cells_computed,
            'worst_case_comparisons': (n - m + max_errors + 1) * (m + 1) * (m + max_errors + 1),
            'operations': ['Match', 'Substitute', 'Insert', 'Delete'],
            'dp_table_size': f"{m+1} × {m+max_errors+1}"
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def _get_levenshtein_explanation(self, i: int, j: int, pattern_char: str, 
                                      text_char: str, is_match: bool, operation: str,
                                      chosen_cost: int, deletion_cost: int,
                                      insertion_cost: int, substitution_cost: int) -> str:
        """Generate human-readable explanation for Levenshtein DP computation."""
        if is_match:
            return f"DP[{i}][{j}]: Pattern[{i-1}]='{pattern_char}' matches Text[{j-1}]='{text_char}'. Cost = DP[{i-1}][{j-1}] = {chosen_cost} (no edit needed)"
        else:
            ops = []
            if deletion_cost == chosen_cost:
                ops.append(f"Delete '{pattern_char}' (cost {deletion_cost})")
            if insertion_cost == chosen_cost:
                ops.append(f"Insert '{text_char}' (cost {insertion_cost})")
            if substitution_cost == chosen_cost:
                ops.append(f"Substitute '{pattern_char}'→'{text_char}' (cost {substitution_cost})")
            
            chosen_op = operation.capitalize()
            return f"DP[{i}][{j}]: Pattern[{i-1}]='{pattern_char}' ≠ Text[{j-1}]='{text_char}'. Min cost = {chosen_cost}. Operation: {chosen_op}"
    
    def _traceback_alignment(self, pattern: str, text: str, dp: List[List[int]]) -> Dict:
        """
        Traceback through DP matrix to get alignment and operations.
        """
        m, n = len(pattern), len(text)
        i, j = m, n
        
        pattern_aligned = []
        text_aligned = []
        operations = []
        
        while i > 0 or j > 0:
            if i == 0:
                # Only insertions left
                text_aligned.insert(0, text[j-1])
                pattern_aligned.insert(0, '-')
                operations.insert(0, 'insert')
                j -= 1
            elif j == 0:
                # Only deletions left
                pattern_aligned.insert(0, pattern[i-1])
                text_aligned.insert(0, '-')
                operations.insert(0, 'delete')
                i -= 1
            elif pattern[i-1] == text[j-1]:
                # Match
                pattern_aligned.insert(0, pattern[i-1])
                text_aligned.insert(0, text[j-1])
                operations.insert(0, 'match')
                i -= 1
                j -= 1
            else:
                # Find which operation was used
                deletion_cost = dp[i-1][j]
                insertion_cost = dp[i][j-1]
                substitution_cost = dp[i-1][j-1]
                
                min_cost = min(deletion_cost, insertion_cost, substitution_cost)
                
                if substitution_cost == min_cost:
                    # Substitution
                    pattern_aligned.insert(0, pattern[i-1])
                    text_aligned.insert(0, text[j-1])
                    operations.insert(0, 'substitute')
                    i -= 1
                    j -= 1
                elif deletion_cost == min_cost:
                    # Deletion
                    pattern_aligned.insert(0, pattern[i-1])
                    text_aligned.insert(0, '-')
                    operations.insert(0, 'delete')
                    i -= 1
                else:
                    # Insertion
                    pattern_aligned.insert(0, '-')
                    text_aligned.insert(0, text[j-1])
                    operations.insert(0, 'insert')
                    j -= 1
        
        return {
            'pattern_aligned': ''.join(pattern_aligned),
            'text_aligned': ''.join(text_aligned),
            'operations': operations
        }


class ShiftOrMatcher(ApproximatePatternMatcher):
    """
    Shift-Or Algorithm with k-mismatches
    
    Time Complexity: O(nm/w) where w is word size
    Space Complexity: O(σ) where σ is alphabet size
    
    Uses bit-parallel operations for efficient matching
    Best for: Small patterns, DNA sequences (small alphabet)
    """
    
    def _create_pattern_mask(self, pattern: str) -> Dict[str, int]:
        """
        Create bit mask for each character in the pattern.
        
        Args:
            pattern: The pattern to create masks for
            
        Returns:
            Dictionary mapping characters to their bit masks
        """
        masks = {}
        m = len(pattern)
        
        # Initialize all masks to have all bits set
        alphabet = set(pattern)
        for char in alphabet:
            masks[char] = (1 << m) - 1
        
        # Set appropriate bits for each character
        for i, char in enumerate(pattern):
            masks[char] &= ~(1 << i)
        
        return masks
    
    def search_exact(self, text: str, pattern: str) -> List[int]:
        """
        Exact matching using Shift-Or algorithm.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not pattern or not text:
            return []
        
        positions = []
        m = len(pattern)
        masks = self._create_pattern_mask(pattern)
        
        # Initialize state
        state = (1 << m) - 1
        match_mask = 1 << (m - 1)
        
        for i, char in enumerate(text):
            # Shift and update state
            state = (state << 1) | masks.get(char, (1 << m) - 1)
            
            # Check if we have a match
            if not (state & match_mask):
                positions.append(i - m + 1)
        
        return positions
    
    def search(self, text: str, pattern: str, max_errors: int = 1) -> List[Tuple[int, int]]:
        """
        Approximate matching using Shift-Or with k-mismatches.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            max_errors: Maximum allowed mismatches
            
        Returns:
            List of tuples (position, mismatches) where matches are found
        """
        if not pattern or not text:
            return []
        
        if max_errors == 0:
            # Use exact matching
            exact_matches = self.search_exact(text, pattern)
            return [(pos, 0) for pos in exact_matches]
        
        # For simplicity, fall back to sliding window approach for k > 0
        matches = []
        m = len(pattern)
        n = len(text)
        
        for i in range(n - m + 1):
            mismatches = 0
            for j in range(m):
                if text[i + j] != pattern[j]:
                    mismatches += 1
                    if mismatches > max_errors:
                        break
            
            if mismatches <= max_errors:
                matches.append((i, mismatches))
        
        return matches
    
    def visualize_search(self, text: str, pattern: str, max_errors: int = 1) -> Dict:
        """
        Generate step-by-step visualization data for Shift-Or search.
        
        Shows bit-parallel operations, pattern masks, state transitions.
        """
        if not text or not pattern or len(pattern) > len(text):
            return {"steps": [], "summary": {}}
        
        n = len(text)
        m = len(pattern)
        steps = []
        total_shifts = 0
        total_or_operations = 0
        matches_found = []
        
        # Create pattern masks
        masks = self._create_pattern_mask(pattern)
        
        # Preprocessing step - show pattern masks
        mask_display = {}
        for char, mask in masks.items():
            mask_display[char] = {
                'decimal': mask,
                'binary': bin(mask)[2:].zfill(m),
                'explanation': f"Bit positions where '{char}' does NOT appear in pattern"
            }
        
        preprocessing_step = {
            'step_number': 0,
            'step_type': 'preprocessing',
            'pattern': pattern,
            'pattern_length': m,
            'pattern_masks': mask_display,
            'max_errors': max_errors,
            'explanation': f"Preprocessing: Created bit masks for pattern '{pattern}'. Each character has a {m}-bit mask."
        }
        steps.append(preprocessing_step)
        
        # Initialize state vectors (one for each error level)
        states = [(1 << m) - 1 for _ in range(max_errors + 1)]
        match_mask = 1 << (m - 1)
        
        # Initialization step
        init_step = {
            'step_number': len(steps),
            'step_type': 'initialization',
            'initial_states': [bin(s)[2:].zfill(m) for s in states],
            'match_mask': bin(match_mask)[2:].zfill(m),
            'explanation': f"Initialized {max_errors + 1} state vector(s). Match mask = {bin(match_mask)[2:].zfill(m)} (checks bit {m-1})"
        }
        steps.append(init_step)
        
        # Process each character in text
        for i, char in enumerate(text):
            old_states = states.copy()
            
            # Get character mask (or default mask if char not in pattern)
            char_mask = masks.get(char, (1 << m) - 1)
            
            # Update states for each error level
            new_states = []
            
            # Level 0 (exact match)
            state_0 = (old_states[0] << 1) | char_mask
            new_states.append(state_0)
            total_shifts += 1
            total_or_operations += 1
            
            # Additional levels for approximate matching
            for k in range(1, max_errors + 1):
                # Three operations: shift-or (exact), insertion, deletion, substitution
                shifted = (old_states[k] << 1) | char_mask  # Match/Substitute
                insertion = old_states[k] << 1  # Insertion (skip text char)
                deletion = old_states[k - 1]  # Deletion (skip pattern char)
                
                # Combine using bitwise AND (minimum)
                state_k = shifted & insertion & deletion
                new_states.append(state_k)
                
                total_shifts += 3
                total_or_operations += 1
            
            states = new_states
            
            # Check for matches at each error level
            matches_at_step = []
            for k in range(max_errors + 1):
                if not (states[k] & match_mask):
                    match_pos = i - m + 1
                    if match_pos >= 0:
                        matches_at_step.append({
                            'position': match_pos,
                            'errors': k,
                            'state_level': k
                        })
                        
                        # Add to overall matches if not already found with fewer errors
                        existing = [pos for pos, err in matches_found]
                        if match_pos not in existing:
                            matches_found.append((match_pos, k))
            
            # Determine if this character appears in pattern
            char_in_pattern = char in masks
            
            # Record step
            step_data = {
                'step_number': len(steps),
                'step_type': 'bit_operation',
                'text_index': i,
                'text_char': char,
                'char_in_pattern': char_in_pattern,
                'char_mask': bin(char_mask)[2:].zfill(m),
                'char_mask_decimal': char_mask,
                'old_states': [bin(s)[2:].zfill(m) for s in old_states],
                'new_states': [bin(s)[2:].zfill(m) for s in states],
                'matches_found': matches_at_step,
                'match_mask': bin(match_mask)[2:].zfill(m),
                'operations': {
                    'shift': '<<',
                    'or': '|',
                    'and': '&'
                },
                'explanation': self._get_shift_or_explanation(
                    i, char, char_in_pattern, matches_at_step, m
                )
            }
            
            steps.append(step_data)
        
        # Summary statistics
        summary = {
            'algorithm': 'Shift-Or Algorithm',
            'text_length': n,
            'pattern_length': m,
            'max_errors': max_errors,
            'pattern_masks': mask_display,
            'total_shifts': total_shifts,
            'total_or_operations': total_or_operations,
            'bit_operations': total_shifts + total_or_operations,
            'positions_checked': n,
            'matches_found': len(matches_found),
            'match_positions': [pos for pos, _ in matches_found],
            'match_details': [{'position': pos, 'errors': err} for pos, err in matches_found],
            'time_complexity': f'O(nm/{8 * 8})' if m <= 64 else 'O(nm)',
            'space_complexity': 'O(σ + k)',
            'worst_case_comparisons': n * (max_errors + 1),
            'average_comparisons_per_position': (total_shifts + total_or_operations) / n if n > 0 else 0,
            'bit_parallel': True,
            'word_size': 64
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def _get_shift_or_explanation(self, i: int, char: str, in_pattern: bool, 
                                   matches: List[Dict], m: int) -> str:
        """Generate human-readable explanation for Shift-Or steps."""
        if len(matches) > 0:
            match_str = ', '.join([f"pos {m['position']} ({m['errors']} errors)" for m in matches])
            return f"✓ Match(es) found! Text[{i}] = '{char}'. Matches at: {match_str}"
        elif in_pattern:
            return f"➜ Text[{i}] = '{char}' (in pattern). Shifted state left, applied mask, checking for match..."
        else:
            return f"✗ Text[{i}] = '{char}' (not in pattern). Shifted state left, applied default mask (all 1s)."
    
    def get_algorithm_name(self) -> str:
        return "Shift-Or Algorithm"


# Export all matcher classes
__all__ = [
    "ApproximatePatternMatcher", 
    "LevenshteinMatcher", 
    "ShiftOrMatcher"
]

