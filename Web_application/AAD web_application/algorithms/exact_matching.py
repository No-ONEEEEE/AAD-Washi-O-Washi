"""
Exact Pattern Matching Algorithms

This module contains implementations of exact string matching algorithms
including KMP, Boyer-Moore, Suffix Tree, and Naive matching.
"""

from typing import List, Optional, Dict, Tuple
import time
from abc import ABC, abstractmethod


class PatternMatcher(ABC):
    """Abstract base class for pattern matching algorithms."""
    
    @abstractmethod
    def search(self, text: str, pattern: str) -> List[int]:
        """Search for pattern in text and return list of starting positions."""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        pass
    
    def benchmark(self, text: str, pattern: str) -> Dict[str, float]:
        """Benchmark the algorithm and return timing information."""
        start_time = time.perf_counter()
        positions = self.search(text, pattern)
        end_time = time.perf_counter()
        
        return {
            "algorithm": self.get_algorithm_name(),
            "execution_time": end_time - start_time,
            "matches_found": len(positions),
            "positions": positions
        }


class NaiveMatcher(PatternMatcher):
    """
    Naive String Matching Algorithm
    
    Time Complexity: O(nm) where n = len(text), m = len(pattern)
    Space Complexity: O(1)
    
    Best for: Short patterns, educational purposes
    """
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Search for pattern in text using naive approach.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not pattern or not text:
            return []
        
        positions = []
        n, m = len(text), len(pattern)
        
        for i in range(n - m + 1):
            # Check if pattern matches at position i
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            
            if match:
                positions.append(i)
        
        return positions
    
    def visualize_search(self, text: str, pattern: str) -> Dict:
        """
        Generate step-by-step visualization data for naive search.
        
        Returns detailed information about each step of the algorithm execution.
        """
        if not pattern or not text:
            return {"steps": [], "summary": {}}
        
        steps = []
        n, m = len(text), len(pattern)
        total_comparisons = 0
        successful_comparisons = 0
        failed_comparisons = 0
        positions_checked = 0
        matches_found = []
        
        for i in range(n - m + 1):
            positions_checked += 1
            step_comparisons = 0
            comparisons_detail = []
            
            # Check if pattern matches at position i
            match = True
            for j in range(m):
                total_comparisons += 1
                step_comparisons += 1
                
                text_char = text[i + j]
                pattern_char = pattern[j]
                is_match = text_char == pattern_char
                
                comparisons_detail.append({
                    'text_index': i + j,
                    'pattern_index': j,
                    'text_char': text_char,
                    'pattern_char': pattern_char,
                    'is_match': is_match
                })
                
                if is_match:
                    successful_comparisons += 1
                else:
                    failed_comparisons += 1
                    match = False
                    break
            
            # Record this step
            step_data = {
                'step_number': len(steps) + 1,
                'text_position': i,
                'pattern_position': 0,
                'window_start': i,
                'window_end': i + m,
                'comparisons': comparisons_detail,
                'comparisons_count': step_comparisons,
                'is_match': match,
                'match_found': match,
                'total_comparisons_so_far': total_comparisons,
                'explanation': self._get_step_explanation(i, j, match, step_comparisons, m)
            }
            
            steps.append(step_data)
            
            if match:
                matches_found.append(i)
        
        # Summary statistics
        summary = {
            'algorithm': 'Naive String Matching',
            'text_length': n,
            'pattern_length': m,
            'total_comparisons': total_comparisons,
            'successful_comparisons': successful_comparisons,
            'failed_comparisons': failed_comparisons,
            'positions_checked': positions_checked,
            'matches_found': len(matches_found),
            'match_positions': matches_found,
            'time_complexity': 'O(n*m)',
            'space_complexity': 'O(1)',
            'worst_case_comparisons': n * m,
            'average_comparisons_per_position': total_comparisons / positions_checked if positions_checked > 0 else 0,
            'efficiency': (len(matches_found) / total_comparisons * 100) if total_comparisons > 0 else 0
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def _get_step_explanation(self, i: int, j: int, match: bool, comparisons: int, m: int) -> str:
        """Generate human-readable explanation for each step."""
        if match:
            return f"✓ Full match found at position {i}! All {m} characters matched."
        else:
            return f"✗ Mismatch at character index {j}. Shifting pattern right by 1 position (from {i} to {i+1})."
    
    def get_algorithm_name(self) -> str:
        return "Naive String Matching"


class KMPMatcher(PatternMatcher):
    """
    Knuth-Morris-Pratt (KMP) Algorithm
    
    Time Complexity: O(n + m) where n = len(text), m = len(pattern)
    Space Complexity: O(m) for failure function
    
    Best for: Long patterns, repeated searches, avoiding backtracking
    """
    
    def _compute_failure_function(self, pattern: str) -> List[int]:
        """
        Compute the failure function (partial match table) for KMP algorithm.
        
        Args:
            pattern: The pattern to compute failure function for
            
        Returns:
            List representing the failure function
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
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Search for pattern in text using KMP algorithm.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not pattern or not text:
            return []
        
        positions = []
        n, m = len(text), len(pattern)
        
        # Compute failure function
        failure = self._compute_failure_function(pattern)
        
        j = 0  # Index for pattern
        for i in range(n):  # Index for text
            # Handle mismatches using failure function
            while j > 0 and text[i] != pattern[j]:
                j = failure[j - 1]
            
            # If characters match, advance pattern index
            if text[i] == pattern[j]:
                j += 1
            
            # If pattern is completely matched
            if j == m:
                positions.append(i - m + 1)
                j = failure[j - 1]  # Continue searching
        
        return positions
    
    def visualize_search(self, text: str, pattern: str) -> Dict:
        """
        Generate step-by-step visualization data for KMP search.
        
        Returns detailed information including failure function.
        """
        if not pattern or not text:
            return {"steps": [], "summary": {}}
        
        n, m = len(text), len(pattern)
        steps = []
        total_comparisons = 0
        successful_comparisons = 0
        failed_comparisons = 0
        backtrack_count = 0
        matches_found = []
        
        # Compute failure function
        failure = self._compute_failure_function(pattern)
        
        # Preprocessing step - show failure function
        preprocessing_step = {
            'step_number': 0,
            'step_type': 'preprocessing',
            'failure_function': failure.copy(),
            'pattern': pattern,
            'explanation': f"Preprocessing: Computed failure function (LPS array) = {failure}"
        }
        steps.append(preprocessing_step)
        
        # KMP search
        j = 0  # Pattern index
        i = 0  # Text index
        step_count = 0
        
        while i < n:
            step_count += 1
            old_j = j
            comparisons_detail = []
            backtracked = False
            backtrack_details = []
            
            # Handle mismatches using failure function
            while j > 0 and text[i] != pattern[j]:
                backtrack_count += 1
                backtracked = True
                old_j_before = j
                j = failure[j - 1]
                backtrack_details.append({
                    'from_j': old_j_before,
                    'to_j': j,
                    'failure_value': failure[old_j_before - 1],
                    'reason': f"Mismatch at pattern[{old_j_before}], using failure[{old_j_before - 1}] = {failure[old_j_before - 1]}"
                })
            
            # Compare current characters
            text_char = text[i]
            pattern_char = pattern[j]
            is_match = text_char == pattern_char
            total_comparisons += 1
            
            comparisons_detail.append({
                'text_index': i,
                'pattern_index': j,
                'text_char': text_char,
                'pattern_char': pattern_char,
                'is_match': is_match
            })
            
            if is_match:
                successful_comparisons += 1
                j += 1
            else:
                failed_comparisons += 1
            
            # Check if pattern is completely matched
            match_found = (j == m)
            if match_found:
                match_position = i - m + 1
                matches_found.append(match_position)
                # Continue searching
                old_j_full = j
                j = failure[j - 1]
                backtrack_details.append({
                    'from_j': old_j_full,
                    'to_j': j,
                    'failure_value': failure[old_j_full - 1],
                    'reason': f"Match found! Continue search using failure[{old_j_full - 1}] = {failure[old_j_full - 1]}"
                })
            
            # Record this step
            step_data = {
                'step_number': len(steps),
                'step_type': 'search',
                'text_index': i,
                'pattern_index': old_j,
                'new_pattern_index': j,
                'window_start': i - j + 1 if j > 0 else i,
                'window_end': i - j + 1 + m if j > 0 else i + m,
                'comparisons': comparisons_detail,
                'comparisons_count': len(comparisons_detail),
                'backtracked': backtracked,
                'backtrack_details': backtrack_details,
                'is_match': match_found,
                'match_position': match_position if match_found else None,
                'total_comparisons_so_far': total_comparisons,
                'failure_function': failure.copy(),
                'current_alignment': i - j,
                'explanation': self._get_kmp_explanation(
                    i, j, old_j, is_match, match_found, backtracked, 
                    backtrack_details, failure, text_char, pattern_char
                )
            }
            
            steps.append(step_data)
            
            # Advance text index
            i += 1
        
        # Summary statistics
        summary = {
            'algorithm': 'Knuth-Morris-Pratt (KMP)',
            'text_length': n,
            'pattern_length': m,
            'failure_function': failure,
            'total_comparisons': total_comparisons,
            'successful_comparisons': successful_comparisons,
            'failed_comparisons': failed_comparisons,
            'backtrack_count': backtrack_count,
            'text_positions_checked': n,
            'positions_checked': n,  # Alias for consistency
            'matches_found': len(matches_found),
            'match_positions': matches_found,
            'time_complexity': 'O(n+m)',
            'space_complexity': 'O(m)',
            'worst_case_comparisons': n + m,
            'average_comparisons_per_position': total_comparisons / n if n > 0 else 0,
            'efficiency': f"{((n - backtrack_count) / n * 100):.1f}%" if n > 0 else "N/A"
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def _get_kmp_explanation(self, i: int, j: int, old_j: int, is_match: bool, 
                             match_found: bool, backtracked: bool, 
                             backtrack_details: List[Dict], failure: List[int], 
                             text_char: str, pattern_char: str) -> str:
        """Generate human-readable explanation for KMP steps."""
        if match_found:
            return f"✓ Full match found at position {i - j + 1}! Pattern fully matched. Using failure function to continue search."
        elif backtracked and len(backtrack_details) > 0:
            bt = backtrack_details[0]
            return f"⚡ Backtrack: Mismatch at pattern[{bt['from_j']}]. Using failure[{bt['from_j'] - 1}] = {bt['failure_value']} to skip to pattern[{bt['to_j']}]. No text backtracking!"
        elif is_match:
            return f"✓ Match: text[{i}] = pattern[{old_j}] = '{pattern_char}'. Advancing both pointers."
        else:
            if old_j == 0:
                return f"✗ Mismatch at pattern start. Advancing text pointer from {i} to {i + 1}."
            else:
                return f"✗ Mismatch: text[{i}] ≠ pattern[{old_j}]. Checking failure function..."
    
    def get_algorithm_name(self) -> str:
        return "Knuth-Morris-Pratt (KMP)"


class BoyerMooreMatcher(PatternMatcher):
    """
    Boyer-Moore Algorithm
    
    Time Complexity: O(n/m) average case, O(nm) worst case
    Space Complexity: O(σ + m) where σ is alphabet size
    
    Best for: Large alphabets, long patterns, right-to-left scanning
    """
    
    def _bad_character_table(self, pattern: str) -> Dict[str, int]:
        """
        Create bad character table for Boyer-Moore algorithm.
        
        Args:
            pattern: The pattern to create table for
            
        Returns:
            Dictionary mapping characters to their rightmost position in pattern
        """
        table = {}
        m = len(pattern)
        
        for i in range(m):
            table[pattern[i]] = i
        
        return table
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Search for pattern in text using Boyer-Moore algorithm.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not pattern or not text:
            return []
        
        positions = []
        n, m = len(text), len(pattern)
        
        # Create bad character table
        bad_char = self._bad_character_table(pattern)
        
        i = 0  # Index for text
        while i <= n - m:
            j = m - 1  # Start from end of pattern
            
            # Compare pattern with text from right to left
            while j >= 0 and pattern[j] == text[i + j]:
                j -= 1
            
            if j < 0:
                # Pattern found
                positions.append(i)
                # Shift based on bad character rule
                if i + m < n:
                    shift = m - bad_char.get(text[i + m], -1) - 1
                else:
                    shift = 1
                i += max(1, shift)
            else:
                # Calculate shift using bad character rule
                bad_char_shift = j - bad_char.get(text[i + j], -1)
                i += max(1, bad_char_shift)
        
        return positions
    
    def visualize_search(self, text: str, pattern: str) -> Dict:
        """
        Generate step-by-step visualization data for Boyer-Moore search.
        
        Returns detailed information including bad character table and shifts.
        """
        if not pattern or not text:
            return {"steps": [], "summary": {}}
        
        n, m = len(text), len(pattern)
        steps = []
        total_comparisons = 0
        successful_comparisons = 0
        failed_comparisons = 0
        skip_count = 0
        total_shift = 0
        matches_found = []
        
        # Create bad character table
        bad_char = self._bad_character_table(pattern)
        
        # Preprocessing step - show bad character table
        preprocessing_step = {
            'step_number': 0,
            'step_type': 'preprocessing',
            'bad_char_table': bad_char.copy(),
            'pattern': pattern,
            'explanation': f"Preprocessing: Created bad character table = {bad_char}"
        }
        steps.append(preprocessing_step)
        
        # Boyer-Moore search
        i = 0  # Text alignment position
        step_count = 0
        
        while i <= n - m:
            step_count += 1
            j = m - 1  # Start from end of pattern (right-to-left)
            comparisons_detail = []
            
            # Compare pattern with text from right to left
            while j >= 0 and pattern[j] == text[i + j]:
                total_comparisons += 1
                successful_comparisons += 1
                comparisons_detail.append({
                    'text_index': i + j,
                    'pattern_index': j,
                    'text_char': text[i + j],
                    'pattern_char': pattern[j],
                    'is_match': True,
                    'direction': 'right-to-left'
                })
                j -= 1
            
            # Check if we have a match
            match_found = (j < 0)
            
            # If not a complete match, record the mismatch
            if not match_found:
                total_comparisons += 1
                failed_comparisons += 1
                mismatch_char = text[i + j]
                comparisons_detail.append({
                    'text_index': i + j,
                    'pattern_index': j,
                    'text_char': mismatch_char,
                    'pattern_char': pattern[j],
                    'is_match': False,
                    'direction': 'right-to-left'
                })
            
            # Calculate shift
            if match_found:
                match_position = i
                matches_found.append(match_position)
                
                # Calculate shift after match
                if i + m < n:
                    next_char = text[i + m]
                    shift = m - bad_char.get(next_char, -1) - 1
                    shift_reason = f"Match found! Shift by {shift} based on next char '{next_char}'"
                else:
                    shift = 1
                    shift_reason = "Match found! End of text, shift by 1"
            else:
                # Bad character rule shift
                mismatch_char = text[i + j]
                bad_char_pos = bad_char.get(mismatch_char, -1)
                shift = j - bad_char_pos
                
                if shift <= 0:
                    shift = 1
                    shift_reason = f"Mismatch at pattern[{j}]. Char '{mismatch_char}' appears later in pattern. Shift by 1"
                else:
                    shift_reason = f"Mismatch at pattern[{j}]. Char '{mismatch_char}' at position {bad_char_pos} in pattern. Shift by {shift}"
                
                if mismatch_char not in bad_char:
                    shift_reason = f"Mismatch at pattern[{j}]. Char '{mismatch_char}' not in pattern. Shift by {shift}"
            
            total_shift += shift
            
            # Characters skipped by this shift
            chars_skipped = shift - 1
            skip_count += chars_skipped
            
            # Record this step
            step_data = {
                'step_number': len(steps),
                'step_type': 'search',
                'text_position': i,
                'pattern_index': j if not match_found else -1,
                'window_start': i,
                'window_end': i + m,
                'comparisons': comparisons_detail,
                'comparisons_count': len(comparisons_detail),
                'is_match': match_found,
                'match_position': match_position if match_found else None,
                'shift': shift,
                'shift_reason': shift_reason,
                'chars_skipped': chars_skipped,
                'total_comparisons_so_far': total_comparisons,
                'bad_char_table': bad_char.copy(),
                'mismatch_char': mismatch_char if not match_found else None,
                'bad_char_position': bad_char_pos if not match_found else None,
                'scan_direction': 'right-to-left',
                'explanation': self._get_boyer_moore_explanation(
                    i, j, match_found, shift, shift_reason, comparisons_detail
                )
            }
            
            steps.append(step_data)
            
            # Advance by shift amount
            i += max(1, shift)
        
        # Summary statistics
        summary = {
            'algorithm': 'Boyer-Moore',
            'text_length': n,
            'pattern_length': m,
            'bad_char_table': bad_char,
            'total_comparisons': total_comparisons,
            'successful_comparisons': successful_comparisons,
            'failed_comparisons': failed_comparisons,
            'total_shifts': total_shift,
            'characters_skipped': skip_count,
            'alignments_checked': step_count,
            'positions_checked': step_count,  # Alias for consistency
            'matches_found': len(matches_found),
            'match_positions': matches_found,
            'time_complexity': 'O(n/m) avg, O(nm) worst',
            'space_complexity': 'O(σ+m)',
            'worst_case_comparisons': n * m,
            'average_comparisons_per_position': total_comparisons / n if n > 0 else 0,
            'skip_efficiency': f"{(skip_count / n * 100):.1f}%" if n > 0 else "N/A",
            'scan_direction': 'Right-to-Left'
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def _get_boyer_moore_explanation(self, i: int, j: int, match_found: bool, 
                                      shift: int, shift_reason: str, 
                                      comparisons: List[Dict]) -> str:
        """Generate human-readable explanation for Boyer-Moore steps."""
        if match_found:
            return f"✓ Full match found at position {i}! Scanned right-to-left. {shift_reason}"
        else:
            num_matches = sum(1 for c in comparisons if c['is_match'])
            if num_matches > 0:
                return f"↩ Partial match: {num_matches} chars matched from right. Mismatch at pattern[{j}]. {shift_reason}"
            else:
                return f"✗ Immediate mismatch at pattern end. {shift_reason}"
    
    def get_algorithm_name(self) -> str:
        return "Boyer-Moore"


class SuffixTreeNode:
    """Node class for Suffix Tree implementation."""
    
    def __init__(self):
        self.children: Dict[str, 'SuffixTreeNode'] = {}
        self.is_end: bool = False
        self.suffix_indices: List[int] = []


class SuffixTreeMatcher(PatternMatcher):
    """
    Suffix Tree Algorithm
    
    Time Complexity: O(m) search after O(n) preprocessing
    Space Complexity: O(n) for the suffix tree
    
    Best for: Multiple pattern queries, longest common substring detection
    """
    
    def __init__(self):
        self.root = None
        self.text = ""
    
    def _build_suffix_tree(self, text: str) -> SuffixTreeNode:
        """
        Build suffix tree for the given text (simplified implementation).
        
        Args:
            text: Text to build suffix tree for
            
        Returns:
            Root node of the suffix tree
        """
        root = SuffixTreeNode()
        n = len(text)
        
        # Add each suffix to the tree
        for i in range(n):
            current = root
            suffix = text[i:]
            
            for j, char in enumerate(suffix):
                if char not in current.children:
                    current.children[char] = SuffixTreeNode()
                current = current.children[char]
                current.suffix_indices.append(i)
            
            current.is_end = True
        
        return root
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Search for pattern in text using suffix tree.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not pattern or not text:
            return []
        
        # Build suffix tree if text has changed
        if self.text != text:
            self.text = text
            self.root = self._build_suffix_tree(text)
        
        # Search for pattern in suffix tree
        current = self.root
        for char in pattern:
            if char not in current.children:
                return []
            current = current.children[char]
        
        # Return the suffix indices at the final node
        # These represent all positions where the pattern starts
        return sorted(list(set(current.suffix_indices)))
    
    def visualize_search(self, text: str, pattern: str) -> Dict:
        """
        Generate step-by-step visualization data for Suffix Tree search.
        
        Returns detailed information including tree construction and traversal.
        """
        if not pattern or not text:
            return {"steps": [], "summary": {}}
        
        n, m = len(text), len(pattern)
        steps = []
        suffixes_built = 0
        tree_nodes = 0
        tree_depth = 0
        
        # Build suffix tree if text has changed
        if self.text != text:
            self.text = text
            self.root = self._build_suffix_tree(text)
        
        # Preprocessing steps - show suffix tree construction
        for i in range(min(n, 5)):  # Show first 5 suffixes as examples
            suffix = text[i:]
            preprocessing_step = {
                'step_number': len(steps),
                'step_type': 'preprocessing',
                'suffix_index': i,
                'suffix': suffix,
                'suffix_preview': suffix[:20] + '...' if len(suffix) > 20 else suffix,
                'explanation': f"Building suffix tree: Adding suffix {i} = '{suffix[:15]}{'...' if len(suffix) > 15 else ''}'"
            }
            steps.append(preprocessing_step)
        
        if n > 5:
            summary_step = {
                'step_number': len(steps),
                'step_type': 'preprocessing',
                'explanation': f"... Built remaining {n - 5} suffixes. Total: {n} suffixes in tree."
            }
            steps.append(summary_step)
        
        # Calculate tree statistics
        def count_nodes(node, depth=0):
            nonlocal tree_nodes, tree_depth
            tree_nodes += 1
            tree_depth = max(tree_depth, depth)
            for child in node.children.values():
                count_nodes(child, depth + 1)
        
        count_nodes(self.root)
        
        # Search for pattern in suffix tree - traverse step by step
        current = self.root
        path_taken = []
        nodes_visited = 0
        
        for char_idx, char in enumerate(pattern):
            nodes_visited += 1
            
            # Check if character exists in current node's children
            char_found = char in current.children
            
            if char_found:
                next_node = current.children[char]
                path_taken.append(char)
                
                # Get suffix indices at this node
                suffix_indices = sorted(list(set(next_node.suffix_indices)))[:10]  # Limit to 10 for display
                
                step_data = {
                    'step_number': len(steps),
                    'step_type': 'search',
                    'char_index': char_idx,
                    'current_char': char,
                    'path': ''.join(path_taken),
                    'char_found': True,
                    'nodes_visited': nodes_visited,
                    'suffix_indices': suffix_indices,
                    'num_matches_so_far': len(next_node.suffix_indices),
                    'depth_in_tree': len(path_taken),
                    'explanation': f"✓ Traversing tree: Found '{char}' at depth {len(path_taken)}. Path so far: '{(''.join(path_taken))}'. {len(next_node.suffix_indices)} potential matches."
                }
                
                current = next_node
            else:
                # Character not found - pattern doesn't exist
                step_data = {
                    'step_number': len(steps),
                    'step_type': 'search',
                    'char_index': char_idx,
                    'current_char': char,
                    'path': ''.join(path_taken),
                    'char_found': False,
                    'nodes_visited': nodes_visited,
                    'explanation': f"✗ Character '{char}' not found in tree at depth {len(path_taken)}. Pattern does not exist in text."
                }
                steps.append(step_data)
                
                # Return early - no matches
                summary = {
                    'algorithm': 'Suffix Tree',
                    'text_length': n,
                    'pattern_length': m,
                    'tree_nodes': tree_nodes,
                    'tree_depth': tree_depth,
                    'suffixes_indexed': n,
                    'nodes_visited': nodes_visited,
                    'pattern_found': False,
                    'matches_found': 0,
                    'match_positions': [],
                    'time_complexity': 'O(m) search, O(n) build',
                    'space_complexity': 'O(n)',
                    'preprocessing_time': 'O(n)',
                    'search_time': 'O(m)',
                    'advantage': 'Multiple patterns can be searched quickly after one-time tree construction'
                }
                
                return {
                    'steps': steps,
                    'summary': summary
                }
            
            steps.append(step_data)
        
        # Pattern found - collect all matches
        match_positions = sorted(list(set(current.suffix_indices)))
        
        # Final step showing all matches
        final_step = {
            'step_number': len(steps),
            'step_type': 'result',
            'pattern_found': True,
            'path': ''.join(path_taken),
            'matches_found': len(match_positions),
            'match_positions': match_positions,
            'explanation': f"✓ Pattern '{pattern}' found! Collected {len(match_positions)} match(es) from suffix indices: {match_positions}"
        }
        steps.append(final_step)
        
        # Summary statistics
        summary = {
            'algorithm': 'Suffix Tree',
            'text_length': n,
            'pattern_length': m,
            'tree_nodes': tree_nodes,
            'tree_depth': tree_depth,
            'suffixes_indexed': n,
            'nodes_visited': nodes_visited,
            'pattern_found': True,
            'matches_found': len(match_positions),
            'match_positions': match_positions,
            'time_complexity': 'O(m) search, O(n) build',
            'space_complexity': 'O(n)',
            'preprocessing_time': 'O(n)',
            'search_time': 'O(m)',
            'comparisons_needed': m,  # Only m character comparisons needed!
            'worst_case_comparisons': m,  # Always just m comparisons
            'total_comparisons': nodes_visited,
            'positions_checked': nodes_visited,
            'advantage': 'Multiple patterns can be searched quickly after one-time tree construction'
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def get_algorithm_name(self) -> str:
        return "Suffix Tree"


class RabinKarpMatcher(PatternMatcher):
    """
    Rabin-Karp Algorithm using rolling hash.
    
    Time Complexity: O(n+m) average case, O(nm) worst case
    Space Complexity: O(1)
    
    Uses rolling hash function for efficient pattern matching.
    Good for multiple pattern matching.
    """
    
    def __init__(self, prime: int = 101):
        """
        Initialize Rabin-Karp matcher.
        
        Args:
            prime: Prime number for hash calculation
        """
        self.prime = prime
        self.base = 256  # Number of characters in the input alphabet
    
    def _calculate_hash(self, text: str, length: int) -> int:
        """Calculate hash value for a string."""
        hash_value = 0
        for i in range(length):
            hash_value = (hash_value * self.base + ord(text[i])) % self.prime
        return hash_value
    
    def _recalculate_hash(self, text: str, old_index: int, new_index: int, 
                         old_hash: int, pattern_length: int, h: int) -> int:
        """Recalculate hash using rolling hash technique."""
        # Remove leading digit
        new_hash = (old_hash - ord(text[old_index]) * h) % self.prime
        # Add trailing digit
        new_hash = (new_hash * self.base + ord(text[new_index])) % self.prime
        # Make sure hash is positive
        if new_hash < 0:
            new_hash += self.prime
        return new_hash
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Search for pattern in text using Rabin-Karp algorithm.
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not text or not pattern or len(pattern) > len(text):
            return []
        
        n = len(text)
        m = len(pattern)
        positions = []
        
        # Calculate hash value for pattern and first window of text
        pattern_hash = self._calculate_hash(pattern, m)
        text_hash = self._calculate_hash(text, m)
        
        # Calculate h = base^(m-1) % prime for rolling hash
        h = 1
        for _ in range(m - 1):
            h = (h * self.base) % self.prime
        
        # Slide the pattern over text
        for i in range(n - m + 1):
            # Check hash values
            if pattern_hash == text_hash:
                # Hash values match, verify actual characters
                if text[i:i+m] == pattern:
                    positions.append(i)
            
            # Calculate hash for next window
            if i < n - m:
                text_hash = self._recalculate_hash(text, i, i + m, text_hash, m, h)
        
        return positions
    
    def visualize_search(self, text: str, pattern: str) -> Dict:
        """
        Generate step-by-step visualization data for Rabin-Karp search.
        
        Returns detailed information about each step including hash values.
        """
        if not text or not pattern or len(pattern) > len(text):
            return {"steps": [], "summary": {}}
        
        n = len(text)
        m = len(pattern)
        steps = []
        total_comparisons = 0
        hash_comparisons = 0
        character_comparisons = 0
        spurious_hits = 0
        matches_found = []
        
        # Calculate pattern hash
        pattern_hash = self._calculate_hash(pattern, m)
        text_hash = self._calculate_hash(text, m)
        
        # Calculate h for rolling hash
        h = 1
        for _ in range(m - 1):
            h = (h * self.base) % self.prime
        
        # Preprocessing step
        preprocessing_step = {
            'step_number': 0,
            'step_type': 'preprocessing',
            'pattern_hash': pattern_hash,
            'base': self.base,
            'prime': self.prime,
            'h_value': h,
            'explanation': f"Preprocessing: Pattern hash = {pattern_hash}, h = {h} (for rolling hash)"
        }
        steps.append(preprocessing_step)
        
        # Slide pattern over text
        for i in range(n - m + 1):
            hash_comparisons += 1
            hash_match = (pattern_hash == text_hash)
            
            # Character-by-character verification if hashes match
            comparisons_detail = []
            actual_match = False
            
            if hash_match:
                # Verify characters
                actual_match = True
                for j in range(m):
                    character_comparisons += 1
                    total_comparisons += 1
                    
                    text_char = text[i + j]
                    pattern_char = pattern[j]
                    is_match = text_char == pattern_char
                    
                    comparisons_detail.append({
                        'text_index': i + j,
                        'pattern_index': j,
                        'text_char': text_char,
                        'pattern_char': pattern_char,
                        'is_match': is_match
                    })
                    
                    if not is_match:
                        actual_match = False
                        break
                
                if hash_match and not actual_match:
                    spurious_hits += 1
            
            # Calculate next window hash
            next_hash = None
            if i < n - m:
                next_hash = self._recalculate_hash(text, i, i + m, text_hash, m, h)
            
            # Record step
            step_data = {
                'step_number': len(steps),
                'step_type': 'search',
                'text_position': i,
                'window_start': i,
                'window_end': i + m,
                'window_text': text[i:i+m],
                'current_hash': text_hash,
                'pattern_hash': pattern_hash,
                'hash_match': hash_match,
                'hash_comparisons_so_far': hash_comparisons,
                'character_comparisons': comparisons_detail,
                'character_comparisons_count': len(comparisons_detail),
                'is_match': actual_match,
                'is_spurious_hit': hash_match and not actual_match,
                'next_hash': next_hash,
                'total_comparisons_so_far': total_comparisons,
                'explanation': self._get_rabin_karp_explanation(
                    i, hash_match, actual_match, text_hash, pattern_hash, 
                    hash_match and not actual_match
                )
            }
            
            steps.append(step_data)
            
            if actual_match:
                matches_found.append(i)
            
            # Update hash for next iteration
            if i < n - m:
                text_hash = next_hash
        
        # Summary statistics
        summary = {
            'algorithm': 'Rabin-Karp',
            'text_length': n,
            'pattern_length': m,
            'pattern_hash': pattern_hash,
            'base': self.base,
            'prime': self.prime,
            'total_comparisons': total_comparisons,
            'hash_comparisons': hash_comparisons,
            'character_comparisons': character_comparisons,
            'spurious_hits': spurious_hits,
            'positions_checked': n - m + 1,
            'matches_found': len(matches_found),
            'match_positions': matches_found,
            'time_complexity': 'O(n+m) average, O(nm) worst',
            'space_complexity': 'O(1)',
            'worst_case_comparisons': n * m,
            'average_comparisons_per_position': total_comparisons / (n - m + 1) if n - m + 1 > 0 else 0,
            'hash_efficiency': f"{((hash_comparisons - spurious_hits) / hash_comparisons * 100):.1f}%" if hash_comparisons > 0 else "N/A"
        }
        
        return {
            'steps': steps,
            'summary': summary
        }
    
    def _get_rabin_karp_explanation(self, i: int, hash_match: bool, actual_match: bool, 
                                     current_hash: int, pattern_hash: int, is_spurious: bool) -> str:
        """Generate human-readable explanation for Rabin-Karp steps."""
        if is_spurious:
            return f"⚠️ Hash collision! Hash matched ({current_hash} = {pattern_hash}) but characters differ. Spurious hit at position {i}."
        elif actual_match:
            return f"✓ Full match found at position {i}! Hash matched ({current_hash} = {pattern_hash}) and all characters verified."
        elif hash_match:
            return f"✓ Hash matched ({current_hash} = {pattern_hash}), verifying characters..."
        else:
            return f"✗ Hash mismatch ({current_hash} ≠ {pattern_hash}). Skipping character comparison. Rolling hash to next position."
    
    def get_algorithm_name(self) -> str:
        return "Rabin-Karp"


# Export all matcher classes
__all__ = [
    "PatternMatcher", 
    "NaiveMatcher", 
    "KMPMatcher", 
    "BoyerMooreMatcher", 
    "SuffixTreeMatcher",
    "RabinKarpMatcher"
]
