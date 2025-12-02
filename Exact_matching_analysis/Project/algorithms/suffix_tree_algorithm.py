"""
Suffix Tree Algorithm
O(m) search time after O(n) preprocessing using Ukkonen's algorithm.
Enables multiple pattern queries and longest common substring detection.
"""

class SuffixTreeNode:
    """Node in the suffix tree"""
    
    def __init__(self, start=None, end=None):
        """
        Initialize a suffix tree node.
        
        Args:
            start: Start index of the edge label
            end: End index of the edge label (can be a reference for open edges)
        """
        self.children = {}
        self.suffix_link = None
        self.start = start
        self.end = end
        self.suffix_index = -1
    
    def edge_length(self):
        """Get the length of the edge to this node"""
        if self.start is None:
            return 0
        return self.end[0] - self.start + 1 if isinstance(self.end, list) else self.end - self.start + 1


class SuffixTree:
    """
    Suffix Tree implementation using Ukkonen's algorithm.
    Provides O(n) construction and O(m) pattern search.
    """
    
    def __init__(self, text):
        """
        Build a suffix tree for the given text.
        
        Args:
            text: The text to build the suffix tree for
        """
        self.text = text + "$"  # Add terminator
        self.n = len(self.text)
        self.root = SuffixTreeNode()
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining_suffix_count = 0
        self.leaf_end = [-1]
        self.split_end = None
        
        # Build the suffix tree
        self._build_suffix_tree()
    
    def _new_node(self, start, end=None):
        """Create a new node in the suffix tree"""
        node = SuffixTreeNode(start, end if end is not None else self.leaf_end)
        return node
    
    def _edge_length(self, node):
        """Get edge length for a node"""
        return node.edge_length()
    
    def _walk_down(self, node):
        """Walk down the tree if active_length is greater than edge length"""
        length = self._edge_length(node)
        if self.active_length >= length:
            self.active_edge += length
            self.active_length -= length
            self.active_node = node
            return True
        return False
    
    def _extend_suffix_tree(self, pos):
        """Extend the suffix tree by adding character at position pos"""
        self.leaf_end[0] = pos
        self.remaining_suffix_count += 1
        last_new_node = None
        
        while self.remaining_suffix_count > 0:
            if self.active_length == 0:
                self.active_edge = pos
            
            if self.text[self.active_edge] not in self.active_node.children:
                # Create new leaf edge
                self.active_node.children[self.text[self.active_edge]] = self._new_node(pos)
                
                if last_new_node is not None:
                    last_new_node.suffix_link = self.active_node
                    last_new_node = None
            else:
                next_node = self.active_node.children[self.text[self.active_edge]]
                
                if self._walk_down(next_node):
                    continue
                
                # Check if current character is already in the tree
                if self.text[next_node.start + self.active_length] == self.text[pos]:
                    if last_new_node is not None and self.active_node != self.root:
                        last_new_node.suffix_link = self.active_node
                        last_new_node = None
                    
                    self.active_length += 1
                    break
                
                # Split the edge
                split_end = [next_node.start + self.active_length - 1]
                split_node = self._new_node(next_node.start, split_end)
                self.active_node.children[self.text[self.active_edge]] = split_node
                
                # Create new leaf
                split_node.children[self.text[pos]] = self._new_node(pos)
                next_node.start += self.active_length
                split_node.children[self.text[next_node.start]] = next_node
                
                if last_new_node is not None:
                    last_new_node.suffix_link = split_node
                
                last_new_node = split_node
            
            self.remaining_suffix_count -= 1
            
            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining_suffix_count + 1
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link if self.active_node.suffix_link else self.root
    
    def _build_suffix_tree(self):
        """Build the suffix tree using Ukkonen's algorithm"""
        for i in range(self.n):
            self._extend_suffix_tree(i)
    
    def search(self, pattern):
        """
        Search for a pattern in the text using the suffix tree.
        
        Args:
            pattern: The pattern to search for
            
        Returns:
            List of starting indices where pattern is found
        """
        if not pattern:
            return []
        
        # Traverse the tree following the pattern
        node = self.root
        i = 0
        
        while i < len(pattern):
            if pattern[i] not in node.children:
                return []
            
            child = node.children[pattern[i]]
            j = child.start
            end = child.end[0] if isinstance(child.end, list) else child.end
            
            while j <= end and i < len(pattern):
                if self.text[j] != pattern[i]:
                    return []
                i += 1
                j += 1
            
            node = child
        
        # Pattern found, collect all leaf nodes (suffix indices)
        matches = []
        self._collect_suffix_indices(node, matches)
        return sorted(matches)
    
    def _collect_suffix_indices(self, node, matches):
        """Collect all suffix indices from leaf nodes under this node"""
        if not node.children:  # Leaf node
            suffix_idx = self.n - (self.leaf_end[0] if isinstance(node.end, list) else node.end) - 1
            if suffix_idx >= 0:
                matches.append(suffix_idx)
            return
        
        for child in node.children.values():
            self._collect_suffix_indices(child, matches)
    
    def longest_common_substring(self, text2):
        """
        Find the longest common substring between the tree's text and text2.
        
        Args:
            text2: Second text to compare
            
        Returns:
            Tuple of (longest common substring, length)
        """
        max_length = 0
        max_substring = ""
        
        # Try all substrings of text2
        for i in range(len(text2)):
            for j in range(i + 1, len(text2) + 1):
                substring = text2[i:j]
                if self.search(substring):
                    if len(substring) > max_length:
                        max_length = len(substring)
                        max_substring = substring
        
        return max_substring, max_length


def main():
    """Example usage of Suffix Tree algorithm"""
    # Example 1: DNA sequence matching
    dna_sequence = "ACGTACGTACGT"
    
    print("Suffix Tree Algorithm - Pattern Matching")
    print("=" * 50)
    print(f"Text: {dna_sequence}")
    
    # Build suffix tree
    suffix_tree = SuffixTree(dna_sequence)
    
    # Search for patterns
    pattern1 = "ACGT"
    matches = suffix_tree.search(pattern1)
    print(f"Pattern '{pattern1}' found at indices: {matches}")
    print()
    
    pattern2 = "CGT"
    matches = suffix_tree.search(pattern2)
    print(f"Pattern '{pattern2}' found at indices: {matches}")
    print()
    
    # Example 2: Multiple pattern queries (efficient with suffix tree)
    text = "BANANA"
    suffix_tree = SuffixTree(text)
    
    print(f"Text: {text}")
    patterns = ["ANA", "NAN", "BAN", "NA"]
    
    for pattern in patterns:
        matches = suffix_tree.search(pattern)
        print(f"Pattern '{pattern}' found at indices: {matches}")
    print()
    
    # Example 3: Longest common substring
    text1 = "ABABC"
    text2 = "BABCA"
    
    suffix_tree = SuffixTree(text1)
    lcs, length = suffix_tree.longest_common_substring(text2)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Longest Common Substring: '{lcs}' (length: {length})")
    print()
    
    # Time complexity demonstration
    print("Time Complexity:")
    print("- Construction: O(n) using Ukkonen's algorithm")
    print("- Search: O(m) for pattern of length m")
    print("- Space: O(n)")
    print("- Multiple queries: Each query is O(m), very efficient!")


if __name__ == "__main__":
    main()
