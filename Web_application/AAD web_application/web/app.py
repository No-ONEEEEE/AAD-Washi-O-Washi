"""
Washi O Washi - Interactive Web-Based Pattern Matching Lab

Flask web application for exploring and visualizing string matching algorithms.
Features: Algorithm visualization, comparison, custom dataset upload, and educational content.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import time
import json
from werkzeug.utils import secure_filename
from typing import Dict, List, Any
import traceback

# Add parent directory to path to import algorithms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.exact_matching import (
    KMPMatcher, BoyerMooreMatcher, NaiveMatcher, SuffixTreeMatcher,
    RabinKarpMatcher
)
from algorithms.approximate_matching import (
    LevenshteinMatcher, ShiftOrMatcher
)

app = Flask(__name__)
app.secret_key = 'washi_o_washi_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'fasta', 'fa', 'seq', 'fastq', 'dat'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize algorithm instances
EXACT_ALGORITHMS = {
    'naive': NaiveMatcher(),
    'kmp': KMPMatcher(),
    'boyer_moore': BoyerMooreMatcher(),
    'suffix_tree': SuffixTreeMatcher(),
    'rabin_karp': RabinKarpMatcher()
}

APPROXIMATE_ALGORITHMS = {
    'levenshtein': LevenshteinMatcher(),
    'shift_or': ShiftOrMatcher()
}

# Combine all algorithms
ALL_ALGORITHMS = {**EXACT_ALGORITHMS, **APPROXIMATE_ALGORITHMS}

# Algorithm metadata for educational content
ALGORITHM_INFO = {
    'naive': {
        'name': 'Naive String Matching',
        'category': 'Exact Matching',
        'time_complexity': 'O(nm)',
        'space_complexity': 'O(1)',
        'description': 'Simple brute-force approach that checks every position in the text for a pattern match.',
        'how_it_works': [
            'Start from the first position in the text',
            'Compare pattern character by character with text',
            'If mismatch occurs, shift pattern by one position',
            'Repeat until end of text is reached'
        ],
        'best_for': 'Small patterns and texts, educational purposes',
        'worst_case': 'Text with many partial matches',
        'color': '#FF6B6B'
    },
    'kmp': {
        'name': 'Knuth-Morris-Pratt (KMP)',
        'category': 'Exact Matching',
        'time_complexity': 'O(n+m)',
        'space_complexity': 'O(m)',
        'description': 'Uses preprocessing to create a failure function that avoids redundant comparisons.',
        'how_it_works': [
            'Preprocess pattern to create failure function (partial match table)',
            'Use failure function to skip unnecessary comparisons',
            'Never backtrack in the text, only in the pattern',
            'Achieves linear time complexity'
        ],
        'best_for': 'General text processing, patterns with repetitive prefixes',
        'worst_case': 'Performs consistently well in all cases',
        'color': '#4ECDC4'
    },
    'boyer_moore': {
        'name': 'Boyer-Moore',
        'category': 'Exact Matching',
        'time_complexity': 'O(nm) worst, O(n/m) average',
        'space_complexity': 'O(σ) where σ is alphabet size',
        'description': 'Scans pattern from right to left using bad character and good suffix heuristics.',
        'how_it_works': [
            'Preprocess pattern to create bad character table',
            'Scan pattern from right to left',
            'On mismatch, use heuristics to skip positions',
            'Can skip multiple characters in favorable cases'
        ],
        'best_for': 'Large alphabets (English text), long patterns',
        'worst_case': 'Highly repetitive patterns in repetitive text',
        'color': '#95E1D3'
    },
    'suffix_tree': {
        'name': 'Suffix Tree',
        'category': 'Exact Matching',
        'time_complexity': 'O(m) search, O(n) preprocessing',
        'space_complexity': 'O(n)',
        'description': 'Builds a tree structure of all suffixes for fast pattern matching.',
        'how_it_works': [
            'Build suffix tree from text (one-time preprocessing)',
            'Search pattern by traversing tree',
            'Each edge represents a substring',
            'Efficient for multiple pattern queries on same text'
        ],
        'best_for': 'Multiple searches on same text, substring queries',
        'worst_case': 'Memory intensive for large texts',
        'color': '#F38181'
    },
    'levenshtein': {
        'name': 'Levenshtein Distance',
        'category': 'Approximate Matching',
        'time_complexity': 'O(nm)',
        'space_complexity': 'O(nm)',
        'description': 'Computes edit distance allowing insertions, deletions, and substitutions.',
        'how_it_works': [
            'Build dynamic programming matrix',
            'Calculate minimum edit operations needed',
            'Supports insertions, deletions, substitutions',
            'Returns positions with distance below threshold'
        ],
        'best_for': 'Fuzzy matching, spell checking, DNA mutation detection',
        'worst_case': 'Large patterns with high error tolerance',
        'color': '#AA96DA'
    },
    'shift_or': {
        'name': 'Shift-Or Algorithm',
        'category': 'Approximate Matching',
        'time_complexity': 'O(nm/w) where w is word size',
        'space_complexity': 'O(σ)',
        'description': 'Bit-parallel algorithm for exact and approximate matching with k mismatches.',
        'how_it_works': [
            'Use bit vectors to represent pattern state',
            'Perform bitwise operations for parallel matching',
            'Track approximate matches with bit masks',
            'Very fast for small error tolerances'
        ],
        'best_for': 'Small patterns, low error tolerance, DNA sequences',
        'worst_case': 'Pattern length exceeds word size',
        'color': '#FCBAD3'
    },
    'rabin_karp': {
        'name': 'Rabin-Karp',
        'category': 'Exact Matching',
        'time_complexity': 'O(n+m) average, O(nm) worst',
        'space_complexity': 'O(1)',
        'description': 'Rolling hash-based pattern matching algorithm.',
        'how_it_works': [
            'Calculate hash value for pattern',
            'Use rolling hash for text windows',
            'Compare hash values for quick filtering',
            'Verify matches character-by-character'
        ],
        'best_for': 'Multiple pattern matching, plagiarism detection',
        'worst_case': 'Many hash collisions',
        'color': '#FFB6B9'
    }
}

# Sample datasets
SAMPLE_DATASETS = {
    'dna_simple': {
        'name': 'Simple DNA Sequence',
        'text': 'ATCGATCGATCGATCGTAGCTAGCTAGCTAGC',
        'pattern': 'ATCG',
        'description': 'Basic DNA sequence with repeating pattern'
    },
    'dna_ecoli': {
        'name': 'E. coli Gene Fragment',
        'text': 'ATGGCAATTCAGGGTATCGACAATGAAGCGACCCCGCTGGCCGAAGAGCTGGCAAAAG' +
                'ATGCCGATGAACTGGTACACGCCTGGGCGAAATTCTGGGTGCCCGCACCCACACCGTCGA',
        'pattern': 'ATG',
        'description': 'E. coli genome fragment searching for start codon'
    },
    'dna_mutation': {
        'name': 'Mutation Detection',
        'text': 'ATCGATCGXTCGATCG',
        'pattern': 'ATCGATCG',
        'description': 'DNA with mutation (X represents mutation)'
    },
    'english': {
        'name': 'English Text',
        'text': 'The quick brown fox jumps over the lazy dog. The fox was very quick.',
        'pattern': 'quick',
        'description': 'Simple English sentence'
    },
    'repetitive': {
        'name': 'Highly Repetitive',
        'text': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
        'pattern': 'AAA',
        'description': 'Tests algorithm behavior with repetitive patterns'
    }
}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page with algorithm selection and input."""
    return render_template('index.html', 
                         algorithms=ALGORITHM_INFO,
                         samples=SAMPLE_DATASETS)


@app.route('/api/search', methods=['POST'])
def search():
    """Perform pattern matching with selected algorithms."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        pattern = data.get('pattern', '').strip()
        selected_algorithms = data.get('algorithms', [])
        search_type = data.get('search_type', 'exact')
        max_errors = int(data.get('max_errors', 1))
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        if not pattern:
            return jsonify({'error': 'Pattern cannot be empty'}), 400
        if not selected_algorithms:
            return jsonify({'error': 'Please select at least one algorithm'}), 400
        
        results = {}
        
        for algo_name in selected_algorithms:
            if algo_name not in ALL_ALGORITHMS:
                continue
                
            try:
                start_time = time.perf_counter()
                algo_instance = ALL_ALGORITHMS[algo_name]
                
                # Check if approximate matching
                if algo_name in APPROXIMATE_ALGORITHMS:
                    raw_matches = algo_instance.search(text, pattern, max_errors)
                    # Approximate algorithms return (position, distance) tuples
                    # Convert to just positions for display
                    matches = [pos if isinstance(pos, int) else pos[0] for pos in raw_matches]
                else:
                    matches = algo_instance.search(text, pattern)
                
                execution_time = time.perf_counter() - start_time
                
                results[algo_name] = {
                    'matches': matches,
                    'count': len(matches),
                    'execution_time': execution_time,
                    'execution_time_ms': execution_time * 1000,
                    'algorithm_name': algo_instance.get_algorithm_name(),
                    'success': True
                }
                    
            except Exception as e:
                results[algo_name] = {
                    'error': str(e),
                    'algorithm_name': ALL_ALGORITHMS[algo_name].get_algorithm_name(),
                    'success': False,
                    'traceback': traceback.format_exc()
                }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/algorithm-info/<algo_name>')
def get_algorithm_info(algo_name):
    """Get detailed information about a specific algorithm."""
    if algo_name in ALGORITHM_INFO:
        return jsonify(ALGORITHM_INFO[algo_name])
    return jsonify({'error': 'Algorithm not found'}), 404


@app.route('/api/all-algorithms')
def get_all_algorithms():
    """Get list of all available algorithms with metadata."""
    return jsonify(ALGORITHM_INFO)


@app.route('/api/sample-datasets')
def get_sample_datasets():
    """Get all sample datasets."""
    return jsonify(SAMPLE_DATASETS)


@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    """Run comprehensive benchmark on selected algorithms."""
    try:
        data = request.get_json()
        selected_algorithms = data.get('algorithms', list(ALL_ALGORITHMS.keys()))
        text_sizes = data.get('text_sizes', [100, 1000, 10000])
        pattern_length = data.get('pattern_length', 10)
        
        results = {}
        
        for size in text_sizes:
            # Generate test data
            test_text = 'ATCG' * (size // 4)
            test_pattern = 'ATCG' * (pattern_length // 4)
            
            results[size] = {}
            
            for algo_name in selected_algorithms:
                if algo_name not in ALL_ALGORITHMS:
                    continue
                
                try:
                    algo_instance = ALL_ALGORITHMS[algo_name]
                    benchmark_result = algo_instance.benchmark(test_text, test_pattern)
                    
                    results[size][algo_name] = {
                        'execution_time': benchmark_result['execution_time'],
                        'execution_time_ms': benchmark_result['execution_time'] * 1000,
                        'matches_found': benchmark_result['matches_found']
                    }
                except Exception as e:
                    results[size][algo_name] = {
                        'error': str(e)
                    }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload for custom datasets."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Limit to first 100KB for display
            if len(content) > 100000:
                content = content[:100000] + '\n... (file truncated for display)'
            
            return jsonify({
                'success': True,
                'filename': filename,
                'content': content,
                'size': os.path.getsize(filepath)
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def visualize_algorithm():
    """Generate step-by-step visualization data for an algorithm."""
    try:
        data = request.get_json()
        algorithm = data.get('algorithm')
        text = data.get('text', '')
        pattern = data.get('pattern', '')
        
        if not algorithm or not text or not pattern:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if algorithm not in ALL_ALGORITHMS:
            return jsonify({'error': f'Algorithm {algorithm} not found'}), 404
        
        algo_instance = ALL_ALGORITHMS[algorithm]
        
        # Check if algorithm has visualization method
        if hasattr(algo_instance, 'visualize_search'):
            viz_data = algo_instance.visualize_search(text, pattern)
            return jsonify(viz_data)
        else:
            return jsonify({'error': f'Algorithm {algorithm} does not support visualization yet'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'algorithms_loaded': len(ALL_ALGORITHMS),
        'exact_algorithms': list(EXACT_ALGORITHMS.keys()),
        'approximate_algorithms': list(APPROXIMATE_ALGORITHMS.keys())
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
