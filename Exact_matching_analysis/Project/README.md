# DNA Pattern Matching Algorithms - Performance Analysis

This project implements and benchmarks various string matching algorithms on DNA sequences, comparing their performance across different pattern categories.

# DNA Pattern Matching Algorithms - Performance Analysis

This project implements and benchmarks various string matching algorithms on DNA sequences, comparing their performance across different pattern categories.

## Quick Start

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Installation & Setup

**Step 1: Install Required Packages**
```bash
# Install all required Python packages
pip3 install pandas matplotlib seaborn psutil numpy

# OR if you prefer using pip
pip install pandas matplotlib seaborn psutil numpy

# OR if you have conda
conda install pandas matplotlib seaborn psutil numpy
```

**Step 2: Verify Installation**
```bash
python3 -c "import pandas, matplotlib, seaborn, psutil, numpy; print('All packages installed successfully!')"
```

### Execution Commands

```bash
# Navigate to project directory
cd /Users/saketh/Desktop/Sem-3/AAD/Project

# Run complete benchmark analysis
python3 benchmark_patterns.py

# View results after completion
cat results/analysis_report_final.txt
cat results/raw_results_final.csv
```

### Quick Test Run (if you want to test with fewer patterns)
```bash
# Run with limited patterns for testing
python3 benchmark_patterns.py --limit-per-category 2 --only-categories average

# Run without plots if you want faster execution
python3 benchmark_patterns.py --no-plots
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'seaborn' (or other packages)**
```bash
# Solution: Install missing packages
pip3 install seaborn pandas matplotlib psutil numpy

# If pip3 doesn't work, try:
pip install seaborn pandas matplotlib psutil numpy
```

**2. Permission errors**
```bash
# Try with user flag
pip3 install --user seaborn pandas matplotlib psutil numpy
```

**3. Python version issues**
```bash
# Check Python version (should be 3.6+)
python3 --version

# If using different Python version
python -m pip install seaborn pandas matplotlib psutil numpy
```

**4. Virtual environment setup (recommended)**
```bash
# Create virtual environment
python3 -m venv pattern_matching_env

# Activate it
source pattern_matching_env/bin/activate  # On macOS/Linux
# pattern_matching_env\Scripts\activate  # On Windows

# Install packages
pip install seaborn pandas matplotlib psutil numpy

# Run benchmark
python3 benchmark_patterns.py
```

## Project Structure

```
├── algorithms/                 # Algorithm implementations
│   ├── naive_string_matching.py
│   ├── kmp_algorithm.py
│   ├── boyer_moore_algorithm.py
│   ├── rabin_karp_algorithm.py
│   └── suffix_tree_algorithm.py
├── patterns/                   # Test pattern files
├── sequences/                  # E. coli genome data
├── results/                    # Benchmark results
├── benchmark_patterns.py       # Main benchmark script
└── README.md
```

## Algorithms Tested

1. **Naive String Matching** - O(nm) brute force
2. **Knuth-Morris-Pratt (KMP)** - O(n+m) with failure function
3. **Boyer-Moore** - O(nm) worst, O(n/m) best case
4. **Rabin-Karp** - O(n+m) expected with rolling hash
5. **Suffix Tree** - O(m) search after O(n) preprocessing
6. **Regex** - Optimized C implementation baseline

## Pattern Categories

- **Average**: Real genome subsequences (55 patterns)
- **Repetitive**: Worst-case patterns (9 patterns)
- **Rare**: Sparse occurrence patterns (10 patterns)  
- **Zero-occurrence**: Non-matching patterns (35 patterns)
- **Rabin-Karp-Adversarial**: Hash collision patterns (10 patterns)

## Results

Performance analysis includes:
- Execution time (with 4-decimal precision)
- Memory usage
- Pattern match counts
- Preprocessing overhead
- Total time (runtime + preprocessing)

See `results/analysis_report_final.txt` for detailed performance comparison.