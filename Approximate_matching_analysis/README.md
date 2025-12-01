# Fuzzy Matching Analysis (DNA FASTA Benchmarks)

## Overview
This project benchmarks approximate string matching algorithms (Regex exact baseline, Shift-Or bit-parallel, and banded Levenshtein) on synthetic DNA FASTA datasets. It measures execution time and memory usage across varying sequence lengths, query lengths, and edit distance thresholds (0–5), producing detailed and summary CSV/TXT reports along with comparison graphs.

## Key Features
- Synthetic DNA dataset generator supporting multiple sequence types (random, high GC, repetitive, low complexity, mutated)
- Benchmark harness sampling query substrings and introducing controlled substitutions
- Algorithms: Python `re` (exact), Shift-Or (bit-parallel), Levenshtein (Ukkonen banded)
- Automatic CSV → human-readable TXT conversion (detailed + summary)
- Time and memory plots; per-edit overall comparison graphs
- Markdown report summarizing findings
- Organized `Results/` folder for deliverables

## Folder Structure
```
Fuzzy_matching_analysis/
  benchmarks/
    benchmark.py            # Main benchmarking harness
    report_generator.py     # Generates plots + markdown report
  datasets/
    generate_datasets.py    # DNA FASTA generator
    dna_test/               # Example small test FASTA files
  src/
    shift_or.py             # Shift-Or (bit-parallel) implementation
    levenshtein.py          # Banded Levenshtein implementation
    regex_wrapper.py        # Wrapper for Python regex baseline
  Results/                  # Collected outputs (CSV, TXT, graphs, reports)
  requirements.txt          # Runtime dependencies
  README.md                 # This guide
```

## Requirements
| Component        | Version/Notes |
|------------------|--------------|
| Python           | 3.10+ (tested on 3.13) |
| OS               | Windows (PowerShell examples below) |
| Dependencies     | numpy, pandas, matplotlib, psutil |

Contents of `requirements.txt`:
```
numpy
pandas
matplotlib
psutil
```

## Installation
Create and activate a virtual environment (recommended) then install dependencies.

```powershell
# From project root
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Verify imports:
```powershell
python - <<'PY'
import numpy, pandas, matplotlib, psutil
print('Dependencies OK')
PY
```

## Generating Datasets
Use the DNA generator to create FASTA files.
Default types: random, repetitive, low_complexity, high_gc, mutated.

Examples:
```powershell
# Small quick set
python datasets/generate_datasets.py --out datasets/dna_quick --types random high_gc --lengths 100 500 --count 2

# Range syntax (100 to 500 step 100) & all types
python datasets/generate_datasets.py --out datasets/dna_range --types random high_gc repetitive low_complexity mutated --lengths 100-500:100 --count 3

# Larger lengths
python datasets/generate_datasets.py --out datasets/dna_large --lengths 100 1000 10000 --count 3
```

Arguments:
- `--out`: output directory
- `--types`: sequence types list
- `--lengths`: list or range spec (`start-end:step`)
- `--count`: sequences per (type,length)
- `--gc-bias`: GC probability for random base generation
- `--mutation-rate`: substitution probability for mutated type

## Running Benchmarks
Run `benchmark.py` pointing to a directory of FASTA files. It will:
1. Read sequences
2. Sample query substrings (`--qlens`, `--samples-per-len`)
3. Apply edit substitutions for each value in `--edits`
4. Time & measure memory for each algorithm
5. Write detailed CSV, auto-convert to TXT, produce summary CSV/TXT, create basic per-edit time plots

Example (using provided test FASTA):
```powershell
python benchmarks/benchmark.py --datasets datasets/dna_test --out benchmarks/results_small_5edits.csv --qlens 20 50 --samples-per-len 3 --edits 0 1 2 3 4 5 --repeats 3 --plots-out benchmarks/plots
```

Key CLI flags:
- `--datasets`: directory containing `.fasta` files
- `--out`: output CSV path (detailed measurements)
- `--qlens`: list of query lengths to sample from sequences
- `--samples-per-len`: queries sampled per query length per sequence
- `--edits`: list of edit thresholds to test (substitution mutations applied)
- `--repeats`: repetitions per measurement for averaging
- `--plots-out`: directory for benchmark latency plots

Outputs produced after run:
- Detailed CSV: `benchmarks/results_small_5edits.csv`
- Detailed TXT: `benchmarks/results_small_5edits.txt`
- Summary CSV: `benchmarks/results_small_5edits_summary.csv`
- Summary TXT: `benchmarks/results_small_5edits_summary.txt`
- Time plots: `benchmarks/plots/time_by_len_edits_*.png`

## Generating Extended Report & Graphs
Use `report_generator.py` to create additional graphs (time vs sequence length per edit, time vs query length, time vs edits, memory boxplot) and a markdown report.

```powershell
python benchmarks/report_generator.py --results benchmarks/results_small_5edits.csv --out-dir bench_report
```

Outputs:
- `bench_report/time_vs_len_edits_X.png`
- `bench_report/time_vs_query_len.png`
- `bench_report/time_vs_edits.png`
- `bench_report/memory_boxplot.png`
- `bench_report/REPORT.md`

## Organized Results Folder
The project may collect final artifacts into `Results/` with subfolders:
```
Results/
  Text/
    CSV/                      # Detailed & summary CSVs
    TXT/                      # Human-readable conversions
  Graphs/
    Time/                     # Time-related plots
    Memory/                   # Memory-related plots
    Overall/                  # Per-edit comparison bars (time & memory)
```
If not already created, you can copy artifacts manually or add automation. Example manual copy:
```powershell
New-Item -ItemType Directory -Force Results/Graphs/Overall | Out-Null
Copy-Item benchmarks/results_small_5edits*.csv Results/Text/CSV/
Copy-Item benchmarks/results_small_5edits*.txt Results/Text/TXT/
Copy-Item bench_report/*.png Results/Graphs/Time/
Copy-Item bench_report/memory_boxplot.png Results/Graphs/Memory/
```

## Reproducing / Customizing Experiments
Adjust:
- Query lengths: `--qlens 30 60 120`
- Edit range: `--edits 0 2 4`
- Repeats for stability: `--repeats 5`
- Larger datasets: generate with longer lengths (e.g., 10000, 50000) but expect longer runtimes.

For very large sequences, consider reducing `--samples-per-len` or the set of `--edits` to control runtime.

## Interpreting Measurements
- Times are averaged over `--repeats` runs.
- Memory is average RSS (resident set size) in bytes; TXT shows MB.
- Regex baseline only tests exact matching; approximate behavior simulated through mutated queries fed into exact search (so regex times remain independent of edit threshold except for changed query content).
- Shift-Or is efficient for small pattern lengths and limited edits.
- Levenshtein (banded) handles larger edit allowances but scales slower.

## Troubleshooting
| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| "No FASTA files found" | Wrong `--datasets` path | Point to directory with `.fasta` files |
| Import errors | Missing dependencies | `pip install -r requirements.txt` |
| Slow runtime | Too many queries / large edits range | Reduce `--samples-per-len` or edit list |
| Memory stable at ~60MB | psutil reports process RSS, not per-function | This is expected; algorithm memory differences are subtle |

## Example End-to-End Workflow
```powershell
# 1. Setup
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# 2. Generate datasets
python datasets/generate_datasets.py --out datasets/dna_experiment --lengths 100-1000:300 --count 2

# 3. Run benchmark
python benchmarks/benchmark.py --datasets datasets/dna_experiment --out benchmarks/exp_results.csv --qlens 30 60 --samples-per-len 4 --edits 0 1 2 3 --repeats 3 --plots-out benchmarks/exp_plots

# 4. Generate extended report
python benchmarks/report_generator.py --results benchmarks/exp_results.csv --out-dir exp_report

# 5. Organize outputs (optional)
New-Item -ItemType Directory -Force Results/Text/CSV,Results/Text/TXT,Results/Graphs/Time,Results/Graphs/Memory,Results/Graphs/Overall | Out-Null
Copy-Item benchmarks/exp_results*.csv Results/Text/CSV/
Copy-Item benchmarks/exp_results*.txt Results/Text/TXT/
Copy-Item exp_report/time_vs_* Results/Graphs/Time/
Copy-Item exp_report/memory_boxplot.png Results/Graphs/Memory/
```

## Extending the Project
Ideas:
- Add insertion/deletion mutations (true edit distance beyond substitutions)
- Integrate a more optimized Levenshtein implementation (e.g., C extension)
- Include additional algorithms (e.g., Bitap with wildcard, Aho-Corasick variants)
- Parallelize benchmarking (multiprocessing) for large datasets
- Add statistical significance tests between algorithms

## License
Specify a license (e.g., MIT) here if distributing publicly.

## Citation
If used in academic work, cite the algorithms' original papers (Myers 1999, Ukkonen 1985) and this repository.

---
For quick help run: `python benchmarks/benchmark.py -h` or `python datasets/generate_datasets.py -h`.
# Fuzzy Matching Analysis

This project benchmarks two approximate/fuzzy string matching approaches:

- Shift-Or / Myers bit-parallel (implemented in `src/shift_or.py`)
- Levenshtein with Ukkonen banded search (implemented in `src/levenshtein.py`)

We compare them to Python's built-in `re` module (exact matching) using the
benchmark harness in `benchmarks/benchmark.py`.

Setup
1. Create and activate a virtualenv (PowerShell):
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```
pip install -r requirements.txt
```

Generate datasets
```
python datasets\generate_datasets.py --out datasets/sample_texts --size 3
```

Run benchmarks
```
python benchmarks\benchmark.py --datasets datasets/sample_texts --patterns-file patterns.txt --out results.csv --max-dist 1 --repeats 3
```

Outputs
- `results.csv` contains average time (s) and RSS memory (bytes) for each algorithm/pattern.

Next steps
- Run the harness on larger real-world datasets (logs, DNA, code tokens).
- Add plotting and summary report generation.
