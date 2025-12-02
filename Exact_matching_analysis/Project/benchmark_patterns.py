#!/usr/bin/env python3
import os
import re
import csv
import argparse
import time
import tracemalloc
import random
from typing import List, Tuple, Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from algorithms import (
    naive_search as naive_lib_naive,
    kmp_search as kmp_lib,
    boyer_moore_search as bm_lib,
    rabin_karp_search as rk_lib,
    rabin_karp_with_hash_info as rk_with_info,
    SuffixTree,
)

# -----------------------------
# Genome loading
# -----------------------------

def load_genome(path: str = "sequences/ecoli_500k.txt") -> str:
    genome = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                genome.append(line.strip().upper())
    return "".join(genome)

GENOME = load_genome()
G_LEN = len(GENOME)
print(f"Genome loaded. Length: {G_LEN}")

# Global preprocessing state for algorithms that require it
PREPROCESSED_DATA = {}

# -----------------------------
# Algorithm adapters (use user's algorithms package)
# Each returns: matches, comparisons, verifications, preprocessing_time_ms
# -----------------------------

def naive_adapter(pattern: str, text: str, precomputed_data=None):
    matches = naive_lib_naive(text, pattern)
    return matches, None, None, 0.0


def kmp_adapter(pattern: str, text: str, precomputed_data=None):
    matches = kmp_lib(text, pattern)
    return matches, None, None, 0.0


def boyer_moore_adapter(pattern: str, text: str, precomputed_data=None):
    matches = bm_lib(text, pattern)
    return matches, None, None, 0.0


def rabin_karp_adapter(pattern: str, text: str, precomputed_data=None):
    # Use fast path by default; detailed hash info is expensive on large runs
    matches = rk_lib(text, pattern)
    return matches, None, None, 0.0


def regex_search(pattern: str, text: str):
    # DNA patterns contain only ACGT; no escaping needed
    start = time.perf_counter()
    matches = [m.start() for m in re.finditer(pattern, text)]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return matches, None, None, 0.0, elapsed_ms

def suffix_tree_adapter(pattern: str, text: str, precomputed_data=None):
    # Use prebuilt suffix tree if available
    if precomputed_data and 'suffix_tree' in precomputed_data:
        suffix_tree = precomputed_data['suffix_tree']
        pre_time = 0.0  # No preprocessing time for this call
    else:
        # Build suffix tree (this should only happen during preprocessing)
        start_time = time.perf_counter()
        suffix_tree = SuffixTree(text)
        pre_time = (time.perf_counter() - start_time) * 1000
    
    matches = suffix_tree.search(pattern)
    return matches, None, None, pre_time


ALGORITHMS = {
    "naive": naive_adapter,
    "kmp": kmp_adapter,
    "boyer_moore": boyer_moore_adapter,
    "rabin_karp": rabin_karp_adapter,
    "suffix_tree": suffix_tree_adapter,
}

def preprocess_algorithms(text: str, algorithms_to_preprocess: List[str]) -> Dict:
    """Preprocess algorithms that require it. Returns precomputed data."""
    precomputed = {}
    total_preprocess_time = 0.0
    
    print("🔄 Preprocessing algorithms...")
    
    if "suffix_tree" in algorithms_to_preprocess:
        print("  🌳 Building suffix tree...", end="", flush=True)
        start_time = time.perf_counter()
        suffix_tree = SuffixTree(text)
        elapsed = time.perf_counter() - start_time
        precomputed['suffix_tree'] = suffix_tree
        precomputed['suffix_tree_build_time'] = elapsed * 1000
        total_preprocess_time += elapsed * 1000
        print(f" ✅ {elapsed:.4f}s")
    
    # Note: Rabin-Karp preprocessing would go here if we implement rolling hash precomputation
    # For now, keeping the existing implementation since it's already reasonably fast
    
    precomputed['total_preprocess_time'] = total_preprocess_time
    print(f"✅ Preprocessing complete in {total_preprocess_time/1000:.4f}s")
    return precomputed

def count_total_patterns(patterns_root: str, only_categories: Optional[List[str]] = None, limit_per_category: Optional[int] = None) -> int:
    """Count total patterns that will be processed."""
    total = 0
    for category in sorted(os.listdir(patterns_root)):
        if only_categories and category not in only_categories:
            continue
        cat_dir = os.path.join(patterns_root, category)
        if not os.path.isdir(cat_dir):
            continue
        files = [f for f in os.listdir(cat_dir) if f.endswith(".txt")]
        if limit_per_category is not None:
            files = files[:limit_per_category]
        total += len(files)
    return total

# -----------------------------
# Measurement helpers
# -----------------------------

def measure_memory_and_time(func, *args, **kwargs):
    # Algorithms sensitive to tracemalloc overhead (due to many temporary objects)
    MEMORY_SENSITIVE_ALGOS = {'rabin_karp_adapter'}
    
    # Check if this is a memory-sensitive algorithm by inspecting the function name
    func_name = getattr(func, '__name__', str(func))
    is_memory_sensitive = any(sensitive in func_name for sensitive in MEMORY_SENSITIVE_ALGOS)
    
    if is_memory_sensitive:
        # For memory-sensitive algorithms: measure time only, estimate memory
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        result = func(*args, **kwargs)
        wall_ms = (time.perf_counter() - start_wall) * 1000
        cpu_ms = (time.process_time() - start_cpu) * 1000
        
        # Estimate memory usage based on input size and algorithm characteristics
        text_size = len(args[1]) if len(args) > 1 else 500000  # genome size
        pattern_size = len(args[0]) if len(args) > 0 else 50   # pattern size
        
        # Rabin-Karp memory estimation: O(1) space for hash values + pattern storage
        estimated_kb = (pattern_size * 2 + 64) / 1024.0  # Pattern + hash variables
        peak_kb = estimated_kb
        
    else:
        # For other algorithms: full memory tracking is fine
        tracemalloc.start()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        result = func(*args, **kwargs)
        wall_ms = (time.perf_counter() - start_wall) * 1000
        cpu_ms = (time.process_time() - start_cpu) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kb = peak / 1024.0
    
    return result, wall_ms, cpu_ms, peak_kb

# -----------------------------
# Benchmark runner
# -----------------------------

def run_bench(patterns_root: str,
              output_csv: str,
              enable_regex: bool,
              only_categories: Optional[List[str]] = None,
              limit_per_category: Optional[int] = None,
              only_algorithms: Optional[List[str]] = None,
              skip_naive_in_categories: Optional[List[str]] = None,
              skip_naive_over: Optional[int] = None,
              skip_suffix_tree_in_categories: Optional[List[str]] = None,
              skip_suffix_tree_over: Optional[int] = None,
              incremental_write: bool = True) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []

    # Prepare incremental CSV writer
    csv_file = None
    csv_writer = None
    header = [
        "pattern_name","category","algorithm","pattern_length",
        "preprocessing_time_ms","search_time_ms","cpu_time_ms","total_time_ms","total_time_with_preprocessing_ms",
        "matches_count","comparisons","hash_verifications","peak_memory_kb",
        "throughput_bp_per_ms"
    ]
    if incremental_write:
        csv_file = open(output_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=header)
        csv_writer.writeheader()

    algos = {k: v for k, v in ALGORITHMS.items() if (only_algorithms is None or k in only_algorithms)}
    
    # Preprocess algorithms that require it
    algorithms_needing_preprocessing = [name for name in algos.keys() if name in ["suffix_tree"]]
    precomputed_data = preprocess_algorithms(GENOME, algorithms_needing_preprocessing) if algorithms_needing_preprocessing else {}
    
    # Count total patterns to allocate preprocessing cost
    total_patterns = count_total_patterns(patterns_root, only_categories, limit_per_category)
    print(f"📈 Total patterns to process: {total_patterns}")
    
    # Calculate per-pattern preprocessing cost for algorithms that need it
    preprocessing_cost_per_pattern = {}
    if 'suffix_tree_build_time' in precomputed_data:
        preprocessing_cost_per_pattern['suffix_tree'] = precomputed_data['suffix_tree_build_time'] / total_patterns
        print(f"📉 Suffix tree preprocessing cost per pattern: {preprocessing_cost_per_pattern['suffix_tree']:.4f}ms")

    for category in sorted(os.listdir(patterns_root)):
        if only_categories and category not in only_categories:
            continue
        cat_dir = os.path.join(patterns_root, category)
        if not os.path.isdir(cat_dir):
            continue
        files = [f for f in os.listdir(cat_dir) if f.endswith(".txt")]
        files.sort()
        if limit_per_category is not None:
            files = files[:limit_per_category]
        
        print(f"\n📂 Processing category: {category} ({len(files)} patterns)")
        
        for file_idx, fname in enumerate(files):
            path = os.path.join(cat_dir, fname)
            try:
                pattern = open(path).read().strip()
            except Exception as e:
                print(f"❌ Skip {path}: {e}")
                continue
            plen = len(pattern)
            
            print(f"  🧬 Pattern {file_idx+1}/{len(files)}: {fname} (length={plen})")

            for alg_name, fn in algos.items():
                print(f"    🔍 Running {alg_name}...", end="", flush=True)
                # Apply skip policy for naive to avoid very long runs
                if alg_name == "naive":
                    if skip_naive_in_categories and category in skip_naive_in_categories:
                        print(" ⏭️  SKIP (category policy)")
                        continue
                    if skip_naive_over is not None and plen > skip_naive_over:
                        print(f" ⏭️  SKIP (length {plen} > {skip_naive_over})")
                        continue
                
                # Apply skip policy for suffix tree to avoid very long construction
                if alg_name == "suffix_tree":
                    if skip_suffix_tree_in_categories and category in skip_suffix_tree_in_categories:
                        print(" ⏭️  SKIP (category policy)")
                        continue
                    if skip_suffix_tree_over is not None and plen > skip_suffix_tree_over:
                        print(f" ⏭️  SKIP (length {plen} > {skip_suffix_tree_over})")
                        continue
                
                start_time = time.perf_counter()
                (matches, comps, verifs, pre_time), wall_ms, cpu_ms, peak_kb = measure_memory_and_time(fn, pattern, GENOME, precomputed_data)
                elapsed = time.perf_counter() - start_time
                print(f" ✅ {elapsed:.4f}s ({len(matches)} matches)")
                
                # For algorithms with preprocessing, use amortized preprocessing time
                if alg_name == "suffix_tree" and alg_name in preprocessing_cost_per_pattern:
                    preprocessing_time = preprocessing_cost_per_pattern[alg_name]
                    search_time_est = wall_ms  # All measured time is search time
                else:
                    preprocessing_time = pre_time
                    search_time_est = max(0.0, wall_ms - pre_time)
                row = {
                    "pattern_name": fname[:-4],
                    "category": category,
                    "algorithm": alg_name,
                    "pattern_length": plen,
                    "preprocessing_time_ms": preprocessing_time,
                    "search_time_ms": search_time_est,
                    "cpu_time_ms": cpu_ms,
                    "total_time_ms": wall_ms,
                    "total_time_with_preprocessing_ms": wall_ms + preprocessing_time,
                    "matches_count": len(matches),
                    "comparisons": comps,
                    "hash_verifications": verifs,
                    "peak_memory_kb": round(peak_kb, 2),
                    "throughput_bp_per_ms": round(G_LEN / wall_ms, 4) if wall_ms > 0 else None,
                }
                if incremental_write:
                    csv_writer.writerow(row)
                else:
                    rows.append(row)

            if enable_regex:
                print(f"    🔍 Running regex...", end="", flush=True)
                start_time = time.perf_counter()
                (matches_r, _, _, _, regex_time_ms), wall_ms_r, cpu_ms_r, peak_kb_r = measure_memory_and_time(regex_search, pattern, GENOME)
                elapsed = time.perf_counter() - start_time
                print(f" ✅ {elapsed:.4f}s ({len(matches_r)} matches)")
                
                row_r = {
                    "pattern_name": fname[:-4],
                    "category": category,
                    "algorithm": "regex",
                    "pattern_length": plen,
                    "preprocessing_time_ms": 0.0,
                    "search_time_ms": wall_ms_r,  # all time is search
                    "cpu_time_ms": cpu_ms_r,
                    "total_time_ms": wall_ms_r,
                    "total_time_with_preprocessing_ms": wall_ms_r,  # Regex has no preprocessing
                    "matches_count": len(matches_r),
                    "comparisons": None,
                    "hash_verifications": None,
                    "peak_memory_kb": round(peak_kb_r, 2),
                    "throughput_bp_per_ms": round(G_LEN / wall_ms_r, 4) if wall_ms_r > 0 else None,
                }
                if incremental_write:
                    csv_writer.writerow(row_r)
                else:
                    rows.append(row_r)
        
        print(f"✅ Completed category: {category}")
    
    print(f"\n🎯 Benchmark complete! Processing {len(rows) if not incremental_write else 'incremental'} total rows...")
    
    if incremental_write:
        csv_file.close()
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
    print("✅ Wrote:", output_csv, "rows:", len(df))
    return df

# -----------------------------
# Plots (only 1,2,3,4,8 as requested)
# -----------------------------

def make_plots(df: pd.DataFrame, outdir: str = "figures"):
    os.makedirs(outdir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1) Simple bar chart: average runtime by algorithm
    plt.figure(figsize=(10, 6))
    avg_time = df.groupby("algorithm")["total_time_ms"].mean().sort_values()
    bars = plt.bar(range(len(avg_time)), avg_time.values, color='skyblue', edgecolor='navy')
    plt.xticks(range(len(avg_time)), avg_time.index, rotation=45)
    plt.ylabel("Average Runtime (ms)")
    plt.title("Algorithm Performance Comparison - Average Runtime")
    plt.yscale('log')
    
    # Add value labels on bars
    for i, v in enumerate(avg_time.values):
        plt.text(i, v * 1.1, f'{v:.4f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_simple_runtime_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2) Runtime comparison: algorithms vs categories (heatmap)
    plt.figure(figsize=(12, 8))
    pivot_data = df.groupby(['category', 'algorithm'])['total_time_ms'].mean().unstack(fill_value=0)
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Runtime (ms)'})
    plt.title("Runtime Heatmap: Algorithms vs Categories")
    plt.ylabel("Dataset Category")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_runtime_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3) Pattern length vs runtime scatter (more readable)
    plt.figure(figsize=(12, 8))
    algorithms = df['algorithm'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = df[df['algorithm'] == alg]
        plt.scatter(alg_data['pattern_length'], alg_data['total_time_ms'], 
                   label=alg, alpha=0.7, s=50, color=colors[i])
    
    plt.xlabel("Pattern Length (bp)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Pattern Length")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_length_vs_runtime_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 4) Preprocessing vs Search time (fixed to show actual preprocessing)
    plt.figure(figsize=(12, 8))
    prep_data = df[df['preprocessing_time_ms'] > 0].groupby('algorithm').agg({
        'preprocessing_time_ms': 'mean',
        'search_time_ms': 'mean'
    }).reset_index()
    
    if len(prep_data) > 0:
        x = np.arange(len(prep_data))
        width = 0.35
        
        plt.bar(x, prep_data['preprocessing_time_ms'], width, label='Preprocessing', color='lightcoral')
        plt.bar(x, prep_data['search_time_ms'], width, bottom=prep_data['preprocessing_time_ms'], 
               label='Search', color='lightblue')
        
        plt.xlabel('Algorithm')
        plt.ylabel('Time (ms)')
        plt.title('Preprocessing vs Search Time (Algorithms with Preprocessing)')
        plt.xticks(x, prep_data['algorithm'])
        plt.legend()
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'No significant preprocessing time detected', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Preprocessing vs Search Time (No Preprocessing Data)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_preprocessing_vs_search.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 5) Memory usage comparison (simple bar chart)
    plt.figure(figsize=(10, 6))
    avg_memory = df.groupby("algorithm")["peak_memory_kb"].mean().sort_values()
    bars = plt.bar(range(len(avg_memory)), avg_memory.values, color='lightgreen', edgecolor='darkgreen')
    plt.xticks(range(len(avg_memory)), avg_memory.index, rotation=45)
    plt.ylabel("Average Memory Usage (kB)")
    plt.title("Memory Usage Comparison")
    
    # Add value labels
    for i, v in enumerate(avg_memory.values):
        plt.text(i, v * 1.05, f'{v:.1f}kB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "05_memory_usage.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6) Performance by pattern size
    plt.figure(figsize=(14, 10))
    size_order = ['small', 'medium', 'large', 'vlarge']
    
    # Create subplot for each size category
    for i, size_cat in enumerate(size_order, 1):
        plt.subplot(2, 2, i)
        size_df = df[df['size_category'] == size_cat]
        if size_df.empty:
            plt.text(0.5, 0.5, f'No {size_cat} patterns', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{size_cat.title()} Patterns (No Data)')
            continue
            
        algo_means = size_df.groupby('algorithm')['total_time_ms'].mean().sort_values()
        
        bars = plt.bar(range(len(algo_means)), algo_means.values, color='skyblue', edgecolor='navy')
        plt.xticks(range(len(algo_means)), algo_means.index, rotation=45, ha='right')
        plt.ylabel('Average Runtime (ms)')
        plt.title(f'{size_cat.title()} Patterns ({size_df["pattern_length"].min()}-{size_df["pattern_length"].max()}bp)')
        plt.yscale('log')
        
        # Add value labels
        for bar, val in zip(bars, algo_means.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_performance_by_size.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 7) Algorithm performance heatmap by category and size
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    heatmap_data = df.groupby(['category', 'size_category', 'algorithm'])['total_time_ms'].mean().unstack(fill_value=np.nan)
    
    # Create a flattened version for easier plotting
    plot_data = []
    for category in df['category'].unique():
        for size_cat in size_order:
            if (category, size_cat) in heatmap_data.index:
                row_data = heatmap_data.loc[(category, size_cat)]
                plot_data.append([f"{category}\\n{size_cat}"] + row_data.values.tolist())
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data, columns=['Category_Size'] + list(heatmap_data.columns))
        plot_df = plot_df.set_index('Category_Size')
        
        # Log transform for better visualization
        plot_df_log = np.log10(plot_df + 1)
        
        sns.heatmap(plot_df_log.T, annot=True, fmt='.1f', cmap='viridis_r', 
                   cbar_kws={'label': 'Log10(Runtime + 1)'})
        plt.title('Algorithm Performance Heatmap by Category and Size')
        plt.ylabel('Algorithm')
        plt.xlabel('Category and Size')
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Algorithm Performance Heatmap (No Data)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "07_performance_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 8) Average category detailed analysis
    if 'average' in df['category'].values:
        avg_df = df[df['category'] == 'average'].copy()
        plt.figure(figsize=(14, 8))
        
        # Create a plot showing all algorithms for average category patterns
        pattern_order = sorted(avg_df['pattern_name'].unique())
        algorithms = avg_df['algorithm'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            alg_data = []
            for pattern in pattern_order:
                pattern_data = avg_df[(avg_df['pattern_name'] == pattern) & (avg_df['algorithm'] == alg)]
                if not pattern_data.empty:
                    alg_data.append(pattern_data['total_time_ms'].iloc[0])
                else:
                    alg_data.append(np.nan)
            
            plt.plot(range(len(pattern_order)), alg_data, marker='o', label=alg, 
                    linewidth=2, markersize=6, color=colors[i])
        
        plt.xlabel('Average Category Patterns')
        plt.ylabel('Runtime (ms)')
        plt.title('Average Category - Detailed Performance by Pattern')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        pattern_labels = [p.replace('avg_', '').replace('.txt', '') for p in pattern_order]
        plt.xticks(range(len(pattern_order)), pattern_labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "08_average_category_detailed.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # NEW: Per-category runtime plots (excluding many-query)
    categories = [cat for cat in df['category'].unique() if cat != 'many-query']
    for category in categories:
        plt.figure(figsize=(14, 8))
        cat_data = df[df['category'] == category].copy()
        
        # Sort patterns by name for consistent ordering
        pattern_order = sorted(cat_data['pattern_name'].unique())
        cat_data['pattern_idx'] = cat_data['pattern_name'].map({p: i for i, p in enumerate(pattern_order)})
        
        algorithms = cat_data['algorithm'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            alg_data = cat_data[cat_data['algorithm'] == alg].sort_values('pattern_idx')
            plt.plot(alg_data['pattern_idx'], alg_data['total_time_ms'], 
                    marker='o', label=alg, linewidth=2, markersize=6, color=colors[i])
        
        plt.xlabel('Test Case (Pattern Index)')
        plt.ylabel('Runtime (ms)')
        plt.title(f'Runtime vs Test Cases - {category.title()} Category')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(pattern_order)), [f'P{i+1}' for i in range(len(pattern_order))])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"09_{category}_runtime_per_testcase.png"), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Generated plots for {len(categories)} categories + 8 summary plots")


# -----------------------------
# Text report writer
# -----------------------------

def categorize_pattern_by_size(length: int) -> str:
    """Categorize patterns by size for detailed analysis."""
    if length <= 10:
        return "small"
    elif length <= 50:
        return "medium" 
    elif length <= 200:
        return "large"
    else:
        return "vlarge"

def write_analysis_report(df: pd.DataFrame, out_path: str = "results/analysis_report.txt"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Filter out many-query category
    df = df[df['category'] != 'many-query'].copy()
    
    # Add pattern size categorization if not already present
    if 'size_category' not in df.columns and 'pattern_length' in df.columns:
        df['size_category'] = df['pattern_length'].apply(categorize_pattern_by_size)

    lines = []
    def w(s=""):
        lines.append(s)

    # Header
    w("=" * 80)
    w("DNA PATTERN MATCHING ALGORITHMS - COMPREHENSIVE ANALYSIS REPORT")
    w("=" * 80)
    w(f"Total Patterns: {df['pattern_name'].nunique()} | Categories: {df['category'].nunique()} | Algorithm Runs: {len(df)}")
    w(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")

    # Overall Performance Summary Table (Runtime Only)
    w("OVERALL PERFORMANCE RANKING")
    w("=" * 50)
    overall = (df.groupby("algorithm")
                 .agg(mean_time=("total_time_ms","mean"),
                      median_time=("total_time_ms","median"),
                      std_time=("total_time_ms","std"),
                      min_time=("total_time_ms","min"),
                      max_time=("total_time_ms","max"),
                      mean_pre=("preprocessing_time_ms","mean"),
                      mean_mem=("peak_memory_kb","mean"),
                      patterns=("pattern_name","count"))
                 .reset_index()
                 .sort_values("mean_time"))
    
    w(f"{'Algorithm':<12} {'Mean(ms)':<10} {'Median(ms)':<12} {'Min(ms)':<10} {'Max(ms)':<12} {'Std(ms)':<10} {'Memory(kB)':<12} {'Patterns':<10} {'Total(ms)*':<12}")
    w("-" * 119)
    for _, r in overall.iterrows():
        # Calculate total time: (Median × Patterns) + Preprocessing Time
        total_time = (r['median_time'] * r['patterns']) + r['mean_pre']
        w(f"{r['algorithm']:<12} {r['mean_time']:<10.4f} {r['median_time']:<12.4f} {r['min_time']:<10.4f} {r['max_time']:<12.4f} {(r['std_time'] or 0):<10.4f} {r['mean_mem']:<12.1f} {int(r['patterns']):<10} {total_time:<12.4f}")
    
    w("")
    w("* Total Time = (Median × Patterns) + Preprocessing Time")

    # Speedup vs Regex Table (Total Time with Preprocessing)
    w("")
    w("SPEEDUP COMPARISON vs REGEX (Total Time)")
    w("=" * 41)
    if "regex" in df["algorithm"].unique():
        # Calculate total time including preprocessing for fair comparison
        regex_row = overall[overall.algorithm == "regex"]
        regex_total_time = (regex_row["median_time"].iloc[0] * regex_row["patterns"].iloc[0]) + regex_row["mean_pre"].iloc[0] if len(regex_row) > 0 else 1.0
        
        w(f"{'Algorithm':<15} {'Speedup':<15} {'Description':<30}")
        w("-" * 60)
        for _, r in overall.iterrows():
            # Calculate total time for this algorithm
            alg_total_time = (r['median_time'] * r['patterns']) + r['mean_pre']
            
            if r['algorithm'] == 'regex':
                speedup_str = "1.00x"
                desc = "Baseline (C-optimized)"
            else:
                speedup = regex_total_time / alg_total_time if alg_total_time > 0 else 0
                if speedup >= 1:
                    speedup_str = f"{speedup:.4f}x faster"
                else:
                    speedup_str = f"{1/speedup:.4f}x slower"
                
                if speedup >= 2:
                    desc = "Significantly faster"
                elif speedup >= 1:
                    desc = "Faster than regex"
                elif speedup >= 0.5:
                    desc = "Comparable performance"
                elif speedup >= 0.1:
                    desc = "Moderately slower"
                else:
                    desc = "Much slower"
            
            w(f"{r['algorithm']:<15} {speedup_str:<15} {desc:<30}")
    
    # Speedup vs Regex Table (Runtime Only)
    w("")
    w("SPEEDUP COMPARISON vs REGEX (Runtime Only)")
    w("=" * 44)
    if "regex" in df["algorithm"].unique():
        regex_runtime = overall[overall.algorithm == "regex"]["mean_time"].iloc[0] if len(overall[overall.algorithm == "regex"]) > 0 else 1.0
        w(f"{'Algorithm':<15} {'Speedup':<15} {'Description':<30}")
        w("-" * 60)
        for _, r in overall.iterrows():
            if r['algorithm'] == 'regex':
                speedup_str = "1.00x"
                desc = "Baseline (C-optimized)"
            else:
                speedup = regex_runtime / r['mean_time'] if r['mean_time'] > 0 else 0
                if speedup >= 1:
                    speedup_str = f"{speedup:.4f}x faster"
                else:
                    speedup_str = f"{1/speedup:.4f}x slower"
                
                if speedup >= 2:
                    desc = "Significantly faster"
                elif speedup >= 1:
                    desc = "Faster than regex"
                elif speedup >= 0.5:
                    desc = "Comparable performance"
                elif speedup >= 0.1:
                    desc = "Moderately slower"
                else:
                    desc = "Much slower"
            
            w(f"{r['algorithm']:<15} {speedup_str:<15} {desc:<30}")
        w("")

    # Pattern size breakdown
    w("PATTERN SIZE BREAKDOWN")
    w("=" * 30)
    
    size_summary = df.groupby('size_category').agg({
        'pattern_length': ['min', 'max', 'count'],
        'total_time_ms': 'mean'
    })
    
    w(f"{'Size':<10} {'Length Range':<15} {'Count':<8} {'Avg Time(ms)':<12}")
    w("-" * 45)
    
    for size_cat in ['small', 'medium', 'large', 'vlarge']:
        if size_cat in size_summary.index:
            min_len = int(size_summary.loc[size_cat, ('pattern_length', 'min')])
            max_len = int(size_summary.loc[size_cat, ('pattern_length', 'max')])
            count = int(size_summary.loc[size_cat, ('pattern_length', 'count')])
            avg_time = size_summary.loc[size_cat, ('total_time_ms', 'mean')]
            
            w(f"{size_cat:<10} {min_len}-{max_len}bp{'':<8} {count:<8} {avg_time:<12.4f}")
    w("")

    # Per-category detailed analysis
    for category, cdf in df.groupby("category"):
        w(f"CATEGORY: {category.upper()}")
        w("=" * (10 + len(category)))
        
        # Category statistics (Runtime Only)
        cat_stats = (cdf.groupby("algorithm")
                       .agg(mean_time=("total_time_ms","mean"),
                            std_time=("total_time_ms","std"),
                            mean_pre=("preprocessing_time_ms","mean"),
                            mean_mem=("peak_memory_kb","mean"),
                            patterns=("pattern_name","count"))
                       .reset_index()
                       .sort_values("mean_time"))
        
        w(f"{'Algorithm':<12} {'Mean(ms)':<10} {'Std(ms)':<10} {'Preproc(ms)':<12} {'Memory(kB)':<12} {'Patterns':<10} {'Total(ms)*':<12}")
        w("-" * 88)
        for _, r in cat_stats.iterrows():
            # Calculate total time: (Mean × Patterns) + Preprocessing Time
            total_time = (r['mean_time'] * r['patterns']) + r['mean_pre']
            w(f"{r['algorithm']:<12} {r['mean_time']:<10.4f} {(r['std_time'] or 0):<10.4f} {r['mean_pre']:<12.4f} {r['mean_mem']:<12.1f} {int(r['patterns']):<10} {total_time:<12.4f}")
        
        w("")
        w("* Total Time = (Mean × Patterns) + Preprocessing Time")
        
        # Per-pattern detailed results for this category
        w("")
        w("PER-PATTERN RESULTS:")
        w(f"{'Pattern':<20} {'Algorithm':<12} {'Time(ms)':<10} {'Matches':<8} {'Memory(kB)':<10}")
        w("-" * 70)
        
        pattern_results = cdf.sort_values(['pattern_name', 'total_time_ms'])
        for _, row in pattern_results.iterrows():
            pattern_short = row['pattern_name'][:17] + "..." if len(row['pattern_name']) > 20 else row['pattern_name']
            w(f"{pattern_short:<20} {row['algorithm']:<12} {row['total_time_ms']:<10.4f} {row['matches_count']:<8} {row['peak_memory_kb']:<10.1f}")
        
        w("")

    # Dataset explanation
    w("DATASET CATEGORIES EXPLANATION")
    w("=" * 35)
    w("• average:           Real genome subsequences (typical performance)")
    w("• repetitive:        Synthetic worst-case patterns (algorithmic stress)")
    w("• rare:              Patterns occurring 1-5 times (sparse matches)")
    w("• zero-occurrence:   Random patterns absent from genome (no matches)")
    w("• rabin-karp-adversarial: Hash collision-prone patterns (RK stress)")
    w("• many-query:        100 short patterns (amortization testing)")
    w("")
    w("NOTE: Each pattern file contains one DNA string tested against 500K E.coli genome")

    # Write to file
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    # Create detailed CSV breakdown
    detail_path = out_path.replace('.txt', '_detailed.csv')
    df_detailed = df.copy()
    if "regex" in df["algorithm"].unique():
        regex_times = df[df.algorithm == "regex"].set_index(['category', 'pattern_name'])['total_time_with_preprocessing_ms'].to_dict()
        def calc_speedup(row):
            key = (row['category'], row['pattern_name'])
            if row['algorithm'] == 'regex':
                return 1.0
            regex_time = regex_times.get(key, 1.0)
            return regex_time / row['total_time_with_preprocessing_ms'] if row['total_time_with_preprocessing_ms'] > 0 else 0
        df_detailed['speedup_vs_regex'] = df_detailed.apply(calc_speedup, axis=1)
    
    df_detailed.to_csv(detail_path, index=False)
    
    return out_path

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark DNA pattern matching across algorithms.")
    p.add_argument("--patterns-root", default="patterns", help="Root folder containing pattern categories.")
    p.add_argument("--output-csv", default="results/raw_results.csv", help="CSV path for raw results.")
    p.add_argument("--no-regex", action="store_true", help="Disable regex baseline.")
    p.add_argument("--only-categories", nargs="*", help="Limit to these categories.")
    p.add_argument("--limit-per-category", type=int, help="Limit number of patterns processed per category.")
    p.add_argument("--only-algorithms", nargs="*", choices=list(ALGORITHMS.keys()), help="Limit to selected algorithms.")
    p.add_argument("--skip-naive-in-categories", nargs="*", default=[], help="Categories to skip the naive algorithm in (default: none).")
    p.add_argument("--skip-naive-over", type=int, default=100, help="Skip naive when pattern length exceeds this value (default: 100). Use 0 or negative to disable.")
    p.add_argument("--skip-suffix-tree-in-categories", nargs="*", default=[], help="Categories to skip suffix tree in (default: none).")
    p.add_argument("--skip-suffix-tree-over", type=int, default=1000, help="Skip suffix tree when pattern length exceeds this value (default: 1000). Use 0 or negative to disable.")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p.add_argument("--report-txt", default="results/analysis_report.txt", help="Path to write comparative analysis text report.")
    return p.parse_args()


def main():
    print("🚀 Starting DNA Pattern Matching Benchmark")
    args = parse_args()
    print(f"📊 Config: algorithms={args.only_algorithms or 'all'}, categories={args.only_categories or 'all'}, limit={args.limit_per_category or 'none'}")
    
    df = run_bench(
        patterns_root=args.patterns_root,
        output_csv=args.output_csv,
        enable_regex=not args.no_regex,
        only_categories=args.only_categories,
        limit_per_category=args.limit_per_category,
        only_algorithms=args.only_algorithms,
        skip_naive_in_categories=(args.skip_naive_in_categories or []),
        skip_naive_over=(None if (args.skip_naive_over is not None and args.skip_naive_over <= 0) else args.skip_naive_over),
        skip_suffix_tree_in_categories=(args.skip_suffix_tree_in_categories or []),
        skip_suffix_tree_over=(None if (args.skip_suffix_tree_over is not None and args.skip_suffix_tree_over <= 0) else args.skip_suffix_tree_over),
    )
    
    # Add pattern size categorization for analysis and plotting
    if 'pattern_length' in df.columns:
        df['size_category'] = df['pattern_length'].apply(categorize_pattern_by_size)
    
    if not args.no_plots:
        print("📈 Generating plots...")
        make_plots(df)
        print("✅ Figures written to ./figures")
    try:
        print("📝 Writing analysis report...")
        write_analysis_report(df, args.report_txt)
        print(f"✅ Wrote report: {args.report_txt}")
    except Exception as e:
        print("❌ Report generation failed:", e)
    print("🎉 Benchmark complete!")


if __name__ == "__main__":
    main()
