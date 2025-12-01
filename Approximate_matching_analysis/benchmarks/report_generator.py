"""Generate a summary report from benchmark CSV results.

Reads the CSV output from benchmark.py and produces:
- Summary statistics by algorithm
- Comparison plots (time, memory vs sequence length, edits)
- Markdown report with findings

Renamed from report.py to report_generator.py for clarity.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def summary_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time/memory stats by algorithm."""
    algos = ["re", "shiftor", "levenshtein"]
    summaries = []
    for algo in algos:
        time_col = f"{algo}_time"
        mem_col = f"{algo}_mem"
        row = {
            "Algorithm": algo.upper(),
            "Mean Time (s)": df[time_col].mean(),
            "Median Time (s)": df[time_col].median(),
            "Std Time (s)": df[time_col].std(),
            "Mean Mem (MB)": df[mem_col].mean() / 1e6,
            "Median Mem (MB)": df[mem_col].median() / 1e6,
        }
        summaries.append(row)
    return pd.DataFrame(summaries)


def plot_time_vs_seq_len(df: pd.DataFrame, out_dir: Path):
    """Plot average time by sequence length."""
    for edits in sorted(df["edits"].unique()):
        sub = df[df["edits"] == edits]
        grp = sub.groupby(["seq_len"]).agg(
            re_time=("re_time", "mean"),
            shiftor_time=("shiftor_time", "mean"),
            levenshtein_time=("levenshtein_time", "mean"),
        ).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        seq_lens = grp["seq_len"]
        ax.plot(seq_lens, grp["re_time"], marker='o', label='Regex (exact)', linewidth=2)
        ax.plot(seq_lens, grp["shiftor_time"], marker='s', label='Shift-Or (bit-parallel)', linewidth=2)
        ax.plot(seq_lens, grp["levenshtein_time"], marker='^', label='Levenshtein (banded)', linewidth=2)
        ax.set_xlabel("Sequence Length (bp)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Average Time vs Sequence Length (edits={edits})")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"time_vs_len_edits_{edits}.png", dpi=150)
        plt.close(fig)


def plot_time_vs_query_len(df: pd.DataFrame, out_dir: Path):
    """Plot average time by query length."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, marker, label in [("re", 'o', "Regex"), ("shiftor", 's', "Shift-Or"), ("levenshtein", '^', "Levenshtein")]:
        grp = df.groupby(["query_len"]).agg({f"{algo}_time": "mean"}).reset_index()
        ax.plot(grp["query_len"], grp[f"{algo}_time"], marker=marker, label=label, linewidth=2)
    ax.set_xlabel("Query Length (bp)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Average Time vs Query Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_query_len.png", dpi=150)
    plt.close(fig)


def plot_time_by_edits(df: pd.DataFrame, out_dir: Path):
    """Plot average time grouped by edits allowed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, marker, label in [("re", 'o', "Regex"), ("shiftor", 's', "Shift-Or"), ("levenshtein", '^', "Levenshtein")]:
        grp = df.groupby(["edits"]).agg({f"{algo}_time": "mean"}).reset_index()
        ax.plot(grp["edits"], grp[f"{algo}_time"], marker=marker, label=label, linewidth=2)
    ax.set_xlabel("Edit Distance Threshold")
    ax.set_ylabel("Time (s)")
    ax.set_title("Average Time vs Edit Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_edits.png", dpi=150)
    plt.close(fig)


def plot_mem_comparison(df: pd.DataFrame, out_dir: Path):
    """Box plot of memory usage by algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df["re_mem"] / 1e6, df["shiftor_mem"] / 1e6, df["levenshtein_mem"] / 1e6]
    bp = ax.boxplot(data, tick_labels=["Regex", "Shift-Or", "Levenshtein"], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage Distribution by Algorithm")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_boxplot.png", dpi=150)
    plt.close(fig)


def write_markdown_report(df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path):
    """Write a markdown report."""
    with out_path.open("w") as f:
        f.write("# Fuzzy Matching Benchmark Report\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Findings\n\n")
        
        # Overall fastest
        mean_times = {
            "Regex": df["re_time"].mean(),
            "Shift-Or": df["shiftor_time"].mean(),
            "Levenshtein": df["levenshtein_time"].mean(),
        }
        fastest = min(mean_times, key=mean_times.get)
        f.write(f"- **Fastest overall**: {fastest} ({mean_times[fastest]:.6f} s avg)\n")
        
        # Memory usage
        mean_mems = {
            "Regex": df["re_mem"].mean() / 1e6,
            "Shift-Or": df["shiftor_mem"].mean() / 1e6,
            "Levenshtein": df["levenshtein_mem"].mean() / 1e6,
        }
        least_mem = min(mean_mems, key=mean_mems.get)
        f.write(f"- **Lowest memory**: {least_mem} ({mean_mems[least_mem]:.2f} MB avg)\n")
        
        # Performance by sequence length
        f.write("\n### Performance by Sequence Length\n")
        for seq_len in sorted(df["seq_len"].unique()):
            sub = df[df["seq_len"] == seq_len]
            f.write(f"\n**Length {seq_len} bp**:\n")
            for algo, name in [("re", "Regex"), ("shiftor", "Shift-Or"), ("levenshtein", "Levenshtein")]:
                t = sub[f"{algo}_time"].mean()
                f.write(f"  - {name}: {t:.6f} s\n")
        
        f.write("\n### Performance by Edit Distance\n")
        for edits in sorted(df["edits"].unique()):
            sub = df[df["edits"] == edits]
            f.write(f"\n**Edits {edits}**:\n")
            for algo, name in [("re", "Regex"), ("shiftor", "Shift-Or"), ("levenshtein", "Levenshtein")]:
                t = sub[f"{algo}_time"].mean()
                f.write(f"  - {name}: {t:.6f} s\n")
        
        f.write("\n## Data Statistics\n\n")
        f.write(f"- Total measurements: {len(df)}\n")
        f.write(f"- Sequence lengths: {sorted(df['seq_len'].unique())}\n")
        f.write(f"- Query lengths: {sorted(df['query_len'].unique())}\n")
        f.write(f"- Edit distances tested: {sorted(df['edits'].unique())}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results", required=True, help="CSV file with benchmark results")
    parser.add_argument("--out-dir", default="bench_report", help="Output directory for plots and markdown")
    args = parser.parse_args()
    
    df = load_results(args.results)
    summary = summary_by_algorithm(df)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    plot_time_vs_seq_len(df, out)
    plot_time_vs_query_len(df, out)
    plot_time_by_edits(df, out)
    plot_mem_comparison(df, out)
    
    print("Writing markdown report...")
    write_markdown_report(df, summary, out / "REPORT.md")
    
    print(f"Report written to {out}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
