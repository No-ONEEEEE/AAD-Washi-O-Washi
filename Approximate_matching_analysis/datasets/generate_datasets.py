"""Generate simple sample datasets for benchmarking.

Creates text files with repeated English words and controlled noise so you can
evaluate fuzzy matching algorithms.
"""
"""Generate datasets including DNA (A/T/G/C) sequences of various types.

This script produces FASTA files containing synthetic DNA sequences suitable
for benchmarking approximate string matching algorithms. It supports several
sequence types to simulate different biological / synthetic composition:

- random: uniform random A/T/G/C
- high_gc: biased toward G/C
- repetitive: short motif repeated
- low_complexity: long homopolymer runs
- mutated: generate a base sequence then introduce mutations

Files are written in FASTA format to the output directory.
"""

import argparse
import random
from pathlib import Path
from typing import List


NUCLEOTIDES = ["A", "T", "G", "C"]


def _random_sequence(length: int, gc_bias: float = 0.5) -> str:
    # gc_bias is fraction probability for G/C versus A/T
    seq = []
    for _ in range(length):
        if random.random() < gc_bias:
            seq.append(random.choice(["G", "C"]))
        else:
            seq.append(random.choice(["A", "T"]))
    return "".join(seq)


def _repetitive_sequence(length: int, motif_len: int = 3) -> str:
    motif = "".join(random.choices(NUCLEOTIDES, k=motif_len))
    return (motif * ((length // len(motif)) + 1))[:length]


def _low_complexity_sequence(length: int) -> str:
    # produce long runs of a single nucleotide with occasional switches
    seq = []
    i = 0
    while i < length:
        remaining = length - i
        if remaining <= 0:
            break
        max_run = min(200, remaining)
        # pick a run length between 1 and max_run (prefer longer runs)
        run_len = min(remaining, random.randint(1, max_run))
        base = random.choice(NUCLEOTIDES)
        seq.append(base * run_len)
        i += run_len
    return "".join(seq)[:length]


def _mutated_sequence(base_seq: str, mutation_rate: float) -> str:
    out = []
    for ch in base_seq:
        if random.random() < mutation_rate:
            # substitute with a different nucleotide (no indels here for simplicity)
            choices = [n for n in NUCLEOTIDES if n != ch]
            out.append(random.choice(choices))
        else:
            out.append(ch)
    return "".join(out)


def _write_fasta(path: Path, header: str, seq: str, width: int = 80):
    lines = [f">{header}\n"]
    for i in range(0, len(seq), width):
        lines.append(seq[i : i + width] + "\n")
    path.write_text("".join(lines))


def generate_dna_dataset(
    out_dir: Path,
    types: List[str],
    lengths: List[int],
    count: int = 3,
    gc_bias: float = 0.5,
    mutation_rate: float = 0.01,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in types:
        for length in lengths:
            for i in range(count):
                if t == "random":
                    seq = _random_sequence(length, gc_bias=gc_bias)
                elif t == "high_gc":
                    seq = _random_sequence(length, gc_bias=0.8)
                elif t == "repetitive":
                    motif_len = random.randint(2, 6)
                    seq = _repetitive_sequence(length, motif_len=motif_len)
                elif t == "low_complexity":
                    seq = _low_complexity_sequence(length)
                elif t == "mutated":
                    base = _random_sequence(length, gc_bias=gc_bias)
                    seq = _mutated_sequence(base, mutation_rate)
                else:
                    raise ValueError(f"Unknown type: {t}")

                fname = f"dna_{t}_len{length}_{i}.fasta"
                header = f"dna|{t}|len{length}|i{i}"
                _write_fasta(out_dir / fname, header, seq)


def parse_lengths(s: List[str]) -> List[int]:
    # accept integers or comma-separated ranges like 100-1000:100
    out = []
    for token in s:
        if "-" in token:
            parts = token.split(":")
            rng = parts[0]
            step = int(parts[1]) if len(parts) > 1 else None
            a, b = rng.split("-")
            a = int(a)
            b = int(b)
            if step is None:
                out.extend([a, b])
            else:
                out.extend(list(range(a, b + 1, step)))
        else:
            out.append(int(token))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequence datasets (FASTA)")
    parser.add_argument("--out", "-o", default="datasets/dna_samples")
    parser.add_argument("--types", "-t", nargs="+", default=["random", "repetitive", "low_complexity", "high_gc", "mutated"],
                        help="Sequence types to generate")
    parser.add_argument("--lengths", "-l", nargs="+", default=[100, 1000, 10000],
                        help="Lengths to generate. Accept integers or ranges like 100-1000:100")
    parser.add_argument("--count", "-c", type=int, default=3, help="Number of sequences per (type,length)")
    parser.add_argument("--gc-bias", type=float, default=0.5, help="GC bias for random sequences (0-1)")
    parser.add_argument("--mutation-rate", type=float, default=0.01, help="Mutation rate for 'mutated' type")
    args = parser.parse_args()
    out = Path(args.out)
    lengths = parse_lengths([str(x) for x in args.lengths])
    generate_dna_dataset(out, args.types, lengths, count=args.count, gc_bias=args.gc_bias, mutation_rate=args.mutation_rate)
    print(f"Wrote DNA datasets to {out.resolve()}")


if __name__ == "__main__":
    main()
