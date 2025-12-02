import random
import csv
import os

# =============================================================
# 1. Load genome
# =============================================================
def load_genome(path="sequences/ecoli_500k.txt"):
    genome = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                genome.append(line.strip().upper())
    return "".join(genome)

GENOME = load_genome()
N = len(GENOME)
print("Genome loaded. Length:", N)

# Utility functions
DNA = ['A', 'C', 'G', 'T']

def random_dna(length):
    return "".join(random.choice(DNA) for _ in range(length))

def extract_random_subsequence(length):
    start = random.randint(0, N - length)
    return GENOME[start:start+length]

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create main folder
BASE = "patterns"
ensure_folder(BASE)

# =============================================================
# Collector for all patterns
# =============================================================
dataset = []

def add_pattern(name, pattern, category):
    """Save pattern + create category folders automatically"""
    folder = os.path.join(BASE, category)
    ensure_folder(folder)

    dataset.append({
        "name": name,
        "pattern": pattern,
        "length": len(pattern),
        "category": category
    })

    with open(f"{folder}/{name}.txt", "w") as f:
        f.write(pattern)


# =============================================================
# CATEGORY 1 — AVERAGE CASE (55 patterns)
# =============================================================
print("Generating Category 1…")

category = "average"

# Short 5–10 bp → 20 patterns
for i in range(20):
    L = random.randint(5, 10)
    add_pattern(f"avg_short_{i}", extract_random_subsequence(L), category)

# Medium 30–60 bp → 20 patterns
for i in range(20):
    L = random.randint(30, 60)
    add_pattern(f"avg_medium_{i}", extract_random_subsequence(L), category)

# Long 150–300 bp → 10 patterns
for i in range(10):
    L = random.randint(150, 300)
    add_pattern(f"avg_long_{i}", extract_random_subsequence(L), category)

# Very long 1000–3000 bp → 5 patterns
for i in range(5):
    L = random.randint(1000, 3000)
    add_pattern(f"avg_vlong_{i}", extract_random_subsequence(L), category)


# =============================================================
# CATEGORY 2 — WORST-CASE REPETITIVE (9 patterns)
# =============================================================
print("Generating Category 2…")

category = "repetitive"

# Pure A, 3 lengths
for L in [20, 50, 100]:
    add_pattern(f"rep_allA_{L}", "A" * L, category)

# Alternating ACAC..., 3 lengths
for L in [20, 50, 100]:
    motif = "AC"
    pattern = (motif * (L // 2))[:L]
    add_pattern(f"rep_AC_{L}", pattern, category)

# Repeating ATAT..., 3 lengths
for L in [20, 50, 100]:
    motif = "AT"
    pattern = (motif * (L // 2))[:L]
    add_pattern(f"rep_AT_{L}", pattern, category)


# =============================================================
# CATEGORY 3 — RARE OCCURRENCE (10 patterns)
# =============================================================
print("Generating Category 3…")

category = "rare"

def count_occurrences(pattern, text):
    """Count non-overlapping occurrences of pattern in text"""
    count = 0
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1  # Allow overlapping for accurate count
    return count

def generate_rare_pattern(min_length=20, max_length=80, max_occurrences=5):
    """Generate pattern that occurs 1-5 times in genome"""
    attempts = 0
    max_attempts = 1000  # Prevent infinite loop
    
    while attempts < max_attempts:
        L = random.randint(min_length, max_length)
        candidate = extract_random_subsequence(L)
        
        occurrences = count_occurrences(candidate, GENOME)
        
        if 1 <= occurrences <= max_occurrences:
            return candidate, occurrences
        
        attempts += 1
    
    # Fallback: if can't find rare pattern, generate longer one
    # (longer patterns are statistically rarer)
    L = random.randint(100, 200)
    candidate = extract_random_subsequence(L)
    return candidate, count_occurrences(candidate, GENOME)

for i in range(10):
    pattern, occ_count = generate_rare_pattern()
    # print(f"  rare_{i}: length={len(pattern)}, occurrences={occ_count}")
    add_pattern(f"rare_{i}", pattern, category)


# =============================================================
# CATEGORY 4 — ZERO OCCURRENCE (35 patterns)
# =============================================================
print("Generating Category 4…")

category = "zero-occurrence"

def generate_absent_pattern(length):
    """Generate until pattern NOT in genome"""
    while True:
        p = random_dna(length)
        if p not in GENOME:
            return p

# Short (20–50 bp) → 20
for i in range(20):
    L = random.randint(20, 50)
    add_pattern(f"zero_short_{i}", generate_absent_pattern(L), category)

# Medium (100–200 bp) → 10
for i in range(10):
    L = random.randint(100, 200)
    add_pattern(f"zero_medium_{i}", generate_absent_pattern(L), category)

# Long (300–500 bp) → 5
for i in range(5):
    L = random.randint(300, 500)
    add_pattern(f"zero_long_{i}", generate_absent_pattern(L), category)


# =============================================================
# CATEGORY 5 — RABIN–KARP HASH-ADVERSARIAL (10 patterns)
# =============================================================
print("Generating Category 5…")

category = "rabin-karp-adversarial"

base = "A" * 30
for i, c in enumerate(["A", "C", "G", "T", "A"]):
    modified = base[:-1] + c
    add_pattern(f"rk_collision_{i}", modified, category)

# almost identical patterns
for i in range(5):
    s = list("A" * 40)
    pos = random.randint(0, 39)
    s[pos] = random.choice("CGT")
    add_pattern(f"rk_similar_{i}", "".join(s), category)


# =============================================================
# CATEGORY 6 — MANY QUERIES (100 patterns)
# =============================================================
print("Generating Category 6…")

category = "many-query"

for i in range(100):
    L = random.randint(5, 15)
    add_pattern(f"query_{i}", extract_random_subsequence(L), category)


# =============================================================
# SAVE CSV
# =============================================================
print("Writing CSV…")

print("DONE! Generated", len(dataset), "patterns across 6 folders.")
