import random

input_file = "ecoli_clean.txt"     # your big genome txt file (5.2M bases)
output_file = "ecoli_500k.txt"    # new smaller file

# Desired reduced length
target_length = 500_000

# Load full genome sequence (assuming one continuous sequence, no headers)
with open(input_file, 'r') as f:
    genome = f.read().replace('\n', '').upper()

genome_length = len(genome)
print(f"Original genome length: {genome_length}")

if genome_length < target_length:
    print(f"Genome is already smaller than {target_length} bases.")
else:
    # Pick random start index so end fits in genome
    start = random.randint(0, genome_length - target_length)
    trimmed_seq = genome[start:start + target_length]
    print(f"Extracted subsequence from {start} to {start + target_length}")

    # Save trimmed sequence to output file (80 chars per line for readability)
    with open(output_file, 'w') as out:
        for i in range(0, len(trimmed_seq), 80):
            out.write(trimmed_seq[i:i+80] + '\n')

    print(f"Trimmed genome saved to {output_file}")
