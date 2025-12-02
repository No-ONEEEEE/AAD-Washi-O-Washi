# Washi O Washi - Interactive Pattern Matching Visualizer

An interactive web-based platform for visualizing and understanding string pattern matching algorithms. Features step-by-step visualization, real-time comparisons, and educational insights into how different algorithms work.

## Team Members
- **Yogansh**
- **Chanakya**  
- **Saketh**
- **Navadeep**
- **Mahanth**

---

## Project Overview

**Washi O Washi** (Japanese for "Matching the Matching") is an interactive educational tool for exploring pattern matching algorithms through step-by-step visualization. The web application provides:

- **7 Implemented Algorithms** with complete visualizers
- **Interactive Step-by-Step Visualization** of algorithm execution
- **Real-time Performance Metrics** and statistics
- **DNA Sequence Analysis** with sample genomic datasets
- **Algorithm Comparison** with detailed explanations
- **Modern Responsive UI** with beautiful glassmorphic design

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Running

```bash
# 1. Clone or navigate to the project directory
cd "AAD web_application"

# 2. Install required dependencies
pip install flask

# 3. Run the web application
python web/app.py

# 4. Open your browser and visit
http://localhost:5000
```

That's it! The application is now running.

---

## Project Structure

```
AAD web_application/
├── algorithms/
│   ├── __init__.py                    # Package initialization
│   ├── exact_matching.py              # 5 exact matching algorithms
│   └── approximate_matching.py        # 2 approximate matching algorithms
├── web/
│   ├── app.py                         # Flask web server (MAIN ENTRY POINT)
│   ├── templates/
│   │   └── index.html                 # Complete UI with all visualizers
│   └── static/
│       └── uploads/                   # User file uploads directory
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Implemented Algorithms

### **Exact Pattern Matching (5 Algorithms)**

| Algorithm | Time Complexity | Space | Description |
|-----------|----------------|-------|-------------|
| **Naive** | O(nm) | O(1) | Simple brute-force, educational purposes |
| **KMP** | O(n+m) | O(m) | Uses failure function, no text backtracking |
| **Boyer-Moore** | O(nm) worst, O(n/m) avg | O(σ+m) | Right-to-left scan, bad character table |
| **Rabin-Karp** | O(n+m) avg, O(nm) worst | O(1) | Rolling hash, detects spurious hits |
| **Suffix Tree** | O(m) search, O(n) build | O(n) | Tree traversal, multiple pattern queries |

### **Approximate Pattern Matching (2 Algorithms)**

| Algorithm | Time Complexity | Space | Description |
|-----------|----------------|-------|-------------|
| **Levenshtein Distance** | O(nm) | O(nm) | Edit distance with DP matrix visualization |
| **Shift-Or** | O(nm/w) | O(σ+k) | Bit-parallel fuzzy matching |

**n** = text length, **m** = pattern length, **σ** = alphabet size, **w** = word size, **k** = max errors

---

## Features

### **Interactive Visualizer**
- **Step-by-step execution** with play/pause controls
- **Character-level highlighting** showing comparisons
- **Real-time statistics** tracking algorithm performance
- **Algorithm state display** (pointers, indices, counters)
- **Summary dashboard** with comprehensive metrics

### **Algorithm-Specific Visualizations**

#### **Rabin-Karp**
- Hash value display for pattern and text windows
- Rolling hash calculation visualization
- Spurious hit detection (hash collisions)

#### **KMP (Knuth-Morris-Pratt)**
- Failure function (LPS array) table
- Backtrack event tracking
- No text backtracking demonstration

#### **Boyer-Moore**
- Bad character table display
- Right-to-left scanning visualization
- Skip efficiency metrics

#### **Suffix Tree**
- Tree traversal path display
- Node visitation tracking
- Suffix indexing visualization

#### **Levenshtein Distance**
- Dynamic Programming matrix (color-coded)
- Edit operations (match, substitute, insert, delete)
- Cost calculation breakdown
- Alignment display with traceback

#### **Shift-Or**
- Bit vector state visualization
- Pattern masks in binary format
- Bit-parallel operations display
- Multi-level error tracking

### **Sample Datasets**

- Simple DNA sequences
- E. coli gene fragments
- Mutation detection samples
- English text examples
- Highly repetitive patterns

### **Additional Features**

- File upload support (TXT, FASTA, FASTQ formats)
- Beautiful glassmorphic UI design
- Responsive layout for all devices
- Color-coded match/mismatch highlighting
- Real-time performance comparison

---

## Usage Guide

### **Basic Usage**

1. **Select Algorithm**: Choose from 7 available algorithms
2. **Enter Input**: Type or use sample datasets
   - Text: The string to search in
   - Pattern: The substring to find
3. **Start Visualization**: Click "Visualize Algorithm"
4. **Control Playback**: Use controls to step through execution
   - Previous Step
   - Play/Pause (auto-play)
   - Next Step
   - Speed control (0.5x to 2x)

### **Sample Datasets**

Click "Load Sample" and choose from:
- **Simple DNA**: Basic pattern matching in DNA
- **E. coli Gene**: Real genomic sequence with start codons
- **Mutation Detection**: Finding mutations in sequences
- **English Text**: Text processing examples
- **Repetitive**: Algorithm behavior with repeated patterns

## VISUALISATION DETAILS

### **What You'll See:**

1. **Text & Pattern Grid**
   - Character-by-character visual alignment
   - Current window highlighting
   - Match/mismatch color coding

2. **Comparison Details**
   - Individual character comparisons
   - Match/mismatch indicators
   - Comparison counter

3. **Algorithm State**
   - Text pointer (i)
   - Pattern pointer (j)
   - Current window position

4. **Algorithm-Specific Info**
   - Hash values (Rabin-Karp)
   - Failure function (KMP)
   - Bad character table (Boyer-Moore)
   - DP matrix (Levenshtein)
   - Bit vectors (Shift-Or)
   - Tree path (Suffix Tree)

5. **Statistics Summary**
   - Total comparisons
   - Matches found
   - Positions checked
   - Time/space complexity
   - Algorithm-specific metrics

