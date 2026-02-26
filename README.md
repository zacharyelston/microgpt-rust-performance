# MicroGPT: The Art of Symmetry

> "Truth is the intersection of beauty and functionality."

A minimalist, aesthetic implementation of a Transformer in Rust. This project is an exploration of **symmetry in systems**, porting [Andrej Karpathy's microGPT](https://karpathy.github.io/2026/02/12/microgpt/) to Rust. By reducing the GPT architecture to its atomic components and defining their interactions through fundamental operators, we reveal the elegant mathematical structure underlying modern AI.

## The Philosophy

### I. The Atom: `Val`
At the heart of the system lies the **Value** (`Val`). It is the indivisible atom of our universe.
- It holds **Data** (the reality).
- It holds **Gradient** (the potential for change).
- It remembers its **History** (the provenance of its existence).

```rust
struct Val(Rc<RefCell<Node>>);
struct Node { data: f64, grad: f64, prev: Vec<(Val, f64)> }
```

### II. The Algebra: Operations
We define interactions not as distinct functions, but as fundamental operators. The symmetry here is in the **Operator Overloading**:

```rust
op!(Add, add, +);
op!(Mul, mul, *);
```

Whether adding two `Val`s, a `Val` and a reference, or a reference and a `Val`, the interaction is identical. The chain rule of calculus (`backward`) is inherent in every interaction, automatically weaving the graph of computation.

### III. The Architecture
The GPT model itself is a study in fractal symmetry:
- **Linear Layer**: A transformation of space.
- **Attention**: The mechanism of relating different points in time.
- **MLP**: The mechanism of processing information at a single point in time.

## Quick Start

The entire core implementation is ~240 lines of code in `src/lib.rs` and `src/main.rs` with zero ML dependencies.

```bash
# Build and run the standard training loop
cargo run --release
```

## Evolutionary Engine

This repository includes a parallel evolutionary engine that treats the MicroGPT configuration as "DNA". It evolves hyperparameters (embedding size, layers, heads, learning rate) to maximize the **aesthetic quality** of generated names (Flow, Symmetry, Creativity).

The engine is written in pure Rust and uses `rayon` for parallel processing across all CPU cores.

```bash
# Run the evolutionary search
cargo run --release --bin evolve
```

### Fitness Function
The "Judge" evaluates generated names based on:
1.  **Flow**: Pronounceability (alternating vowel/consonant patterns).
2.  **Symmetry**: Palindromes and repeating sub-patterns.
3.  **Creativity**: Penalty for memorizing training data; reward for novelty.

## Scaling & Parameter Sweeps

This repository includes tools to analyze how MicroGPT scales with **training steps** and **model size**.

### 1. Running Tests
```bash
# Step Scaling (Time): Train models for 1k, 3k, ... 21k steps
./run_scaling_tests.sh

# Parameter Scaling (Size): Train Small, Medium, and Large models
./run_param_scaling.sh
```

### 2. Analyzing Results
We provide Python scripts to analyze the output logs:

```bash
# View basic tables of Loss vs Steps/Params
python3 analyze_scaling.py
python3 analyze_param_scaling.py

# Calculate "Return on Investment" (Loss drop per 1,000 units)
python3 analyze_efficiency.py
```

## Acknowledgments

- **Andrej Karpathy** for the original [microGPT](https://github.com/karpathy/microGPT) and his [blog post](https://karpathy.github.io/2026/02/12/microgpt/).
- **The Rust Language** for allowing high-level abstractions with low-level control.
