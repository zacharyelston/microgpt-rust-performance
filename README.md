# MicroGPT: The Art of Symmetry

> "Truth is the intersection of beauty and functionality."

A minimalist, aesthetic implementation of a Transformer in Rust. This project is an exploration of **symmetry in systems**. By reducing the GPT architecture to its atomic components and defining their interactions through fundamental operators, we reveal the elegant mathematical structure underlying modern AI.

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

The entire implementation is ~240 lines of code in `src/main.rs` with zero ML dependencies.

```bash
# Build and run with default settings
cargo run --release
```

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
*   **Positive Return:** Learning efficiently.
*   **Near Zero:** Diminishing returns.
*   **Negative:** Degradation/Instability.

### 3. CLI Reference
You can manually run specific configurations:
```bash
./target/release/microgpt_rust --steps 5000 --ctx 64 --emb 128 --head 8 --layer 4
```

## Performance

While prioritizing beauty and simplicity, Rust's efficiency still delivers respectable performance.

| Version | Time (1000 steps) |
|---------|-------------------|
| Python (Original) | ~62s |
| **Rust (Artistic)** | **~35s** |
| Rust (Optimized) | ~1.5s |

*Note: The "Optimized" version (on the main branch) sacrifices simplicity for raw speed using fused kernels. This "Artistic" branch uses pure, atomic autograd for conceptual elegance.*

## Acknowledgments

- **Andrej Karpathy** for the original [microGPT](https://github.com/karpathy/microGPT).
- **The Rust Language** for allowing high-level abstractions with low-level control.
