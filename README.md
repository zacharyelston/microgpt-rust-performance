# MicroGPT: The Art of Symmetry

> "Truth is the intersection of beauty and functionality."

A minimalist, aesthetic implementation of a Transformer in Rust.

This project is an exploration of **symmetry in systems**. By reducing the GPT architecture to its atomic components and defining their interactions through fundamental operators, we reveal the elegant mathematical structure underlying modern AI.

## The Philosophy

- **The Atom**: A single `Val` struct representing both data and gradient, history and potential.
- **The Algebra**: Operator overloading (`+`, `*`, `-`) allows the neural network to be expressed as pure mathematical equations, with the chain rule of calculus woven invisibly into every interaction.
- **The Architecture**: A fractal structure of Linear layers, Attention mechanisms, and MLPs, all built from the same atomic units.

See [SYMMETRY.md](SYMMETRY.md) for the manifesto.

## The Code

The entire implementation is contained in a single file: `src/main.rs`.
- **~240 Lines of Code**
- **Zero ML Dependencies** (only `rand`)
- **Full Autograd Engine**
- **GPT Training & Inference**

## Quick Start

```bash
cargo run --release
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
