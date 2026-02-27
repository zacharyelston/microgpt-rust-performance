# MicroGPT: The Art of Symmetry

## Overview

A minimalist Rust implementation of a Transformer (GPT), porting Karpathy's microGPT with automatic differentiation from scratch. Includes a parallel evolutionary engine that evolves hyperparameters for aesthetic name generation.

## Project Structure

```
├── src/
│   ├── lib.rs            # Shared library: Val autograd, GPT model, training logic
│   ├── main.rs           # Main binary: CLI training + generation
│   └── bin/
│       └── evolve.rs     # Evolutionary engine binary (parallel via rayon)
├── Cargo.toml            # Rust dependencies (rand, rayon)
├── input.txt             # Training dataset (names from makemore)
├── analyze_*.py          # Python analysis scripts for scaling experiments
├── run_*.sh              # Shell scripts for parameter sweep experiments
├── README.md             # Project documentation
└── COMPARISON.md         # Performance comparison notes
```

## Running

```bash
cargo run --release                  # Main training + generation
cargo run --release --bin evolve     # Evolutionary hyperparameter search
```

## Architecture

- `Val` type with `Rc<RefCell<Node>>` for autograd computation graph
- Operator overloading via `op!` macro (Add, Sub, Mul)
- GPT struct: embeddings, multi-head attention, MLP, RMSNorm
- `TrainingConfig` struct for parameterized training
- `train_and_generate()` shared function used by both binaries
- Evolutionary engine: population of Genomes with fitness scoring (flow, symmetry, creativity)

## Dependencies

- `rand = "0.8"` — random number generation
- `rayon = "1.10"` — parallel iteration for evolutionary engine

## Workflow

- **Start application**: `cargo run --release` (console output, runs main binary)
