# MicroGPT: The Art of Symmetry

## Overview

A minimalist Rust implementation of a Transformer (GPT), porting Karpathy's microGPT with automatic differentiation from scratch. Includes parallel evolutionary engines that evolve hyperparameters for aesthetic name generation and loss minimization.

## Project Structure

```
├── src/
│   ├── lib.rs            # Shared library: Val autograd, GPT model, training logic
│   ├── main.rs           # Main binary: CLI training + generation
│   └── bin/
│       ├── evolve.rs     # Aesthetic evolution binary (parallel via rayon)
│       └── evolve_loss.rs # Loss-targeting evolution binary (target < 1.9)
├── Cargo.toml            # Rust dependencies (rand, rayon)
├── input.txt             # Training dataset (names from makemore)
├── analyze_*.py          # Python analysis scripts for scaling experiments
├── run_*.sh              # Shell scripts for parameter sweep experiments
├── README.md             # Project documentation
└── COMPARISON.md         # Performance comparison notes
```

## Running

```bash
cargo run --release                       # Main training + generation
cargo run --release --bin evolve          # Aesthetic evolutionary search
cargo run --release --bin evolve_loss     # Loss-targeting evolution (< 1.9)
```

## Architecture

- `Val` type with `Rc<RefCell<Node>>` for autograd computation graph
- Operator overloading via `op!` macro (Add, Sub, Mul)
- GPT struct: embeddings, multi-head attention, MLP, RMSNorm
- `TrainingConfig` struct for parameterized training
- `TrainingResult` struct returns names, final_loss, num_params
- `train_and_generate()` shared function used by all binaries
- `load_training_data()` and `build_vocab()` shared data loading
- Aesthetic evolution: fitness scoring (flow, symmetry, creativity)
- Loss evolution: targets loss < 1.9 with crossover + mutation

## Dependencies

- `rand = "0.8"` — random number generation
- `rayon = "1.10"` — parallel iteration for evolutionary engines

## Workflow

- **Start application**: `cargo run --release --bin evolve_loss` (console output)
