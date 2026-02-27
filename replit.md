# MicroGPT: A Living Transformer

## Overview

A minimalist Rust implementation of a self-evolving Transformer (GPT), porting Karpathy's microGPT with automatic differentiation from scratch. The program evolves itself: the evolution engine discovers optimal hyperparameters and writes them to `genome.json`, transforming what the main program becomes.

## How It Works

1. **Primordial state**: `cargo run --release` runs with default parameters
2. **Evolution**: `cargo run --release --bin evolve_loss` evolves hyperparameters across generations
3. **Self-modification**: Evolution writes the winning genome to `genome.json`
4. **Evolved state**: `cargo run --release` now runs as the evolved creature

## Project Structure

```
├── src/
│   ├── lib.rs            # Shared library: Val autograd, GPT model, genome save/load
│   ├── main.rs           # Main binary: runs as evolved or primordial creature
│   └── bin/
│       ├── evolve.rs     # Aesthetic evolution binary (parallel via rayon)
│       └── evolve_loss.rs # Loss-targeting evolution engine (target < 1.2)
├── genome.json            # The organism's evolved DNA (written by evolution)
├── experiments/           # Auto-generated experiment logs (timestamped)
├── Cargo.toml            # Rust dependencies (rand, rayon, chrono)
├── input.txt             # Training dataset (names from makemore)
├── analyze_*.py          # Python analysis scripts for scaling experiments
├── run_*.sh              # Shell scripts for parameter sweep experiments
├── README.md             # Project documentation
└── COMPARISON.md         # Performance comparison notes
```

## Running

```bash
cargo run --release                       # Run the creature (evolved or primordial)
cargo run --release --bin evolve_loss     # Evolve — rewrites genome.json
cargo run --release --bin evolve          # Aesthetic evolutionary search
```

## Architecture

- `Val` type with `Rc<RefCell<Node>>` for autograd computation graph
- Operator overloading via `op!` macro (Add, Sub, Mul)
- GPT struct: embeddings, multi-head attention, MLP, RMSNorm
- `TrainingConfig` struct with `save_genome()` / `load_genome()` for self-modification
- `TrainingResult` struct returns names, final_loss, num_params
- `train_and_generate()` shared function used by all binaries
- `load_training_data()` and `build_vocab()` shared data loading
- Aesthetic evolution: fitness scoring (flow, symmetry, creativity)
- Loss evolution: tournament selection, diversity-aware, panic recovery
  - Experiment results saved to `experiments/evolve_YYYYMMDD_HHMMSS.log`
  - Debug logging via stderr for real-time organism evaluation tracking
  - Writes `genome.json` on completion — the program becomes its best self

## Dependencies

- `rand = "0.8"` — random number generation
- `rayon = "1.10"` — parallel iteration for evolutionary engines
- `chrono = "0.4"` — timestamped experiment filenames

## Workflow

- **Start application**: `cargo run --release --bin evolve_loss` (evolution engine)
