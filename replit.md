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
│       ├── ab_test.rs     # A/B comparison: words vs noise vs names
│       ├── evolve.rs     # Aesthetic evolution binary (parallel via rayon)
│       └── evolve_loss.rs # Loss evolution engine v2 (species-aware)
├── data/
│   ├── words.txt          # 7,622 common English words (2-8 chars)
│   └── nonwords.txt       # 7,622 random letter strings (matched lengths)
├── genome.json            # The organism's evolved DNA (written by evolution)
├── experiments/           # Auto-generated experiment logs (timestamped)
├── research/             # Analysis reports from evolution experiments
│   ├── embedding_size_analysis.md  # Scaling curve Emb:4-32
│   ├── emb4_10gen_complexity_penalty.md  # Emb:4, 10 generations
│   ├── emb4_20gen_long_run.md     # Emb:4, 20 generations deep study
│   ├── emb2_8_spectrum_sweep.md   # Emb:2-8 mixed competition
│   └── step_ceiling_and_penalty_experiments.md  # Step cap & penalty tuning
├── Cargo.toml            # Rust dependencies (rand, rayon, chrono)
├── input.txt             # Training dataset (names from makemore)
├── README.md             # Project documentation
└── COMPARISON.md         # Performance comparison notes
```

## Evolution Engine v2 Mechanics

The evolution engine now operates at the **species level**, not just individual organisms:

- **Species tracking**: Organisms are grouped by architecture family (Emb-Head-Lay-Ctx-FF). Monoculture (one species dominating) is tracked and reported.
- **Loser blacklist**: Architectures that fail consistently (loss > 2.3 across 2+ samples) are remembered and avoided when generating new organisms.
- **Growth mutation (Fibonacci/polydactyl)**: Proven winners earn structural upgrades — an extra layer, doubled heads, expanded context, or wider FF. The organism literally grows.
- **Championship breeding**: On mild stagnation (2 gens), the top 3 winners are mated together with fine-tuning mutations. Elite is force re-evaluated.
- **Cataclysm**: On deep stagnation (4+ gens), the population is blown up and rebuilt from an expanded search space, avoiding blacklisted species.
- **Fine-tuning**: Championship mode generates variants of the winner with small LR/steps tweaks instead of random mutations.
- **Complexity penalty**: Fitness = loss + 2% per log-unit of energy cost. Simpler organisms are preferred when loss is similar. Energy is estimated as params * steps.

### Stagnation Response Ladder

| Stagnation | Response | Strategy |
|-----------|----------|----------|
| 0-1 | Normal breeding | Elite + immigrants + crossover/mutant/hyper |
| 2-3 | Championship | Re-eval elite + growth mutation + mate top 3 + fine-tune |
| 4+ | Cataclysm | Re-eval elite + wide random search (avoiding blacklist) |

### Origin Tags

Each organism's creation method is tracked: `[random]`, `[elite]`, `[mutant]`, `[cross]`, `[hyper]`, `[immigrant]`, `[re-eval]`, `[cataclysm]`, `[grown]`, `[champion]`, `[tuned]`

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

## Dependencies

- `rand = "0.8"` — random number generation
- `rayon = "1.10"` — parallel iteration for evolutionary engines
- `chrono = "0.4"` — timestamped experiment filenames

## Workflow

- **Start application**: `cargo run --release --bin evolve_loss` (evolution engine)
