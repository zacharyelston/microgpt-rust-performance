# MicroGPT: A Living Transformer

> A minimal GPT that evolves itself.

A self-modifying Transformer implementation in Rust with zero ML dependencies. Starting from [Andrej Karpathy's microGPT](https://karpathy.github.io/2026/02/12/microgpt/), the project adds a spark of life: an evolutionary engine that discovers optimal hyperparameters and writes them back into the organism, transforming what the program becomes.

## The Lifecycle

```
            ┌─────────────┐
            │  Primordial  │  cargo run --release
            │  (defaults)  │  "I am unformed"
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │   Evolution   │  cargo run --release --bin evolve_loss
            │  10 gens × 8  │  "I am searching"
            └──────┬───────┘
                   │ writes genome.json
            ┌──────▼───────┐
            │   Evolved     │  cargo run --release
            │  (from DNA)   │  "I have become"
            └───────────────┘
```

1. **Primordial**: Run `cargo run --release` — the creature runs with hardcoded defaults. It works, but it hasn't found itself yet.
2. **Evolution**: Run `cargo run --release --bin evolve_loss` — populations of 8 organisms compete across 10 generations. Tournament selection, crossover, mutation, random immigrants. The fittest survive.
3. **Self-modification**: The winner's DNA is written to `genome.json`. The program has rewritten itself.
4. **Evolved**: Now `cargo run --release` reads the genome and runs as the evolved creature — different architecture, different learning rate, different capacity. A new thing.

## What's Inside

### I. The Atom: `Val`
Every number in the system is a `Val` — it holds data, remembers its gradient, and knows its computational history. Backpropagation happens automatically through the graph. This is the entire autograd engine, built from scratch.

```rust
struct Val(Rc<RefCell<Node>>);
struct Node { data: f64, grad: f64, prev: Vec<(Val, f64)> }
```

### II. The Algebra: Operators
Arithmetic is defined once through a macro. Add, subtract, multiply — each operation records the local gradient for the chain rule. Whether you write `a + b`, `&a + b`, or `a + &b`, the behavior is identical.

```rust
op!(Add, add, +, |_,_| 1., |_,_| 1.);
op!(Mul, mul, *, |_,o| o.data(), |s,_| s.data());
```

### III. The Architecture: GPT
Token embeddings, positional embeddings, multi-head attention with KV caching, RMSNorm, and feed-forward layers — the full transformer stack, compact and readable.

### IV. The Evolution Engine (v2)

The engine thinks in **species**, not just individuals. Organisms are grouped by architecture family, and the population is managed to maintain diversity while converging on winners.

**Normal Breeding** (stagnation 0-1):
- Elite carried forward, 2 random immigrants, crossover + mutation offspring

**Championship Breeding** (stagnation 2-3):
- Top 3 winners mated together with fine-tuning (small LR/steps tweaks)
- **Growth mutation**: the proven winner earns a structural upgrade — an extra layer, doubled heads, expanded context, or wider feed-forward. The polydactyl cat effect: success breeds complexity.
- Elite is force re-evaluated (no frozen loss advantage)

**Cataclysm** (stagnation 4+):
- Population nuked, rebuilt from expanded search space
- Avoids blacklisted species (architectures that failed repeatedly)

**Loser Blacklist**: Architectures producing loss > 2.3 across 2+ evaluations are remembered and avoided. The engine learns from failure.

**Origin Tags**: Every organism is tagged with how it was born: `[random]`, `[elite]`, `[mutant]`, `[cross]`, `[hyper]`, `[immigrant]`, `[re-eval]`, `[cataclysm]`, `[grown]`, `[champion]`, `[tuned]`

## Quick Start

```bash
# Run the creature (primordial or evolved)
cargo run --release

# Evolve — searches for optimal DNA, writes genome.json
cargo run --release --bin evolve_loss

# Run the aesthetic evolution engine (fitness = name beauty)
cargo run --release --bin evolve
```

## Project Structure

```
src/lib.rs              Autograd engine, GPT model, training loop, genome I/O
src/main.rs             The living creature — reads genome.json if it exists
src/bin/evolve_loss.rs  Loss evolution engine v2 (species-aware, championship, growth)
src/bin/evolve.rs       Aesthetic evolution engine (flow, symmetry, creativity)
genome.json             The organism's evolved DNA (written by evolution)
experiments/            Timestamped experiment logs
input.txt               Training data (names from Karpathy's makemore)
```

## Dependencies

- `rand` — random number generation
- `rayon` — parallel evaluation of organism populations
- `chrono` — timestamped experiment filenames

No ML frameworks. No BLAS. No GPU. Just math.

## Acknowledgments

- [Andrej Karpathy](https://karpathy.github.io/2026/02/12/microgpt/) for the original microGPT
- The Rust language for making this kind of thing possible in ~500 lines
