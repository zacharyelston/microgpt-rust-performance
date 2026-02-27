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

### IV. The Evolution Engine
The `evolve_loss` binary treats hyperparameters as DNA:
- **Tournament selection** (k=3) — any organism can become a parent by winning its bracket
- **Random immigrants** (2/gen) — fresh DNA injected to prevent stagnation
- **Multi-gene mutation** — 1–3 parameters change per mutation event
- **Panic recovery** — crashed configs get MAX loss instead of killing the run
- **Diversity tracking** — each generation reports unique architecture count

The search space covers embedding size (8–32), heads (1–4), layers (1–4), context window (8–24), FF multiplier (1–4), learning rate (log-uniform 0.001–0.05), and training steps (100–2000).

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
src/bin/evolve_loss.rs  Loss evolution engine with diversity-aware selection
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
