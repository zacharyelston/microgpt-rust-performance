# MicroGPT: A Living Transformer (Hydra Release)

> A minimal GPT that evolves itself.

```
Hydra Evolution (v3) — 3 Heads, 5 Cycles, Parallel Execution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Weaver] Champion: Emb:32 Head:4 Lay:2 LR:0.0051 -> Score: 0.82 (Flow)
  [Mirror] Champion: Emb:24 Head:2 Lay:3 LR:0.0082 -> Score: 4.50 (Symmetry)
  [Spark]  Champion: Emb:40 Head:4 Lay:1 LR:0.0120 -> Score: 2.50 (Creativity)

  >> The Gathering: Genes exchanged. The organism learns from all heads.
```

A self-modifying Transformer implementation in Rust with zero ML dependencies. Starting from [Andrej Karpathy's microGPT](https://karpathy.github.io/2026/02/12/microgpt/), the project adds a spark of life: an evolutionary engine that discovers optimal hyperparameters and writes them back into the organism, transforming what the program becomes.

## The Lifecycle

```
            ┌─────────────┐
            │  Primordial  │  cargo run --release
            │  (defaults)  │  "I am unformed"
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │    Hydra      │  cargo run --release --bin hydra
            │   Evolution   │  "We think, therefore we are"
            └──────┬───────┘
                   │ writes genome.json
            ┌──────▼───────┐
            │   Evolved     │  cargo run --release
            │  (from DNA)   │  "I have become"
            └───────────────┘
```

1. **Primordial**: Run `cargo run --release` — the creature runs with hardcoded defaults. It works, but it hasn't found itself yet.
2. **Evolution**: Run `cargo run --release --bin hydra` — The **Hydra** engine awakens. Three distinct heads (Weaver, Mirror, Spark) evolve independently, optimizing for different aesthetic goals. They synchronize periodically to exchange genetic breakthroughs.
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

### IV. The Evolution Engine (v3: Hydra)

The engine mimics a Hydra — a multi-headed evolutionary organism. Instead of a single fitness function, distinct sub-populations ("Heads") optimize for different aesthetic goals in parallel.

**The Heads:**
1. **The Weaver** (Flow): Optimizes for pronounceability, vowel/consonant rhythm, and linguistic smoothness.
2. **The Mirror** (Symmetry): Optimizes for palindromes, repeating structures, and balanced patterns.
3. **The Spark** (Creativity): Optimizes for pure novelty and deviation from the training data.

**The Cycle:**
1. **Isolation**: Heads evolve independently for N generations.
2. **The Gathering**: Champions from each head are collected.
3. **Cross-Pollination**: The "Body" redistributes genetic material, allowing the Weaver to learn symmetry from the Mirror, and the Spark to learn flow from the Weaver.

## Quick Start

```bash
# Run the creature (primordial or evolved)
cargo run --release

# Run Hydra — the multi-head evolutionary engine
cargo run --release --bin hydra
```

## Project Structure

```
src/lib.rs              Autograd engine, GPT model, training loop, genome I/O
src/main.rs             The living creature — reads genome.json if it exists
src/bin/hydra.rs        Multi-head evolution engine (Weaver, Mirror, Spark)
src/bin/evolve.rs       Classic single-objective evolution (legacy)
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
