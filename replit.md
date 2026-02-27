# microGPT Rust vs Python Performance Comparison

## Overview

A command-line implementation of Karpathy's microGPT, comparing performance between Python and Rust. Implements a tiny GPT-style transformer with automatic differentiation from scratch.

## Project Structure

```
microgpt/
├── src/main.rs       # Rust implementation (optimized)
├── microgpt.py       # Python implementation (optimized)
├── Cargo.toml        # Rust dependencies
├── Cargo.lock        # Dependency lockfile
├── input.txt         # Training dataset (names from makemore)
├── README.md         # Performance comparison results
└── COMPARISON.md     # Detailed comparison notes
```

## Running the Project

### Rust (main workflow)
```bash
cargo run --release
```

### Python
```bash
python3 microgpt.py
```

## Architecture

- GPT-style transformer with multi-head attention
- Custom automatic differentiation via computation graphs
- Adam optimizer with learning rate decay and bias correction
- RMSNorm layer normalization
- 1 transformer layer, 16 embedding dims, 4 attention heads
- 4,192 total parameters trained on 32,033 names

## Dependencies

- `rand = "0.8"` — random number generation
- `reqwest = "0.11"` (blocking) — HTTP client for downloading dataset

## Performance Results

| Version | Time | Graph Size |
|---------|------|------------|
| Original Python | 62.3s | ~68k nodes |
| Optimized Python | 20.6s | ~6k nodes |
| Original Rust | 12.7s | ~68k nodes |
| Optimized Rust | 1.48s | ~6k nodes |

Rust is ~14x faster than optimized Python using the same algorithm.

## Workflow

- **Start application**: `cargo run --release` (console output)
