# MicroGPT: Python vs Rust Comparison

## Overview
This document compares two implementations of Karpathy's microGPT:
- **Python version** (Original `microgpt.py`): Reference implementation (see [original blog](https://karpathy.github.io/2026/02/12/microgpt/)).
- **Rust version** (`src/main.rs`): Port using `Rc<RefCell<>>` for autograd.

## Performance Results

| Version | Time | Speedup | Loss | Graph Size | Status |
|---------|------|---------|------|-------------|---------|
| Original Python | 62.3s | 1x | 2.65 | ~68k nodes | Baseline |
| **Optimized Python** | **20.6s** | **3.0x** | **2.64** | ~6k nodes | ✓ Working |
| Original Rust | 12.7s | 4.9x | 2.29 | ~68k nodes | ✓ Working |
| **Optimized Rust** | **1.48s** | **42x** | **2.41** | ~6k nodes | ✓ Working |

**Final result: Optimized Rust is 14x faster than optimized Python.**

### Generated Samples (Truncated)
Both versions learn the dataset (names) distribution effectively.

**Python:** `kamon`, `ann`, `karai`, `jaire`, `vialan`...
**Rust:** `ioflalo`, `ravats`, `zeele`, `ahirle`, `labli`...

## Key Architectural Differences

| Feature | Python Implementation | Rust Implementation |
|---------|-----------------------|---------------------|
| **Objects** | Native Python objects | `Rc<RefCell<Value>>` structs |
| **Memory** | Garbage Collected | Reference Counted (Manual) |
| **Dispatch** | Dynamic (Runtime) | Static (Compile-time) |
| **Graph** | Direct references | Interior Mutability Pattern |

## Optimization Techniques

The dramatic speedups (Phase 3 & 4) were achieved through identical algorithmic improvements in both languages:

1.  **Linear Operation Fusion**:
    -   *Problem*: `y = w * x + b` created thousands of intermediate nodes.
    -   *Fix*: Combined the dot product and bias add into a single computation node.
    -   *Impact*: Reduced graph size from **68,000** to **6,000** nodes per step (91% reduction).

2.  **Stack-Based Backward Pass**:
    -   *Problem*: Recursive topological sorting was slow and memory-intensive.
    -   *Fix*: Replaced with an iterative stack-based traversal.
    -   *Impact*: Backward pass time dropped from 79% to 66% of total runtime.

## Conclusion

1.  **Algorithms Matter Most**: The biggest gains came from reducing graph size (91% reduction), benefiting both languages.
2.  **Rust has a High Ceiling**: Even with identical logic, Rust's compiled nature and zero-cost abstractions allow it to be **14x faster** than the best Python version.
3.  **Safety & Speed**: The Rust implementation proves that complex pointer-based graphs (`Rc<RefCell>`) can be implemented safely without sacrificing performance.
