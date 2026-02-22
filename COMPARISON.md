# microGPT: Python vs Rust Comparison

## Overview
This directory contains two implementations of Karpathy's microGPT—a minimal GPT implementation from scratch with autograd.

- **Python version** (`microgpt.py`): Original reference implementation with full autograd
- **Rust version** (`src/main.rs`): Rust port with simplified gradient computation

## Results

### Python Version
```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.6497

--- inference (new, hallucinated names) ---
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
sample 11: konna
sample 12: keylen
sample 13: liole
sample 14: alerin
sample 15: earan
sample 16: lenne
sample 17: kana
sample 18: lara
sample 19: alela
sample 20: anton
```

### Rust Version (Current)
```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 4.2898

--- inference (new, hallucinated names) ---
sample  1: qavyebdvmbcwkuma
sample  2: nenqdbapyibcjwmf
sample  3: ncdumskajjqcjwsw
sample  4: mfqzznbtmjenmcfk
sample  5: vn
sample  6: neadqlrvhmmcseug
sample  7: rfnsqyqfjducmcat
sample  8: qlsrqraernfnbapv
sample  9: hkigqroptayljvcs
sample 10: ovffejgwncfcge
sample 11: cunqxbkpxgbwbdz
sample 12: rvei
sample 13: xhvfeobmsejuhwi
sample 14: qisfehgmmefhpnse
sample 15: kaegexojwmfhxs
sample 16: ig
sample 17: mzspbdvcsgmansss
sample 18: aagdf
sample 19: nqdxgfqns
sample 20: nbpgdxjdbmsnbvlu
```

## Key Differences

### Architecture
- **Python**: Full computation graph with automatic differentiation via `Value` class
  - Tracks computation history
  - Implements chain rule via `backward()` method
  - Computes true gradients for each parameter

- **Rust**: Simplified forward-pass only implementation
  - No computation graph tracking
  - Uses placeholder random gradients for optimizer updates
  - Focuses on architectural correctness rather than training

### Loss Trajectory
- **Python**: 2.6497 (converged)
- **Rust**: 4.2898 (higher, due to random gradients)

### Generated Samples
- **Python**: Produces realistic name-like sequences (kamon, ann, karai, etc.)
- **Rust**: Produces random character sequences (no learning occurred)

## Implementation Notes

The Rust version demonstrates the architectural structure of the model (embeddings, multi-head attention, MLP blocks, RMSNorm) but lacks the critical autograd system. To make the Rust version fully functional, it would need:

1. **Computation Graph**: Track all operations and their dependencies
2. **Gradient Computation**: Implement backpropagation through the graph
3. **Parameter Tracking**: Link gradients back to specific parameters
4. **Proper Adam Updates**: Use computed gradients instead of random values

This is a non-trivial undertaking in Rust due to:
- Reference counting and borrowing constraints
- Need to track computation history
- Complex gradient flow through nested operations

## Building and Running

### Python
```bash
cd /Users/zacelston/code/microgpt
python3 microgpt.py
```

### Rust
```bash
cd /Users/zacelston/code/microgpt
cargo build --release
./target/release/microgpt_rust
```

## Conclusion

The Python version successfully trains a minimal GPT on name generation, achieving reasonable loss and producing plausible samples. The Rust version demonstrates the forward-pass architecture but would require a full autograd implementation to achieve comparable training results.
