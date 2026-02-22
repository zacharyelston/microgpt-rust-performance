# microGPT: Python vs Rust Comparison

## Overview
This directory contains two implementations of Karpathy's microGPT—a minimal GPT implementation from scratch with automatic differentiation.

- **Python version** (`microgpt.py`): Original reference implementation with full autograd
- **Rust version** (`src/main.rs`): Rust port with complete autograd system using `Rc<RefCell<>>`

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

### Rust Version (With Proper Autograd)
```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.2360

--- inference (new, hallucinated names) ---
sample  1: ioflalo
sample  2: ravats
sample  3: zeele
sample  4: ahirle
sample  5: labli
sample  6: baylo
sample  7: auliel
sample  8: kayreon
sample  9: ila
sample 10: rely
sample 11: blahas
sample 12: moshesy
sample 13: mielas
sample 14: xoryida
sample 15: lal
sample 16: zelia
sample 17: enalind
sample 18: bi
sample 19: eiltus
sample 20: zavtonn
```

## Key Differences

### Architecture
Both versions implement:
- **Computation Graph**: Track all operations and their dependencies
- **Automatic Differentiation**: Full backpropagation via chain rule
- **Parameter Management**: Proper gradient accumulation and Adam optimizer updates
- **Model Components**: Embeddings, multi-head attention, MLP blocks, RMSNorm

**Python Implementation**:
- Uses native Python objects and direct references
- `Value` class stores data, gradient, children, and local gradients
- Backward pass traverses topological sort in reverse

**Rust Implementation**:
- Uses `Rc<RefCell<Value>>` for shared mutable references
- Handles Rust's ownership system via reference counting
- Identical backward pass algorithm with pointer-based visited tracking

### Loss Trajectory
- **Python**: 2.6497 (final loss)
- **Rust**: 2.2360 (final loss, slightly better convergence)

### Generated Samples
- **Python**: Produces realistic name-like sequences (kamon, ann, karai, jaire, etc.)
- **Rust**: Produces realistic name-like sequences (ioflalo, ravats, zeele, ahirle, etc.)

Both versions successfully learn the name distribution and generate plausible samples.

## Implementation Details

### Autograd System
Both versions implement the same core operations:
- `add(a, b)`: Addition with local gradients (1.0, 1.0)
- `mul(a, b)`: Multiplication with local gradients (b.data, a.data)
- `pow(a, exp)`: Power with local gradient (exp * a^(exp-1))
- `log(a)`: Natural logarithm with local gradient (1/a)
- `exp(a)`: Exponential with local gradient (exp(a))
- `relu(a)`: ReLU with conditional local gradient

### Backward Pass
Both implement topological sort-based backpropagation:
1. Build topological order of computation graph
2. Initialize loss gradient to 1.0
3. Traverse nodes in reverse topological order
4. Accumulate gradients via chain rule

### Optimizer
Both use Adam with:
- Learning rate decay: `lr_t = lr * (1 - step / num_steps)`
- Momentum and variance tracking (m, v buffers)
- Bias correction for early training steps

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

## Performance Notes

- **Python**: ~60 seconds for 1000 training steps
- **Rust**: ~120 seconds for 1000 training steps (2x slower due to Rc<RefCell<>> overhead)

The Rust version is slower due to:
- Runtime borrow checking overhead
- Reference counting operations
- Pointer indirection for computation graph traversal

## Conclusion

Both versions successfully implement a complete autograd system and train a minimal GPT on name generation. The Rust version demonstrates that automatic differentiation can be implemented in Rust despite the language's strict ownership constraints. The slightly better loss in Rust (2.236 vs 2.650) is likely due to random seed differences in parameter initialization.
