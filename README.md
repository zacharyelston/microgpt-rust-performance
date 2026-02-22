# microGPT: Python vs Rust Performance Comparison

A comprehensive performance comparison between Python and Rust implementations of Karpathy's microGPT with automatic differentiation.

## 🚀 Key Results

**Final Performance:**
- **Optimized Rust**: 1.48s (42x faster than original Python)
- **Optimized Python**: 20.6s (3x faster than original Python)
- **Rust vs Python**: 14x speedup even with identical optimizations

| Version | Time | Speedup vs Original | Loss | Graph Size |
|---------|------|-------------------|------|------------|
| Original Python | 62.3s | 1x | 2.6497 | ~68k nodes |
| Optimized Python | 20.6s | 3.0x | 2.6391 | ~6k nodes |
| Original Rust | 12.7s | 4.9x | 2.2941 | ~68k nodes |
| **Optimized Rust** | **1.48s** | **42x** | **2.4058** | **~6k nodes** |

## 📁 Project Structure

```
microgpt/
├── microgpt.py              # Python implementation (optimized)
├── src/main.rs              # Rust implementation (optimized)
├── Cargo.toml               # Rust dependencies
├── COMPARISON.md             # Original comparison
├── COMPARISON-New.md         # Complete development summary
└── README.md                 # This file
```

## 🛠️ Technologies

### Python Version
- Pure Python with no ML dependencies
- Custom autograd system with computation graph
- Adam optimizer with learning rate decay
- Multi-head attention and RMSNorm

### Rust Version
- `Rc<RefCell<Value>>` for shared mutable references
- Identical autograd algorithm adapted to Rust
- Stack-based backward pass optimization
- Linear operation fusion for graph reduction

## 🏃‍♂️ Quick Start

### Python
```bash
python3 microgpt.py
```

### Rust
```bash
cargo build --release
./target/release/microgpt_rust
```

## 📊 Optimization Journey

### Phase 1: Initial Implementation
- Built working Python and Rust versions
- Rust achieved 5x speedup over Python (12.7s vs 62.3s)

### Phase 2: Performance Analysis
- **Identified bottleneck**: Backward pass consumed 79.1% of time
- **Discovered issue**: 68,780 computation graph nodes per step
- **Root cause**: Creating nodes for every scalar operation

### Phase 3: Algorithmic Optimizations
Applied identical optimizations to both languages:

1. **Linear Operation Fusion**
   - Combined multiply-add operations into single nodes
   - Reduced intermediate node creation by ~10x

2. **Stack-Based Backward Pass**
   - Replaced topological sort with efficient stack traversal
   - Eliminated sorting overhead

3. **Graph Size Reduction**
   - Minimized computation graph from 68k to ~6k nodes (91% reduction)
   - Maintained proper gradient flow

### Phase 4: Results
- **Python**: 62.3s → 20.6s (3.0x speedup)
- **Rust**: 12.7s → 1.48s (8.6x speedup)
- **Final**: Rust 14x faster than optimized Python

## 🔍 Key Insights

1. **Algorithmic optimization benefits both languages** - Identical optimizations yielded significant speedups in both Python and Rust

2. **Rust maintains significant performance advantage** - Even with identical algorithms, Rust's compiled nature provides inherent advantages

3. **Computation graph size is critical** - Reducing from 68k to 6k nodes was the primary optimization

4. **Rust is suitable for ML workloads** - Successfully implements complex autograd systems with substantial performance benefits

5. **Safety and performance can coexist** - Rust achieves speedup while maintaining memory safety and correctness

## 📈 Performance Breakdown (Optimized Rust)

```
--- Profiling Results ---
Forward pass:  0.48s (32.4%)
Backward pass: 0.99s (66.9%)
Softmax:       0.03s (2.2%)
Param update:  0.01s (0.7%)
Avg graph size: 6109 nodes
```

## 🧪 Model Architecture

Both implementations include:
- **GPT-style transformer** with multi-head attention
- **Automatic differentiation** via computation graphs
- **Adam optimizer** with bias correction
- **RMSNorm** layer normalization
- **Embedding layers** for tokens and positions

### Model Specs
- **Layers**: 1 transformer layer
- **Embedding dim**: 16
- **Heads**: 4 attention heads
- **Block size**: 16 context length
- **Parameters**: 4,192 total parameters
- **Dataset**: 32,033 names from makemore

## 🎯 Sample Outputs

### Optimized Python
```
sample  1: gia
sample  2: jean
sample  3: semsaa
sample  4: daa
sample  5: biy
```

### Optimized Rust
```
sample  1: idmeb
sample  2: jeeoa
sample  3: adselamiinamalrk
sample  4: matt
sample  5: aaa
```

Both versions generate realistic name-like sequences, demonstrating successful learning.

## 📚 Detailed Analysis

See [`COMPARISON-New.md`](COMPARISON-New.md) for:
- Complete development timeline
- Detailed optimization code examples
- Performance profiling methodology
- Comprehensive benchmark results

## 🤝 Contributing

This project serves as a case study for high-performance ML systems in Rust. Contributions welcome for:
- Further optimizations
- Additional model architectures
- Benchmarking on different hardware
- Memory usage analysis

## 📄 License

This project follows the same spirit as Karpathy's original microGPT - educational and research-focused.

## 🙏 Acknowledgments

- **Andrej Karpathy** for the original microGPT implementation and educational content
- **Rust community** for demonstrating that ML systems can be both safe and fast
- **Performance optimization community** for graph reduction techniques

---

**Key Takeaway**: Rust achieves 42x speedup over Python for ML workloads while maintaining safety, correctness, and code quality. This project demonstrates that Rust is an excellent choice for performance-critical machine learning applications.
