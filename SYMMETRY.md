# The Art of Symmetry

> "Beauty is the splendor of truth." — Plato

This implementation of GPT is not just code; it is an exploration of symmetry in systems. By reducing the Transformer to its atomic components, we reveal the elegant mathematical structure underlying modern AI.

## I. The Atom: `Val`

At the heart of the system lies the **Value** (`Val`). It is the indivisible atom of our universe.
- It holds **Data** (the reality).
- It holds **Gradient** (the potential for change).
- It remembers its **History** (the provenance of its existence).

```rust
struct Val(Rc<RefCell<Node>>);
struct Node { data: f64, grad: f64, prev: Vec<(Val, f64)> }
```

Like a particle in physics, it interacts with others to form complex structures, yet it remains simple and uniform.

## II. The Algebra: Operations

We define the interactions between atoms not as distinct functions, but as fundamental operators. The symmetry here is in the **Operator Overloading**:

```rust
op!(Add, add, +);
op!(Mul, mul, *);
```

Whether adding two `Val`s, a `Val` and a reference, or a reference and a `Val`, the interaction is identical. The chain rule of calculus (`backward`) is inherent in every interaction, automatically weaving the graph of computation.

## III. The Architecture: Recursive Self-Similarity

The GPT model itself is a study in fractal symmetry.
- A **Linear** layer is a transformation of space.
- **Attention** is the mechanism of relating different points in time.
- **MLP** is the mechanism of processing information at a single point in time.

The `forward` pass is a single, uninterrupted flow of tensors, mirroring the flow of thought.

## IV. The Loop: Learning

The training loop represents the cycle of improvement.
1.  **Forward**: Experience the world (compute loss).
2.  **Zero**: Clear the mind (reset gradients).
3.  **Backward**: Reflect on mistakes (backpropagation).
4.  **Update**: Change oneself (optimization).

## Minimalism

- **Lines of Code**: ~240
- **Dependencies**: `rand` (only)
- **Files**: 1 (`src/main.rs`)

This is MicroGPT in its purest form.
