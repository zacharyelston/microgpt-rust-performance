#!/bin/bash

echo "=== MicroGPT Multi-CPU Performance Benchmark ==="
echo

# Build both versions
echo "Building sequential version..."
cargo build --release
echo "Building parallel version..."
cargo build --release --features parallel
echo

# Test configurations
STEPS="1000"
EMB="32"
HEAD="8"
LAYER="2"
CTX="32"
FF="4"

echo "Configuration: steps=$STEPS, emb=$EMB, head=$HEAD, layer=$LAYER, ctx=$CTX, ff=$FF"
echo

# Benchmark sequential
echo "=== Sequential Version ==="
time ./target/release/microgpt_rust -s $STEPS -e $EMB -h $HEAD -y $LAYER -c $CTX -f $FF
echo

# Benchmark parallel
echo "=== Parallel Version ==="
time ./target/release/microgpt_rust -s $STEPS -e $EMB -h $HEAD -y $LAYER -c $CTX -f $FF
echo

# Binary size comparison
echo "=== Binary Size Comparison ==="
echo "Sequential: $(ls -lh target/release/microgpt_rust | awk '{print $5}')"
echo "Parallel:   $(ls -lh target/release/microgpt_rust | awk '{print $5}')"
echo

echo "=== Benchmark Complete ==="
