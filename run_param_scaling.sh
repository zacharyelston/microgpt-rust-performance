#!/bin/bash
set -e

STEPS=5000

# 1. Small (Base): emb=16, head=4, layer=1 (~4K params)
echo "Running Param Scaling Test: Small..."
./target/release/microgpt_rust --steps $STEPS --emb 16 --head 4 --layer 1 > output_param_small.txt

# 2. Medium: emb=32, head=4, layer=2 (~16K params)
echo "Running Param Scaling Test: Medium..."
./target/release/microgpt_rust --steps $STEPS --emb 32 --head 4 --layer 2 > output_param_medium.txt

# 3. Large: emb=64, head=8, layer=4 (~64K params)
echo "Running Param Scaling Test: Large..."
./target/release/microgpt_rust --steps $STEPS --emb 64 --head 8 --layer 4 > output_param_large.txt

echo "Param scaling tests complete."
