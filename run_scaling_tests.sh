#!/bin/bash
set -e

# Test A: 1K, 3K, 9K
echo "Running Scaling Test A..."
./target/release/microgpt_rust --steps 1000 > output_scale_A_1000.txt
./target/release/microgpt_rust --steps 3000 > output_scale_A_3000.txt
./target/release/microgpt_rust --steps 9000 > output_scale_A_9000.txt

# Test B: Fibonacci 1K - 21K (1, 2, 3, 5, 8, 13, 21)
echo "Running Scaling Test B..."
./target/release/microgpt_rust --steps 1000 > output_scale_B_1000.txt
./target/release/microgpt_rust --steps 2000 > output_scale_B_2000.txt
./target/release/microgpt_rust --steps 3000 > output_scale_B_3000.txt
./target/release/microgpt_rust --steps 5000 > output_scale_B_5000.txt
./target/release/microgpt_rust --steps 8000 > output_scale_B_8000.txt
./target/release/microgpt_rust --steps 13000 > output_scale_B_13000.txt
./target/release/microgpt_rust --steps 21000 > output_scale_B_21000.txt

echo "All tests complete."
