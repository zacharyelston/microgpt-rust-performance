# Scaling & Parameter Sweep Guide

## Overview
This branch (`artistic-symmetry`) contains tools to analyze how MicroGPT scales with **training steps** and **model size**.

## 1. Running Tests

### Step Scaling (Time)
Run `run_scaling_tests.sh` to train models for 1k, 3k, 9k, ... 21k steps.
```bash
./run_scaling_tests.sh
```

### Parameter Scaling (Size)
Run `run_param_scaling.sh` to train Small, Medium, and Large models.
```bash
./run_param_scaling.sh
```

## 2. Analyzing Results

### Basic Tables
View simple tables of Loss vs Steps/Params:
```bash
python3 analyze_scaling.py
python3 analyze_param_scaling.py
```

### Efficiency & Diminishing Returns
Calculate the "Return on Investment" (Loss drop per 1,000 units):
```bash
python3 analyze_efficiency.py
```
**Interpretation:**
- **Positive Return:** The model is learning efficiently.
- **Near Zero:** Diminishing returns; further training/scaling is wasteful.
- **Negative:** The model is degrading (overfitting or instability).

## 3. CLI Reference
You can manually run specific configurations using the updated CLI:
```bash
# Example: Custom context length and embedding size
./target/release/microgpt_rust --steps 5000 --ctx 64 --emb 128 --head 8 --layer 4
```
