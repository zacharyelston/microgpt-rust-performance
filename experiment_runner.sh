#!/bin/bash

echo "=== MicroGPT Loss Reduction Experiment ==="
echo "Target: Loss < 1.9"
echo "Current baseline: ~2.34"
echo

# Create results directory
mkdir -p experiment_results
cd experiment_results

# Initialize results file
echo "timestamp,lr,emb,layer,head,batch,ctx,final_loss,time_seconds,steps_used" > loss_reduction_results.csv

# Experiment configurations
LR_OPTIONS=(0.005 0.01 0.02 0.03)
EMB_OPTIONS=(16 32 48 64)
LAYER_OPTIONS=(1 2)
HEAD_OPTIONS=(4 8)
BATCH_OPTIONS=(4 8 16)
CTX_OPTIONS=(16 32)

echo "Starting systematic hyperparameter search..."
echo "Total experiments: ${#LR_OPTIONS[*]} * ${#EMB_OPTIONS[*]} * ${#LAYER_OPTIONS[*]} * ${#HEAD_OPTIONS[*]} * ${#BATCH_OPTIONS[*]} = $(( ${#LR_OPTIONS[*]} * ${#EMB_OPTIONS[*]} * ${#LAYER_OPTIONS[*]} * ${#HEAD_OPTIONS[*]} * ${#BATCH_OPTIONS[*]} ))"
echo

# Counter for progress
total_experiments=$(( ${#LR_OPTIONS[*]} * ${#EMB_OPTIONS[*]} * ${#LAYER_OPTIONS[*]} * ${#HEAD_OPTIONS[*]} * ${#BATCH_OPTIONS[*]} ))
current_experiment=0

# Grid search
for lr in "${LR_OPTIONS[@]}"; do
  for emb in "${EMB_OPTIONS[@]}"; do
    for layer in "${LAYER_OPTIONS[@]}"; do
      for head in "${HEAD_OPTIONS[@]}"; do
        for batch in "${BATCH_OPTIONS[@]}"; do
          # Skip invalid combinations (heads must divide embedding)
          if (( emb % head != 0 )); then
            continue
          fi
          
          current_experiment=$((current_experiment + 1))
          echo "[$current_experiment/$total_experiments] Testing: lr=$lr, emb=$emb, layer=$layer, head=$head, batch=$batch"
          
          # Run experiment
          start_time=$(date +%s.%N)
          
          # Capture output
          output=$(cd .. && cargo run --release -- -s 500 -l $lr -e $emb -y $layer -h $head -b $batch -c 16 2>/dev/null)
          
          end_time=$(date +%s.%N)
          duration=$(echo "$end_time - $start_time" | bc)
          
          # Extract results
          if echo "$output" | grep -q "Early Stopped"; then
            final_loss=$(echo "$output" | grep -A1 "Early Stopped" | grep "Best loss" | grep -o "loss [0-9.]*" | cut -d' ' -f2)
            steps_used=$(echo "$output" | grep -A1 "Early Stopped" | grep "Best loss" | grep -o "step [0-9]*" | cut -d' ' -f2)
            early_stop=1
          else
            final_loss=$(echo "$output" | grep "step.*loss" | tail -1 | grep -o "loss [0-9.]*" | cut -d' ' -f2)
            steps_used=$(echo "$output" | grep -c "step.*loss")
            early_stop=0
          fi
          
          # Log results
          timestamp=$(date +"%Y-%m-%d %H:%M:%S")
          echo "$timestamp,$lr,$emb,$layer,$head,$batch,16,$final_loss,$duration,$steps_used" >> loss_reduction_results.csv
          
          # Check if target achieved
          if (( $(echo "$final_loss < 1.9" | bc -l) )); then
            echo "🎯 TARGET ACHIEVED! Loss: $final_loss"
            echo "Configuration: lr=$lr, emb=$emb, layer=$layer, head=$head, batch=$batch"
            echo "Time: ${duration}s, Steps: $steps_used"
            echo
            echo "=== TARGET ACHIEVED ===" >> target_achievements.log
            echo "Loss: $final_loss" >> target_achievements.log
            echo "Config: lr=$lr, emb=$emb, layer=$layer, head=$head, batch=$batch" >> target_achievements.log
            echo "Time: ${duration}s, Steps: $steps_used" >> target_achievements.log
            echo "Timestamp: $timestamp" >> target_achievements.log
            echo "" >> target_achievements.log
          fi
          
          # Show progress
          echo "  → Loss: $final_loss, Time: ${duration}s, Steps: $steps_used"
          
          # Quick check for very bad results
          if (( $(echo "$final_loss > 4.0" | bc -l) )); then
            echo "  ⚠️  High loss, continuing..."
          fi
          
        done
      done
    done
  done
done

echo
echo "=== Experiment Complete ==="
echo "Results saved to: experiment_results/loss_reduction_results.csv"

# Show best results
echo
echo "=== Top 10 Results ==="
sort -t',' -k8 -n loss_reduction_results.csv | head -11 | column -t -s','

# Check if target was achieved
if [ -f "target_achievements.log" ]; then
  echo
  echo "🎉 TARGET < 1.9 ACHIEVED!"
  echo "See target_achievements.log for details:"
  cat target_achievements.log
else
  echo
  echo "❌ Target < 1.9 not achieved"
  echo "Best loss achieved:"
  best_loss=$(sort -t',' -k8 -n loss_reduction_results.csv | head -2 | tail -1 | cut -d',' -f8)
  echo "Loss: $best_loss"
fi

echo
echo "=== Analysis Complete ==="
