
import glob
import re

def parse_files(pattern):
    data = []
    files = glob.glob(pattern)
    for f in files:
        with open(f, 'r') as file:
            content = file.read()
            
        # Extract steps/params from filename or content
        steps_match = re.search(r'_(\d+)\.txt', f)
        steps = int(steps_match.group(1)) if steps_match else 0
        
        params_match = re.search(r'MicroGPT: (\d+) params', content)
        params = int(params_match.group(1)) if params_match else 0
        
        # Extract final loss
        loss_matches = re.findall(r'loss\s+([\d\.]+)', content)
        if loss_matches:
            final_loss = float(loss_matches[-1])
            data.append({'file': f, 'steps': steps, 'params': params, 'loss': final_loss})
            
    return sorted(data, key=lambda x: x['steps'] if 'scale_B' in pattern else x['params'])

def analyze_diminishing_returns(data, metric_key):
    print(f"\n--- Efficiency Analysis: {metric_key.capitalize()} ---")
    print(f"{metric_key.capitalize():<10} | {'Loss':<10} | {'Delta Loss':<12} | {'Return (Loss Drop / 1k unit)':<25}")
    print("-" * 65)
    
    for i in range(len(data)):
        curr = data[i]
        val = curr[metric_key]
        loss = curr['loss']
        
        if i == 0:
            print(f"{val:<10} | {loss:<10.4f} | {'-':<12} | {'-':<25}")
        else:
            prev = data[i-1]
            delta_val = val - prev[metric_key]
            delta_loss = prev['loss'] - loss # Positive means improvement
            
            # efficiency = loss drop per 1000 units
            efficiency = (delta_loss / delta_val) * 1000 if delta_val > 0 else 0
            
            eff_str = f"{efficiency:.4f}"
            print(f"{val:<10} | {loss:<10.4f} | {delta_loss:<12.4f} | {eff_str:<25}")

# 1. Analyze Step Scaling (Resource: Compute Time)
step_data = parse_files('output_scale_B_*.txt')
if step_data:
    analyze_diminishing_returns(step_data, 'steps')

# 2. Analyze Param Scaling (Resource: Memory/Model Size)
# We need to manually map small/medium/large to params since filenames don't have numbers
param_files = {
    'output_param_small.txt': 4192,
    'output_param_medium.txt': 16480,
    'output_param_large.txt': 201088 
}
param_data = []
for fname, p_count in param_files.items():
    try:
        with open(fname, 'r') as f:
            content = f.read()
            loss_matches = re.findall(r'loss\s+([\d\.]+)', content)
            if loss_matches:
                param_data.append({'params': p_count, 'loss': float(loss_matches[-1])})
    except FileNotFoundError:
        pass
        
param_data.sort(key=lambda x: x['params'])
if param_data:
    analyze_diminishing_returns(param_data, 'params')
