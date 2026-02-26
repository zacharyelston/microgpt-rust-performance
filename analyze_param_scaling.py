
import glob
import re

def parse_file(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return 0, 0, 0, 0, 999.0, []
    
    # Extract config from first line
    # MicroGPT: 16480 params, training for 5000 steps (lr=0.005, emb=32, head=4, layer=2, ctx=16, ff=4)
    params_match = re.search(r'MicroGPT: (\d+) params', content)
    params = int(params_match.group(1)) if params_match else 0
    
    emb_match = re.search(r'emb=(\d+)', content)
    emb = int(emb_match.group(1)) if emb_match else 0
    
    layer_match = re.search(r'layer=(\d+)', content)
    layer = int(layer_match.group(1)) if layer_match else 0
    
    # Extract final loss
    loss_matches = re.findall(r'step\s+(\d+)\s+\|\s+loss\s+([\d\.]+)', content)
    final_loss = float(loss_matches[-1][1]) if loss_matches else 999.0
    
    # Extract generated samples
    gen_match = re.split(r'--- Generation ---', content)
    samples = []
    if len(gen_match) > 1:
        samples = [line.strip().replace('> ', '') for line in gen_match[1].strip().split('\n') if line.strip().startswith('>')]
    
    return params, emb, layer, final_loss, samples

files = ['output_param_small.txt', 'output_param_medium.txt', 'output_param_large.txt']
labels = ['Small', 'Medium', 'Large']

print(f"{'Model':<10} | {'Params':<10} | {'Emb':<5} | {'Lay':<5} | {'Loss':<10} | {'Samples (First 3)'}")
print("-" * 80)

for label, f in zip(labels, files):
    params, emb, layer, loss, samples = parse_file(f)
    samples_str = ", ".join(samples[:3])
    print(f"{label:<10} | {params:<10} | {emb:<5} | {layer:<5} | {loss:<10.4f} | {samples_str}")
