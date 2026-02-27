
import glob
import re

def parse_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract steps from filename
    steps_match = re.search(r'_(\d+)\.txt', filename)
    steps = int(steps_match.group(1)) if steps_match else 0
    
    # Extract final loss
    # The progress bar uses \r, so we need to split by \r or find the last occurrence
    loss_matches = re.findall(r'step\s+(\d+)\s+\|\s+loss\s+([\d\.]+)', content)
    final_loss = float(loss_matches[-1][1]) if loss_matches else 999.0
    
    # Extract generated samples
    gen_match = re.split(r'--- Generation ---', content)
    samples = []
    if len(gen_match) > 1:
        samples = [line.strip().replace('> ', '') for line in gen_match[1].strip().split('\n') if line.strip().startswith('>')]
    
    return steps, final_loss, samples

files = sorted(glob.glob('output_scale_*.txt'), key=lambda x: int(re.search(r'_(\d+)\.txt', x).group(1)))

print(f"{'Steps':<10} | {'Loss':<10} | {'Samples (First 3)'}")
print("-" * 60)

for f in files:
    steps, loss, samples = parse_file(f)
    samples_str = ", ".join(samples[:3])
    print(f"{steps:<10} | {loss:<10.4f} | {samples_str}")
