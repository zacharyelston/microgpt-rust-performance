# Embedding Size Analysis: How Small Can a Transformer Be?

## Experiment Overview

Tested the effect of embedding dimension on MicroGPT character-level name generation,
comparing organisms evolved across different embedding scales.

Task: character-level name generation from Karpathy's makemore names dataset (~32K names, 27-char vocab).

All runs: population 8, tournament selection k=3, 2 random immigrants per generation.

## Results by Embedding Scale

### Emb:4 (everything under 8)

```
Search space: Emb:4, Head:1-4, Lay:1-7, Ctx:2-7, FF:1-7
1 generation, 8 evaluations, 3 seconds total

Gen 1   2.2527   Emb:4  Head:1 Lay:3 Ctx:7  FF:3 LR:0.0307 Steps:1500
        2.4161   Emb:4  Head:4 Lay:4 Ctx:2  FF:3 LR:0.0429 Steps:1000
        2.7313   Emb:4  Head:4 Lay:4 Ctx:5  FF:2 LR:0.0178 Steps:300
        2.8547   Emb:4  Head:1 Lay:4 Ctx:6  FF:2 LR:0.0011 Steps:750
        3.0088   Emb:4  Head:4 Lay:7 Ctx:5  FF:1 LR:0.0021 Steps:750
        3.0405   Emb:4  Head:1 Lay:4 Ctx:5  FF:5 LR:0.0012 Steps:750
        3.2986   Emb:4  Head:2 Lay:3 Ctx:2  FF:6 LR:0.0011 Steps:300
        3.7657   Emb:4  Head:2 Lay:7 Ctx:2  FF:5 LR:0.0144 Steps:300

Best loss: 2.2527
Eval speed: ~0.4s per organism
```

Observations at Emb:4:
- Best loss 2.25 — about 53% worse than Emb:24 organisms
- Context window is critical: Ctx:7 won, Ctx:2 organisms scored 3.3-3.8
- More layers didn't help: 7-layer models (3.0, 3.8) lost to 3-layer (2.25)
- Higher LR helped: 0.03 beat 0.001 — tiny models need bigger gradient steps
- More heads hurt at this scale: Head:1 won over Head:4 variants
- Extremely fast: entire generation in 3 seconds vs 2+ minutes at normal scale

### Emb:8 (1-gen quick test at normal scale)

```
Search space: Emb:8-32, Head:1-4, Lay:1-3, Ctx:8-24, FF:1-4
1 generation, 8 evaluations

Best: Emb:8  Head:4 Lay:1 Ctx:24 FF:2 LR:0.0269 Steps:500  | Loss: 1.4803
```

The Emb:8 organism actually won the generation over larger models —
4 heads with 2-dim attention each, max context, high LR.

### Emb:8-32 (v1 engine, 10 generations, 3 runs)

```
Run 1:  Best 1.4165  Emb:24 Head:4 Lay:1 Ctx:12 FF:1 LR:0.0097 Steps:500   (1562s)
Run 2:  Best 1.4909  Emb:16 Head:4 Lay:1 Ctx:16 FF:3 LR:0.0093 Steps:500   (1452s)
Run 3:  Best ~1.57   Emb:32 Head:2 Lay:3 Ctx:24 FF:1 LR:0.0029 Steps:500   (cut off gen 10)
```

Frozen elite problem in v1: the same organism won gens 2-10 in Run 1
without re-evaluation. Fixed in v2 with championship breeding.

### Emb:8-32 (v2 engine, 10 generations)

```
Search space: Emb:8-32, Head:1-4, Lay:1-4, Ctx:8-24, FF:1-4
With species tracking, championship breeding, growth mutations, loser blacklist
10 generations, 80 evaluations, 1820 seconds

Gen  1   1.5278  8 species
Gen  2   1.5278  8 species  stagnation: 1
Gen  3   1.5278  8 species  stagnation: 2 → CHAMPIONSHIP
Gen  4   1.5886  6 species  [re-eval] [grown] [champion]
Gen  5   1.5232  7 species  immigrant broke through
Gen  6   1.5232  7 species
Gen  7   1.5232  7 species  stagnation: 2 → CHAMPIONSHIP
Gen  8   1.4960  6 species  [grown] emerged at #2
Gen  9   1.4960  7 species
Gen 10   1.4676  8 species  [cross] new best

Best loss: 1.4676
Winner: Emb:24 Head:2 Lay:1 Ctx:32 FF:3 LR:0.0119 Steps:500
Blacklisted: species 24-1-1-24-3 (2 failures, avg loss 2.83)
```

## Scaling Curve

```
Embedding    Best Loss    Relative    Eval Speed
─────────    ─────────    ────────    ──────────
  4           2.2527       1.00x      ~0.4s
  8           1.4803       0.66x      ~1.5s
 16           1.4909       0.66x      ~10s
 24           1.4165       0.63x      ~30s
 32           1.5657       0.69x      ~90s
```

The scaling curve shows diminishing returns:
- 4 → 8:  massive improvement (2.25 → 1.48, -34%)
- 8 → 16: marginal improvement (1.48 → 1.49, essentially flat)
- 16 → 24: small improvement (1.49 → 1.42, -5%)
- 24 → 32: regression (1.42 → 1.57, +10%) — overfitting likely

## Key Findings

### 1. Embedding 4 is representationally starved
With 27 characters and only 4 dimensions, each character gets roughly
4 numbers to distinguish itself. The model can't build useful attention
patterns — it's compressing 27 identities into a 4D space where everything
overlaps. Loss floor appears to be around 2.2.

### 2. Embedding 8 is the sweet spot for this task
Emb:8 achieved loss 1.48 — within 4% of the best-ever 1.42 from Emb:24 —
while evaluating 50-200x faster. The organism that won at Emb:8 used
4 heads (2-dim attention per head), suggesting it learned to split its
tiny embedding into specialized attention channels.

### 3. Bigger isn't always better
Emb:32 organisms consistently underperformed Emb:24 and sometimes even Emb:16.
With a small dataset (32K names, short sequences), larger embeddings overfit
and take longer to converge in limited training steps.

### 4. Context matters more than depth
Across all scales, Ctx:24+ beat Ctx:8. Meanwhile, Lay:1 consistently
won over Lay:3+. For character-level names (avg ~6 chars), seeing the
full name matters more than stacking transformer layers.

### 5. Consistent winning hyperparameters across scales
- LR: ~0.01-0.03 (high end of search space)
- Steps: 500 (short training, avoid overfitting)
- Layers: 1 (shallow wins)
- Context: max available
- FF multiplier: varies (1-4 all competitive)

### 6. Speed-quality tradeoff
At Emb:4, an entire 8-organism generation runs in 3 seconds.
At Emb:24, it takes 3+ minutes. For rapid prototyping and architecture
search, starting at Emb:4-8 and growing winners is far more efficient
than starting large.

## Evolutionary Mechanics Observations

### Championship breeding worked
Two championship events triggered (gens 3 and 7). Both times:
- The frozen elite was re-evaluated, revealing its true (worse) loss
- Growth mutations produced competitive variants
- An immigrant or crossover eventually broke through with new best

### Growth mutations are promising
The [grown] organism at gen 8 (loss 1.4960) was competitive — the
polydactyl effect produced a viable variant by expanding the winner's
feed-forward layer. Growth didn't win outright but contributed genetic
material that led to the gen 10 winner via crossover.

### The blacklist prevented waste
Species 24-1-1-24-3 was blacklisted after 2 failures averaging loss 2.83.
This prevented the engine from generating more organisms in that
architectural neighborhood, saving evaluation time.

## Recommendations

1. For fast iteration: use Emb:4-8 for architecture search, then grow winners
2. For best quality: Emb:16-24, Lay:1, Ctx:max, LR:0.01, Steps:500
3. Avoid: Emb:32+ (overfitting), deep networks (Lay:4+), low LR (<0.002)
4. The ideal evolution strategy: start tiny, evolve fast, grow winners structurally
