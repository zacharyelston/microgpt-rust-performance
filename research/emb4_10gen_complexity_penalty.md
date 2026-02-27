# Emb:4 with Complexity Penalty — 10 Generation Run

## Experiment

```
Engine: v2 (species-aware, championship breeding, growth mutations, loser blacklist)
Embedding: 4 (fixed)
Search space: Head:1-4, Lay:1-7, Ctx:2-7, FF:1-7
Fitness: loss + 2% * ln(energy)
Energy: params * steps (rough training FLOPs)
Population: 8, Generations: 10
Total time: ~30 seconds
```

## Generation-by-Generation

```
Gen  Loss    Fitness  Species  Event
───  ──────  ───────  ───────  ─────
  1  1.8311  2.0728   8/8      Initial random population
  2  1.8311  2.0728   8/8      Elite holds, stagnation 1
  3  1.6249  1.8911   8/8      Immigrant breaks through
  4  1.5729  1.8391   6/8      Mutant improves, best ever
  5  1.5729  1.8391   6/8      Elite holds, stagnation 1
  6  1.5729  1.8391   7/8      Stagnation 2 → CHAMPIONSHIP
  7  1.6473  1.9135   4/8      Re-eval hurt, [grown] at #8
  8  1.6473  1.9135   7/8      Stagnation 2 → CHAMPIONSHIP
  9  1.5937  1.8768   8/8      Recovery, diversity restored
 10  1.5962  1.8495   8/8      Lean mutant climbs to #2
```

## Complexity Penalty in Action

The two best organisms show the penalty working:

```
#1: Lay:5 FF:1 Steps:1000  E:604K  Loss:1.5729  Fit:1.8391  (deep, lean)
#2: Lay:2 FF:1 Steps:1000  E:316K  Loss:1.5962  Fit:1.8495  (shallow, cheap)
```

Organism #2 uses **half the energy** (316K vs 604K) but is only 0.01 worse
in fitness. Without the complexity penalty, #2 would rank much lower
(raw loss 1.5962 vs 1.5729). The penalty keeps lean architectures competitive.

Compare with an organism that was complex but not better:
```
#3: Lay:5 FF:6 Steps:1000  E:1.4M  Loss:1.5937  Fit:1.8768
```
Same loss neighborhood, but **2.3x the energy** of #1. Penalized to #3.

## Evolutionary Events

### Championship #1 (gen 6)
- Elite frozen at fitness 1.8391 for 2 gens
- Re-evaluation in gen 7: elite got worse (1.6473 vs 1.5729)
- Growth mutation produced [grown] organism but ranked #8
- Species collapsed to 4/8 (monoculture risk)

### Championship #2 (gen 8)
- Same fitness frozen again for 2 gens
- Gen 9 recovery: diversity restored to 8/8
- Mutant at 1.5937 nearly matched the all-time best

## What the Engine Learned

### Winning traits (top 10 vs bottom 10)
```
Param         Top 10    Bottom 10   Trend
─────         ──────    ─────────   ─────
Heads           1.2        2.0      fewer is better
Layers          4.7        4.3      deeper helps
Context         4.0        3.7      slightly wider
FF Mult         2.0        3.5      leaner is better ←
LR            0.020      0.011      higher LR needed
Steps          1000        533      more training
```

The complexity penalty's biggest effect: **FF multiplier**. Top configs
average FF:2.0 vs bottom's FF:3.5. The engine learned that expanding
the feed-forward network at Emb:4 wastes energy without improving loss.

### Blacklisted species (4 total)
```
4-1-5-4-1  — 5 failures, avg loss 2.79  (winner's species! high variance)
4-1-7-7-5  — 3 failures, avg loss 2.82
4-2-6-4-1  — 2 failures, avg loss 2.44
4-2-7-3-3  — 2 failures, avg loss 3.28
```

The winner's own species (4-1-5-4-1) was blacklisted — it scored 1.57 once
but failed 5 times. At Emb:4, the same architecture can vary wildly
depending on random weight initialization.

## Key Findings

1. **Depth over width at tiny scale**: Lay:5 with FF:1 beat Lay:1 with FF:7.
   When embedding is only 4, stacking layers compensates for narrow dimensions.

2. **Complexity penalty steered toward FF:1**: The engine would have converged
   on higher FF multipliers without the energy cost. The penalty made it
   find that FF:1 achieves nearly the same loss at fraction of the compute.

3. **High variance at Emb:4**: The same architecture can score 1.57 or 2.79.
   Random weight initialization dominates when the model is this small.
   This suggests re-evaluation should happen more often at tiny scales.

4. **30 seconds for a full evolution**: At Emb:4, the entire 10-generation
   run takes ~30s. This makes it feasible to run hundreds of experiments.

## Winner

```
Emb:4  Head:1  Lay:5  Ctx:4  FF:1  LR:0.0204  Steps:1000
Energy: 604,000
Loss: 1.5729
Fitness: 1.8391
Origin: [mutant] from gen 4
```
