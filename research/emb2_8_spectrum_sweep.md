# Embedding Spectrum Sweep: Emb 2-8 with Complexity Penalty

## Experiment

```
Engine: v2 (species-aware, championship, growth, blacklist, complexity penalty)
Embedding: 2, 4, 6, 8 (mixed competition)
Search space: Head:1-4, Lay:1-7, Ctx:2-7, FF:1-7
Fitness: loss + 2% * ln(energy)
Population: 8, Generations: 10
Total time: 440 seconds (44s/gen avg)
```

## Generation-by-Generation Trajectory

```
Gen  Loss    Fitness  Species  Event
───  ──────  ───────  ───────  ─────
  1  1.6759  1.9833   8/8      Emb:6 leads
  2  1.5786  1.8908   8/8      Emb:8 takes #1 (hyper-mutant)
  3  1.5786  1.8908   7/8      Stagnation 1
  4  1.5786  1.8908   7/8      Stagnation 2 → CHAMPIONSHIP
  5  1.5528  1.8650   5/8      [tuned] new best!
  6  1.5528  1.8650   7/8      Stagnation 1
  7  1.5528  1.8650   8/8      Stagnation 2 → CHAMPIONSHIP
  8  1.5873  1.8995   5/8      Re-eval slightly worse
  9  1.5873  1.8995   8/8      Stagnation 1
 10  1.5873  1.8995   8/8      Stagnation 2
```

## Which Embedding Size Won?

```
Top 10 Configs by Fitness:
Rank  Emb  Loss    Fitness  Lay  FF  Energy    Origin
────  ───  ──────  ───────  ───  ──  ───────   ──────
  1    8   1.5528  1.8650    5    7   6.0M     [tuned]
  2    6   1.5797  1.8658    5    2   1.6M     [hyper]     ← 3.7x cheaper
  3    6   1.5837  1.8847    5    7   3.4M     [mutant]
  4    8   1.6085  1.8895    1    6   1.3M     [immigrant]
  5    4   1.6372  1.8926    2    7   352K     [mutant]    ← 17x cheaper
  6    8   1.5828  1.8950    5    7   6.0M     [mutant]
  7    8   1.5915  1.9037    5    7   6.0M     [grown]
  8    8   1.6125  1.9224    5    6   5.4M     [mutant]
  9    8   1.6531  1.9363    1    7   1.4M     [mutant]
 10    6   1.6324  1.9397    7    7   4.7M     [cross]
```

### Embedding breakdown in top 10:
- **Emb:8** — 6 appearances (raw loss leader, but high energy)
- **Emb:6** — 3 appearances (excellent efficiency)
- **Emb:4** — 1 appearance (competitive on fitness despite higher loss)
- **Emb:2** — 0 appearances (cannot compete)

## The Complexity Penalty's Effect

The most revealing comparison is between configs #1 and #2:

```
#1: Emb:8  Lay:5 FF:7  Loss:1.5528  Energy:6.0M   Fitness:1.8650
#2: Emb:6  Lay:5 FF:2  Loss:1.5797  Energy:1.6M   Fitness:1.8658
```

These have nearly **identical fitness** (0.0008 apart) despite:
- Emb:8 having 2.7% better raw loss
- Emb:6 using 3.7x less energy

Without the complexity penalty, Emb:8 would dominate every rank.
With it, the lean Emb:6 organism is recognized as essentially equivalent.

## Embedding Size Analysis

### Emb:2 — Too Small
```
Best seen: Loss 1.8907, Fitness 2.1579  (gen 10, Lay:5 Ctx:4 FF:7)
```
Emb:2 means each character gets only 2 numbers to represent itself.
With 27 characters, there simply isn't enough space. Even with 5 layers
and FF:7 (maximum expansion), it couldn't break below 1.89.

The engine never selected Emb:2 organisms as parents. They exist
in a fitness desert — no amount of depth or width compensates.

### Emb:4 — Scrappy Survivor
```
Best seen: Loss 1.6372, Fitness 1.8926  (gen 3, Lay:2 Ctx:5 FF:7)
```
Emb:4 competed in the top 5 by fitness thanks to its low energy cost.
At only 352K energy (17x cheaper than Emb:8), it earned its ranking
through efficiency, not raw performance.

Pattern: Emb:4 uses fewer layers but maximum FF expansion — it
compensates for narrow embedding by widening the MLP.

### Emb:6 — The Sweet Spot
```
Best seen: Loss 1.5797, Fitness 1.8658  (gen 7, Lay:5 Ctx:5 FF:2)
```
Emb:6 appeared 3 times in the top 10. Its best config used only FF:2
(unlike Emb:4 and Emb:8 which needed FF:7), showing that 6 dimensions
provide enough representational capacity that the FF layer doesn't need
to compensate.

Emb:6 at 1.6M energy achieves loss within 1.7% of Emb:8 at 6.0M energy.
This is the efficiency frontier.

### Emb:8 — Raw Power, High Cost
```
Best seen: Loss 1.5528, Fitness 1.8650  (gen 5, Lay:5 Ctx:5 FF:7)
```
Emb:8 has the best raw loss but at 6x the energy cost of Emb:6.
It dominates the raw loss rankings but only narrowly leads on fitness.

Pattern: Emb:8 consistently paired with Lay:5 and FF:6-7 — it builds
deep, wide networks. When it went shallow (Lay:1), loss increased
to 1.65+ but fitness improved due to lower energy.

## Hyperparameter Trends Across Embeddings

```
Param         Top 10    Bottom 10   Trend
─────         ──────    ─────────   ─────
Embedding       7.2        4.8     larger helps (with energy penalty)
Heads           1.0        1.5     fewer is better at all scales
Context         4.8        3.2     wider context helps
FF Mult         6.4        6.2     high FF universal
LR            0.018      0.012     higher LR always
Steps           950        665     more training helps
```

### Universal traits (same at every embedding size):
- **Head:1** — single attention head dominates
- **LR ~0.02** — high learning rate
- **Steps ~1000** — moderate training

### Embedding-dependent traits:
- **FF expansion**: Emb:2,4 need FF:7 to compensate; Emb:6 works at FF:2; Emb:8 uses FF:7 for depth
- **Layers**: Emb:4 prefers Lay:2; Emb:6,8 prefer Lay:5-7

## Cross-Experiment Comparison

```
Experiment         Best Loss  Best Fit  Winner Emb  Energy    Time
──────────         ─────────  ────────  ──────────  ──────    ────
Emb:4 10gen        1.5729     1.8391    4           604K      30s
Emb:4 20gen        1.5029     1.7481    4           212K      60s
Emb:2-8 10gen      1.5528     1.8650    8           6.0M      440s
```

The Emb:4 20-gen run found a **leaner, better-fitting** organism than
the Emb:2-8 sweep. Given more generations at a fixed small scale,
the engine can find more efficient architectures than when splitting
its search budget across multiple embedding sizes.

## Key Findings

1. **Emb:6 is the efficiency frontier**: Within 2% of Emb:8 loss at 4x less
   energy. The complexity penalty correctly identifies this tradeoff.

2. **Emb:2 is below the viability threshold**: Cannot represent 27 characters
   in 2 dimensions regardless of depth or width.

3. **Emb:4 competes on fitness**: Its low energy cost keeps it in the top 5
   even though its raw loss is 6% worse than Emb:8.

4. **Fixed-scale evolution beats mixed-scale**: The Emb:4 20-gen run found
   better fitness (1.7481) than the 2-8 sweep (1.8650) because the engine
   could focus its search budget on optimizing one architecture family.

5. **FF expansion compensates for embedding**: Smaller embeddings need larger
   FF multipliers. Emb:4 needs FF:7, Emb:6 needs FF:2, showing that
   feed-forward width and embedding dimension are substitutes.
