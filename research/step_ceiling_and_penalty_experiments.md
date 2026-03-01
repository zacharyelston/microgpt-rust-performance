# Step Ceiling & Complexity Penalty Experiments

## Hypothesis

Two potential bottlenecks were identified in the evolution engine:
1. **Step ceiling too low** — The winning config (Steps:1782) was near the maximum allowed (1500 normal, 2000 cataclysm). Raising the ceiling might unlock lower loss.
2. **Complexity penalty too high** — At 2%, the penalty may be over-penalizing larger models that could achieve meaningfully lower loss.

## Baseline

The best result prior to these experiments:

```
Run:      Emb:2-6, 100 generations, penalty 2%, step cap 1500
Winner:   Emb:3  Head:1  Lay:1  Ctx:3  FF:2  LR:0.0354  Steps:1782
Loss:     1.3779 (record)
Fitness:  1.6293
Energy:   289K
Origin:   [re-eval] gen 99
```

## Experiment 1: Higher Step Ceiling

```
Change:   Step cap raised from 1500/2000 to 4000/5000
Penalty:  2% (unchanged)
Emb:      2-6
Gens:     100
Time:     705s (7s/gen avg)
```

### Result

```
Winner:   Emb:5  Head:1  Lay:1  Ctx:3  FF:1  LR:0.0143  Steps:940
Loss:     1.4072
Fitness:  1.6582
Energy:   282K
Origin:   [cross] gen 59
```

### Top 10 Step Distribution

```
Steps:  940 appeared in 8/10 top configs
Range:  939-1000
```

The engine converged to Steps:940 — well below both the old ceiling (1500) and the new one (5000). It actually chose *fewer* steps than the baseline winner (940 vs 1782).

### Interpretation

The step ceiling was not a bottleneck. When given room to go higher, the engine went lower. The baseline's 1782 steps may have been a local optimum specific to Emb:3, while this run's Emb:5 winner preferred a shorter, more aggressive training schedule (LR:0.014 vs LR:0.035).

Loss was 2.1% worse than baseline (1.4072 vs 1.3779). The higher step ceiling introduced more variance in the search space without providing benefit.

## Experiment 2: Lower Complexity Penalty

```
Change:   Penalty reduced from 2% to 0.5%
Step cap: 5000 (kept from experiment 1)
Emb:      2-6
Gens:     100
Time:     526s (5s/gen avg)
```

### Result

```
Winner:   Emb:4  Head:1  Lay:4  Ctx:2  FF:3  LR:0.0500  Steps:200
Loss:     1.4647
Fitness:  1.5244
Energy:   151K
Origin:   [mutant] gen 59
```

### Top 10 Analysis

```
Steps:    200 in ALL 10/10 top configs (the minimum allowed)
LR:       0.042-0.050 (near maximum, extremely aggressive)
Emb:      4-6 dominated (larger models viable with less penalty)
Layers:   2-7 (deeper architectures explored)
```

### Interpretation

With a weaker penalty, the engine shifted strategy entirely:
- **Minimum steps (200)** — train as little as possible
- **Maximum LR (0.050)** — learn as aggressively as possible
- **Larger embeddings** — Emb:5-6 appeared 8/10 times

This is a "sprint" strategy: large model, fast learning rate, minimal training. The fitness numbers look excellent (1.5244) but that's misleading — the low penalty barely counts energy cost. Raw loss was 6.3% worse than baseline (1.4647 vs 1.3779).

The 0.5% penalty was too weak to meaningfully guide the search. It allowed expensive models to compete without requiring them to actually achieve proportionally better loss.

## Comparison Table

```
Experiment              Penalty  Step Cap  Best Loss  Steps  Emb  Energy   Fitness
──────────              ───────  ────────  ─────────  ─────  ───  ──────   ───────
Baseline (record)       2.0%     1500      1.3779     1782   3    289K     1.6293
Exp 1: High steps       2.0%     5000      1.4072     940    5    282K     1.6582
Exp 2: Low penalty      0.5%     5000      1.4647     200    4    151K     1.5244
```

## Key Findings

### 1. The 2% penalty was already well-calibrated

Neither reducing nor maintaining the penalty with a wider search space improved raw loss. The 2% penalty strikes a balance: it's strong enough to kill wasteful configurations but weak enough to let genuinely better (if expensive) models compete.

### 2. Step count is architecture-dependent, not ceiling-dependent

- Emb:3 preferred 1782 steps (baseline)
- Emb:5 preferred 940 steps (experiment 1)
- Emb:4-6 preferred 200 steps with aggressive LR (experiment 2)

The optimal step count is a function of the architecture and learning rate, not the maximum allowed. Raising the ceiling didn't help because the engine wasn't constrained by it.

### 3. Weak penalties produce misleading fitness

Experiment 2's winner has the "best" fitness (1.5244) but the worst loss (1.4647). With a near-zero penalty, fitness ≈ loss, and the metric loses its ability to distinguish efficient from wasteful configurations.

### 4. The sprint strategy (low steps + high LR) has a loss floor

At Emb:4-6 with Steps:200 and LR:0.050, loss plateaus around 1.46-1.53. There isn't enough training to learn fine-grained patterns. The baseline's longer training (1782 steps) at moderate LR (0.035) achieves genuinely deeper learning.

### 5. Emb:3 remains the champion at this scale

Across all experiments, the Emb:3 Lay:1 config from the baseline holds the record. Three dimensions is enough to represent the 27-character vocabulary when paired with sufficient training duration. This is a remarkable finding: the smallest viable embedding with the simplest architecture (single layer) achieves the best loss — provided it trains long enough.

## Recommendation

Revert to the proven configuration:
- **Complexity penalty: 2%**
- **Step cap: 1500 (normal) / 2000 (cataclysm)**
- **Emb:2-6 search space**

Future improvements are more likely to come from:
- Longer evolution (200+ generations at Emb:3)
- New evolvable parameters (dropout, weight decay, LR schedule)
- Architectural innovations (skip connections, different attention patterns)
- Larger or different training datasets
