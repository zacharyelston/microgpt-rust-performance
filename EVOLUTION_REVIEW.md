# Evolution System Review (2026-02-27)

## Should we refactor?

Yes—targeted refactoring is warranted before extending features.

The current evolution binaries (`evolve.rs` and `evolve_loss.rs`) work, but they duplicate core behaviors (genome constraints, mutation/evaluation orchestration, training-data bootstrap, logging style) and mix concerns (search strategy + process control + reporting) in single files. This is increasing risk for silent logic drift between engines.

## Critical gaps

1. **No deterministic/reproducible runs (critical).**
   - `thread_rng()` is used everywhere, without a seed path or run manifest.
   - This prevents scientific comparison, debugging, and reliable regression checks.

2. **No validation harness for evolution outcomes (critical).**
   - There are no tests asserting invariants such as:
     - embedding/head divisibility is always preserved,
     - tournament selection never panics on edge population sizes,
     - blacklist/cataclysm logic preserves population size.
   - No statistical regression gate (e.g., median best-loss across N seeded runs).

3. **Evaluation objective is noisy and under-controlled (critical).**
   - Each genome is usually evaluated once, on one stochastic training trajectory.
   - Selection pressure is dominated by random variance; this can reward lucky runs over robust configs.

4. **Config and artifact management is fragile.**
   - `genome.json` is parsed with ad-hoc string splitting instead of typed serialization.
   - External data fetch via `curl` has no explicit error handling/retry policy.

5. **Performance telemetry is insufficient for optimization decisions.**
   - Runtime logs print progress, but there is no structured record of compute cost per evaluation (time/step, params/sec, memory), making multi-objective optimization difficult.

## Recommended refactor plan

1. **Extract a shared `evolution` module** for:
   - genome type + constraints,
   - mutation/crossover/select operators,
   - run context (RNG seed, dataset path, logging sink),
   - common reporting structs.

2. **Add deterministic mode**:
   - CLI seed flag (`--seed`),
   - all RNG passed explicitly (`StdRng`),
   - run metadata persisted with seed + git commit SHA.

3. **Introduce typed persistence**:
   - `serde` + `serde_json` for `genome.json` and experiment summaries,
   - version field for forward-compatible schema upgrades.

4. **Stabilize fitness estimation**:
   - evaluate each candidate over `k` repeats (or mini-ensemble of seeds),
   - select by robust statistic (median loss / lower-confidence bound),
   - optionally early-stop hopeless candidates.

5. **Add invariant and smoke tests**:
   - mutation/constraint property tests,
   - breeding cycle maintains exact population size,
   - one short deterministic evolution run in CI.

## Minimal acceptance criteria before adding new strategies

- Seeded deterministic evolution runs produce identical top genome and logs.
- Shared evolution library is consumed by both binaries.
- `genome.json` read/write roundtrip test passes.
- One CI job runs short seeded evolution and checks invariant assertions.
