/*
    MicroGPT Loss Evolution Engine v2

    A species-aware evolutionary engine that treats hyperparameter
    configs as organisms within species (architecture families).

    Evolutionary mechanics:
    - Species tracking: monitors architecture diversity, detects monoculture
    - Tournament selection (k=3): any organism can parent if it wins
    - Growth mutations: winners that prove themselves earn structural upgrades
      (extra layer, head, context) — the Fibonacci/polydactyl effect
    - Championship breeding: on stagnation, mates top winners with fine-tuning
    - Loser blacklist: remembers failed architectures to avoid repeating them
    - Cataclysm: on deep stagnation, expands the search space entirely
    - Panic recovery: crashed configs get MAX loss, don't kill the run

    When evolution completes, the winning genome is written to genome.json,
    transforming what the main binary becomes on its next run.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

// --- Evolution Parameters ---
const POPULATION_SIZE: usize = 8;
const NUM_GENERATIONS: usize = 100;
const TOURNAMENT_SIZE: usize = 3;
const NUM_IMMIGRANTS: usize = 2;
const TARGET_LOSS: f64 = 1.2;
const STAGNATION_CHAMPIONSHIP: usize = 2; // Fine-tune the winner
const STAGNATION_CATACLYSM: usize = 4;    // Blow up and start wider
const LOSER_THRESHOLD: f64 = 2.3;         // Architectures with loss above this get blacklisted
const LOSER_MIN_SAMPLES: usize = 2;       // Need this many bad samples to blacklist
const COMPLEXITY_WEIGHT: f64 = 0.02;      // Penalty per log-unit of energy cost
const INPUT_FILE: &str = "input.txt";

#[derive(Clone, Debug)]
struct Genome {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    n_ctx: usize,
    n_ff_exp: usize,
    lr: f64,
    steps: usize,
    loss: f64,
    evaluated: bool,
    origin: String,
}

impl Genome {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let n_emb = *[2, 3, 4, 5, 6].choose(&mut rng).unwrap();
        let n_head = *[1, 2, 4].choose(&mut rng).unwrap();
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(1..=7),
            n_ctx: *[2, 3, 4, 5, 6, 7].choose(&mut rng).unwrap(),
            n_ff_exp: rng.gen_range(1..=7),
            lr: 10f64.powf(rng.gen_range(-3.0..-1.3)),
            steps: *[200, 300, 500, 750, 1000, 1500].choose(&mut rng).unwrap(),
            loss: f64::MAX,
            evaluated: false,
            origin: "random".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn new_random_wide() -> Self {
        let mut rng = rand::thread_rng();
        let n_emb = *[2, 3, 4, 5, 6].choose(&mut rng).unwrap();
        let n_head = *[1, 2, 4].choose(&mut rng).unwrap();
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(1..=7),
            n_ctx: *[2, 3, 4, 5, 6, 7].choose(&mut rng).unwrap(),
            n_ff_exp: rng.gen_range(1..=7),
            lr: 10f64.powf(rng.gen_range(-4.0..-1.0)),
            steps: *[300, 500, 1000, 1500, 2000, 3000].choose(&mut rng).unwrap(),
            loss: f64::MAX,
            evaluated: false,
            origin: "cataclysm".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let num_mutations = rng.gen_range(1..=3);
        for _ in 0..num_mutations {
            match rng.gen_range(0..7) {
                0 => self.n_emb = *[2, 3, 4, 5, 6].choose(&mut rng).unwrap(),
                1 => self.n_head = *[1, 2, 4].choose(&mut rng).unwrap(),
                2 => self.n_layer = rng.gen_range(1..=7),
                3 => self.lr = 10f64.powf(rng.gen_range(-3.0..-1.3)),
                4 => {
                    let delta = *[-500, -250, -100, 100, 250, 500].choose(&mut rng).unwrap();
                    self.steps = (self.steps as i32 + delta).clamp(100, 2000) as usize;
                },
                5 => self.n_ctx = *[2, 3, 4, 5, 6, 7].choose(&mut rng).unwrap(),
                6 => self.n_ff_exp = rng.gen_range(1..=7),
                _ => {},
            }
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    // Fine-tune mutation: small adjustments to LR and steps only.
    // Used during championship breeding to explore near the winner.
    fn fine_tune(&mut self) {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..3) {
            0 => {
                // Nudge LR by ±10-30%
                let factor = rng.gen_range(0.7..1.3);
                self.lr = (self.lr * factor).clamp(0.0001, 0.05);
            }
            1 => {
                // Nudge steps by ±50-200
                let delta = rng.gen_range(-200..=200);
                self.steps = (self.steps as i32 + delta).clamp(100, 3000) as usize;
            }
            2 => {
                // Nudge both
                let factor = rng.gen_range(0.8..1.2);
                self.lr = (self.lr * factor).clamp(0.0001, 0.05);
                let delta = rng.gen_range(-100..=100);
                self.steps = (self.steps as i32 + delta).clamp(100, 3000) as usize;
            }
            _ => {}
        }
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    // Growth mutation: the Fibonacci/polydactyl effect.
    // A proven winner earns a structural upgrade — an extra layer, head,
    // context expansion, or FF expansion. The organism literally grows.
    fn grow(&mut self) {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..4) {
            0 => {
                // Add a layer (the organism deepens)
                self.n_layer += 1;
            }
            1 => {
                // Double the heads (more parallel attention)
                let new_heads = self.n_head * 2;
                if self.n_emb % new_heads == 0 {
                    self.n_head = new_heads;
                } else {
                    self.n_layer += 1; // fallback: grow deeper instead
                }
            }
            2 => {
                // Expand context window (one step up, capped at 7)
                if self.n_ctx < 7 {
                    self.n_ctx += 1;
                }
            }
            3 => {
                // Expand feed-forward (wider MLP)
                self.n_ff_exp += 1;
            }
            _ => {}
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            let valid: Vec<usize> = [1, 2, 4, 8].iter().copied()
                .filter(|h| self.n_emb % h == 0)
                .collect();
            self.n_head = *valid.last().unwrap_or(&1);
        }
    }

    fn evaluate(&mut self, id: usize) {
        if self.evaluated {
            eprintln!("[eval] organism {} already evaluated (loss={:.4})", id, self.loss);
            return;
        }
        eprintln!("[eval] organism {} starting: {}", id, self.desc());
        let start = Instant::now();
        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: self.steps,
            input_file: INPUT_FILE.to_string(),
            gen_samples: 1,
            ..Default::default()
        };
        let result = std::panic::catch_unwind(|| {
            train_and_generate(&config, true)
        });
        match result {
            Ok(r) => {
                self.loss = r.final_loss;
                self.evaluated = true;
                eprintln!("[eval] organism {} done: loss={:.4} ({:.1}s)", id, self.loss, start.elapsed().as_secs_f64());
            }
            Err(e) => {
                eprintln!("[eval] organism {} PANICKED: {:?} | config: {}", id, e, self.desc());
                self.loss = f64::MAX;
                self.evaluated = true;
            }
        }
    }

    // Rough estimate of training energy: parameter count * steps.
    // Prefers organisms that achieve good loss with less compute.
    fn energy(&self) -> f64 {
        let emb = self.n_emb as f64;
        let params = 27.0 * emb                                  // token embeddings
            + self.n_ctx as f64 * emb                             // position embeddings
            + self.n_layer as f64 * (4.0 * emb * emb              // attention (QKV + out)
                + 2.0 * emb * emb * self.n_ff_exp as f64);       // feed-forward
        params * self.steps as f64
    }

    fn fitness(&self) -> f64 {
        if self.loss >= f64::MAX / 2.0 { return f64::MAX; }
        self.loss + COMPLEXITY_WEIGHT * self.energy().ln()
    }

    fn desc(&self) -> String {
        format!("Emb:{:<3} Head:{} Lay:{} Ctx:{:<2} FF:{} LR:{:.4} Steps:{:<4} E:{:.0}",
            self.n_emb, self.n_head, self.n_layer, self.n_ctx, self.n_ff_exp, self.lr, self.steps, self.energy())
    }

    fn species(&self) -> String {
        format!("{}-{}-{}-{}-{}", self.n_emb, self.n_head, self.n_layer, self.n_ctx, self.n_ff_exp)
    }

    fn to_config(&self, gen_samples: usize) -> TrainingConfig {
        TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: self.steps,
            input_file: INPUT_FILE.to_string(),
            gen_samples,
            ..Default::default()
        }
    }
}

// --- Genetic Operators ---

fn crossover(a: &Genome, b: &Genome) -> Genome {
    let mut rng = rand::thread_rng();
    let mut child = Genome {
        n_emb: if rng.gen() { a.n_emb } else { b.n_emb },
        n_head: if rng.gen() { a.n_head } else { b.n_head },
        n_layer: if rng.gen() { a.n_layer } else { b.n_layer },
        n_ctx: if rng.gen() { a.n_ctx } else { b.n_ctx },
        n_ff_exp: if rng.gen() { a.n_ff_exp } else { b.n_ff_exp },
        lr: if rng.gen() { a.lr } else { b.lr },
        steps: if rng.gen() { a.steps } else { b.steps },
        loss: f64::MAX,
        evaluated: false,
        origin: "cross".to_string(),
    };
    child.enforce_constraints();
    child
}

fn tournament_select<'a>(pop: &'a [Genome], rng: &mut ThreadRng) -> &'a Genome {
    let mut best: Option<&Genome> = None;
    for _ in 0..TOURNAMENT_SIZE {
        let candidate = &pop[rng.gen_range(0..pop.len())];
        if best.is_none() || candidate.fitness() < best.unwrap().fitness() {
            best = Some(candidate);
        }
    }
    best.unwrap()
}

// --- Species & Diversity Analysis ---

fn species_census(pop: &[Genome]) -> HashMap<String, Vec<usize>> {
    let mut species: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, g) in pop.iter().enumerate() {
        species.entry(g.species()).or_default().push(i);
    }
    species
}

// --- Loser Blacklist ---
// Tracks architectures that consistently produce bad results.
// If we've seen a species fail multiple times, don't waste evals on it.

struct Blacklist {
    failures: HashMap<String, Vec<f64>>,
}

impl Blacklist {
    fn new() -> Self { Blacklist { failures: HashMap::new() } }

    fn record(&mut self, genome: &Genome) {
        if genome.loss > LOSER_THRESHOLD && genome.loss < f64::MAX {
            self.failures.entry(genome.species()).or_default().push(genome.loss);
        }
    }

    fn is_blacklisted(&self, species: &str) -> bool {
        if let Some(losses) = self.failures.get(species) {
            losses.len() >= LOSER_MIN_SAMPLES
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.failures.values().filter(|v| v.len() >= LOSER_MIN_SAMPLES).count()
    }

    // Generate a random organism that isn't from a blacklisted species.
    // Gives up after 20 tries and returns whatever it got.
    fn random_avoiding(&self) -> Genome {
        for _ in 0..20 {
            let g = Genome::new_random();
            if !self.is_blacklisted(&g.species()) {
                return g;
            }
        }
        Genome::new_random()
    }

    fn random_wide_avoiding(&self) -> Genome {
        for _ in 0..20 {
            let g = Genome::new_random_wide();
            if !self.is_blacklisted(&g.species()) {
                return g;
            }
        }
        Genome::new_random_wide()
    }
}

// --- Experiment Logging ---

#[derive(Clone, Debug)]
struct HistoryEntry {
    gen: usize,
    genome: Genome,
}

macro_rules! log {
    ($log:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        println!("{}", msg);
        if let Some(ref f) = *$log.lock().unwrap() {
            let _ = writeln!(f.try_clone().unwrap(), "{}", msg);
        }
    }};
}

fn experiment_filename() -> String {
    let now = chrono::Local::now();
    format!("experiments/evolve_{}.log", now.format("%Y%m%d_%H%M%S"))
}

// --- Main Evolution Loop ---

fn main() {
    std::fs::create_dir_all("experiments").ok();
    let log_path = experiment_filename();
    let log_file: Mutex<Option<std::fs::File>> = Mutex::new(
        std::fs::File::create(&log_path).ok()
    );

    log!(log_file, "=== MicroGPT Loss Evolution Engine v2 ===");
    log!(log_file, "Experiment: {}", log_path);
    log!(log_file, "Target: loss < {:.1}", TARGET_LOSS);
    log!(log_file, "Population: {}, Generations: {}", POPULATION_SIZE, NUM_GENERATIONS);
    log!(log_file, "Selection: tournament(k={}), {} immigrants/gen", TOURNAMENT_SIZE, NUM_IMMIGRANTS);
    log!(log_file, "Stagnation: championship@{}, cataclysm@{}", STAGNATION_CHAMPIONSHIP, STAGNATION_CATACLYSM);
    log!(log_file, "Blacklist: loss > {:.1} after {} samples", LOSER_THRESHOLD, LOSER_MIN_SAMPLES);
    log!(log_file, "Complexity penalty: {:.0}% per log-unit energy", COMPLEXITY_WEIGHT * 100.0);
    log!(log_file, "");

    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    let mut population: Vec<Genome> = (0..POPULATION_SIZE).map(|_| Genome::new_random()).collect();
    let mut best_ever = Genome::new_random();
    best_ever.loss = f64::MAX;
    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut gen_bests: Vec<(usize, f64, f64, f64)> = Vec::new();
    let total_start = Instant::now();
    let mut target_gen: Option<usize> = None;
    let mut stagnation_count: usize = 0;
    let mut prev_best_fitness: f64 = f64::MAX;
    let mut blacklist = Blacklist::new();

    for gen in 0..NUM_GENERATIONS {
        let gen_start = Instant::now();
        log!(log_file, "--- Generation {}/{} ---", gen + 1, NUM_GENERATIONS);

        eprintln!("[gen {}] evaluating {} organisms...", gen + 1, population.len());
        population.par_iter_mut().enumerate().for_each(|(i, genome)| {
            genome.evaluate(i + 1);
        });

        population.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

        // Record losers in the blacklist
        for g in &population {
            blacklist.record(g);
        }

        // Species census: count architecture families
        let census = species_census(&population);
        let num_species = census.len();
        let largest_species = census.values().map(|v| v.len()).max().unwrap_or(0);
        let monoculture = largest_species as f64 / POPULATION_SIZE as f64;

        for (i, g) in population.iter().enumerate() {
            let marker = if i == 0 { ">" } else { " " };
            log!(log_file, "{} #{}: {} | Loss: {:.4} Fit: {:.4} [{}]", marker, i + 1, g.desc(), g.loss, g.fitness(), g.origin);
            history.push(HistoryEntry { gen: gen + 1, genome: g.clone() });
        }

        let gen_best = &population[0];
        let gen_worst = &population[population.len() - 1];
        if gen_best.fitness() < best_ever.fitness() {
            best_ever = gen_best.clone();
        }
        let diversity = num_species as f64 / POPULATION_SIZE as f64;
        gen_bests.push((gen + 1, gen_best.loss, gen_best.fitness(), diversity));

        // Track stagnation (based on fitness, not raw loss)
        if (gen_best.fitness() - prev_best_fitness).abs() < 1e-8 {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_best_fitness = gen_best.fitness();

        if target_gen.is_none() && best_ever.loss < TARGET_LOSS {
            target_gen = Some(gen + 1);
            log!(log_file, "  ** Target {:.1} reached! Continuing to evolve... **", TARGET_LOSS);
        }

        let spread = gen_worst.fitness() - gen_best.fitness();
        let elapsed = gen_start.elapsed().as_secs_f64();
        log!(log_file, "  Best: {:.4} (fit {:.4}) | Worst: {:.4} | Spread: {:.4} | Species: {} ({:.0}% largest) | Stagnation: {} | Blacklisted: {} | {:.0}s\n",
            gen_best.loss, gen_best.fitness(), gen_worst.loss, spread, num_species, monoculture * 100.0,
            stagnation_count, blacklist.len(), elapsed);

        // --- Breed the next generation ---
        if gen < NUM_GENERATIONS - 1 {
            if stagnation_count >= STAGNATION_CATACLYSM {
                // === CATACLYSM ===
                // Deep stagnation. Force re-eval the elite and flood with wide randoms.
                log!(log_file, "  *** CATACLYSM: {} gens stagnant — expanding search space ***", stagnation_count);
                eprintln!("[gen {}] CATACLYSM triggered", gen + 1);

                let mut new_pop: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);

                let mut elite = population[0].clone();
                elite.origin = "re-eval".to_string();
                elite.evaluated = false;
                elite.loss = f64::MAX;
                new_pop.push(elite);

                while new_pop.len() < POPULATION_SIZE {
                    let mut g = blacklist.random_wide_avoiding();
                    g.origin = "cataclysm".to_string();
                    new_pop.push(g);
                }
                eprintln!("[breed] cataclysm: 1 re-eval + {} wide randoms (avoiding {} blacklisted species)",
                    POPULATION_SIZE - 1, blacklist.len());

                stagnation_count = 0;
                population = new_pop;

            } else if stagnation_count >= STAGNATION_CHAMPIONSHIP {
                // === CHAMPIONSHIP BREEDING ===
                // The winner is stuck. Instead of random exploration, take the
                // top performers and breed them together with fine-tuning.
                // Also apply growth mutations — the polydactyl/Fibonacci effect:
                // proven winners earn a structural upgrade.
                log!(log_file, "  *** CHAMPIONSHIP: {} gens stagnant — breeding winners with growth ***", stagnation_count);
                eprintln!("[gen {}] CHAMPIONSHIP breeding triggered", gen + 1);

                let mut new_pop: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);

                // Re-evaluate the elite (remove frozen advantage)
                let mut elite = population[0].clone();
                elite.origin = "re-eval".to_string();
                elite.evaluated = false;
                elite.loss = f64::MAX;
                new_pop.push(elite);
                eprintln!("[breed] elite forced re-eval: {}", population[0].desc());

                // Growth mutation: the winner earns a structural upgrade
                let mut grown = population[0].clone();
                grown.grow();
                grown.origin = "grown".to_string();
                new_pop.push(grown.clone());
                eprintln!("[breed] growth mutation: {}", grown.desc());

                // Mate the top 3 winners together (championship crossover)
                let top3: Vec<&Genome> = population.iter().take(3).collect();
                for i in 0..top3.len() {
                    if new_pop.len() >= POPULATION_SIZE { break; }
                    let j = (i + 1) % top3.len();
                    let mut child = crossover(top3[i], top3[j]);
                    child.fine_tune();
                    child.origin = "champion".to_string();
                    new_pop.push(child);
                    eprintln!("[breed] champion cross: #{} x #{}", i + 1, j + 1);
                }

                // Fine-tune variants of the winner
                while new_pop.len() < POPULATION_SIZE - 1 {
                    let mut variant = population[0].clone();
                    variant.fine_tune();
                    variant.origin = "tuned".to_string();
                    new_pop.push(variant);
                }

                // One immigrant to maintain some diversity
                if new_pop.len() < POPULATION_SIZE {
                    let mut immigrant = blacklist.random_avoiding();
                    immigrant.origin = "immigrant".to_string();
                    new_pop.push(immigrant);
                }

                eprintln!("[breed] championship: 1 re-eval + 1 grown + {} champions + tuned + 1 immigrant = {}",
                    top3.len().min(POPULATION_SIZE), new_pop.len());
                population = new_pop;

            } else {
                // === NORMAL BREEDING ===
                eprintln!("[gen {}] breeding next generation...", gen + 1);
                let mut new_pop: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);

                let mut elite = population[0].clone();
                elite.origin = "elite".to_string();
                new_pop.push(elite);
                eprintln!("[breed] kept elite: {}", population[0].desc());

                let mut rng = rand::thread_rng();

                for i in 0..NUM_IMMIGRANTS {
                    if new_pop.len() < POPULATION_SIZE {
                        let mut immigrant = blacklist.random_avoiding();
                        immigrant.origin = "immigrant".to_string();
                        eprintln!("[breed] immigrant {}: {}", i + 1, immigrant.desc());
                        new_pop.push(immigrant);
                    }
                }

                let mut crossover_count = 0;
                let mut mutant_count = 0;
                let mut hypermutant_count = 0;

                while new_pop.len() < POPULATION_SIZE {
                    let strategy: f64 = rng.gen();
                    if strategy < 0.4 {
                        let p1 = tournament_select(&population, &mut rng);
                        let p2 = tournament_select(&population, &mut rng);
                        let mut child = crossover(p1, p2);
                        child.mutate();
                        child.origin = "cross".to_string();
                        new_pop.push(child);
                        crossover_count += 1;
                    } else if strategy < 0.8 {
                        let parent = tournament_select(&population, &mut rng);
                        let mut child = parent.clone();
                        child.mutate();
                        child.origin = "mutant".to_string();
                        new_pop.push(child);
                        mutant_count += 1;
                    } else {
                        let parent = tournament_select(&population, &mut rng);
                        let mut child = parent.clone();
                        child.mutate();
                        child.mutate();
                        child.origin = "hyper".to_string();
                        new_pop.push(child);
                        hypermutant_count += 1;
                    }
                }
                eprintln!("[breed] next gen: 1 elite + {} immigrants + {} cross + {} mutant + {} hyper = {}",
                    NUM_IMMIGRANTS, crossover_count, mutant_count, hypermutant_count, new_pop.len());
                population = new_pop;
            }
        }
    }

    // ========================================
    // Post-evolution analysis and reporting
    // ========================================

    let total_time = total_start.elapsed().as_secs_f64();
    let total_evals = history.len();

    log!(log_file, "========================================");
    log!(log_file, "       EVOLUTION COMPLETE");
    log!(log_file, "========================================");
    log!(log_file, "  Generations: {}", NUM_GENERATIONS);
    log!(log_file, "  Total evaluations: {}", total_evals);
    log!(log_file, "  Total time: {:.0}s ({:.0}s/gen avg)", total_time, total_time / NUM_GENERATIONS as f64);
    log!(log_file, "  Best loss: {:.4} (fitness: {:.4})", best_ever.loss, best_ever.fitness());
    log!(log_file, "  Best config: {}", best_ever.desc());
    log!(log_file, "  Blacklisted species: {}", blacklist.len());
    if let Some(g) = target_gen {
        log!(log_file, "  Target {:.1} first reached: generation {}", TARGET_LOSS, g);
    } else {
        log!(log_file, "  Target {:.1} NOT reached", TARGET_LOSS);
    }

    log!(log_file, "\n--- Evolution Trajectory ---");
    log!(log_file, "  {:>4}  {:>8}  {:>8}  {:>9}", "Gen", "Loss", "Fitness", "Species");
    for (gen, loss, fitness, div) in &gen_bests {
        let bar_len = ((4.0 - fitness) * 12.0).max(0.0).min(40.0) as usize;
        let bar: String = "#".repeat(bar_len);
        log!(log_file, "  {:>4}  {:>8.4}  {:>8.4}  {:>8.0}%  {}", gen, loss, fitness, div * 100.0, bar);
    }

    log!(log_file, "\n--- Top Configs Across All Generations ---");
    let mut sorted_history = history.clone();
    sorted_history.sort_by(|a, b| a.genome.fitness().partial_cmp(&b.genome.fitness()).unwrap());

    let mut seen_sigs: HashSet<String> = HashSet::new();
    let mut unique_top: Vec<&HistoryEntry> = Vec::new();
    for entry in &sorted_history {
        if seen_sigs.insert(entry.genome.species()) {
            unique_top.push(entry);
            if unique_top.len() >= 10 { break; }
        }
    }

    for (i, entry) in unique_top.iter().enumerate() {
        log!(log_file, "  {:2}. Gen {} | Loss {:.4} Fit {:.4} | {} [{}]",
            i + 1, entry.gen, entry.genome.loss, entry.genome.fitness(), entry.genome.desc(), entry.genome.origin);
    }

    log!(log_file, "\n--- Hyperparameter Analysis ---");

    let top_n = std::cmp::min(10, sorted_history.len());
    let top_configs: Vec<&Genome> = sorted_history.iter().take(top_n).map(|e| &e.genome).collect();
    let bot_configs: Vec<&Genome> = sorted_history.iter().rev().take(top_n).map(|e| &e.genome).collect();

    fn avg_f(genomes: &[&Genome], f: fn(&Genome) -> f64) -> f64 {
        genomes.iter().map(|g| f(g)).sum::<f64>() / genomes.len() as f64
    }

    log!(log_file, "  {:12} {:>10} {:>10} {:>10}", "Param", "Top 10", "Bottom 10", "Delta");

    let params: Vec<(&str, fn(&Genome) -> f64)> = vec![
        ("Embedding", |g: &Genome| g.n_emb as f64),
        ("Heads", |g: &Genome| g.n_head as f64),
        ("Layers", |g: &Genome| g.n_layer as f64),
        ("Context", |g: &Genome| g.n_ctx as f64),
        ("FF Mult", |g: &Genome| g.n_ff_exp as f64),
        ("Learn Rate", |g: &Genome| g.lr),
        ("Steps", |g: &Genome| g.steps as f64),
    ];

    for (name, f) in &params {
        let top_avg = avg_f(&top_configs, *f);
        let bot_avg = avg_f(&bot_configs, *f);
        let delta = top_avg - bot_avg;
        let arrow = if delta.abs() < 0.01 { "  " }
            else if delta > 0.0 { " ^" }
            else { " v" };
        if *name == "Learn Rate" {
            log!(log_file, "  {:12} {:>10.4} {:>10.4} {:>+9.4}{}", name, top_avg, bot_avg, delta, arrow);
        } else {
            log!(log_file, "  {:12} {:>10.1} {:>10.1} {:>+9.1}{}", name, top_avg, bot_avg, delta, arrow);
        }
    }

    log!(log_file, "\n--- Insights ---");
    let mut insights = Vec::new();

    let top_layer = avg_f(&top_configs, |g| g.n_layer as f64);
    let bot_layer = avg_f(&bot_configs, |g| g.n_layer as f64);
    if top_layer > bot_layer + 0.3 {
        insights.push(format!("Depth matters: top configs avg {:.1} layers vs {:.1} in bottom", top_layer, bot_layer));
    }

    let top_emb = avg_f(&top_configs, |g| g.n_emb as f64);
    let bot_emb = avg_f(&bot_configs, |g| g.n_emb as f64);
    if (top_emb - bot_emb).abs() > 3.0 {
        let dir = if top_emb > bot_emb { "larger" } else { "smaller" };
        insights.push(format!("Embedding size: {} is better (top avg {:.0} vs bottom {:.0})", dir, top_emb, bot_emb));
    }

    let top_steps = avg_f(&top_configs, |g| g.steps as f64);
    let bot_steps = avg_f(&bot_configs, |g| g.steps as f64);
    if (top_steps - bot_steps).abs() > 100.0 {
        let dir = if top_steps > bot_steps { "more" } else { "fewer" };
        insights.push(format!("Training duration: {} steps preferred (top avg {:.0} vs bottom {:.0})", dir, top_steps, bot_steps));
    }

    let top_lr = avg_f(&top_configs, |g| g.lr);
    let bot_lr = avg_f(&bot_configs, |g| g.lr);
    if top_lr > bot_lr * 1.5 || top_lr < bot_lr * 0.67 {
        let dir = if top_lr > bot_lr { "higher" } else { "lower" };
        insights.push(format!("Learning rate: {} is better (top avg {:.4} vs bottom {:.4})", dir, top_lr, bot_lr));
    }

    let top_ctx = avg_f(&top_configs, |g| g.n_ctx as f64);
    let bot_ctx = avg_f(&bot_configs, |g| g.n_ctx as f64);
    if (top_ctx - bot_ctx).abs() > 2.0 {
        let dir = if top_ctx > bot_ctx { "longer" } else { "shorter" };
        insights.push(format!("Context window: {} preferred (top avg {:.0} vs bottom {:.0})", dir, top_ctx, bot_ctx));
    }

    if insights.is_empty() {
        log!(log_file, "  No strong hyperparameter trends detected (more generations may help).");
    } else {
        for insight in &insights {
            log!(log_file, "  - {}", insight);
        }
    }

    // --- Blacklist report ---
    if blacklist.len() > 0 {
        log!(log_file, "\n--- Blacklisted Species (loss > {:.1}, {}+ failures) ---", LOSER_THRESHOLD, LOSER_MIN_SAMPLES);
        let mut bl_entries: Vec<_> = blacklist.failures.iter()
            .filter(|(_, v)| v.len() >= LOSER_MIN_SAMPLES)
            .collect();
        bl_entries.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        for (species, losses) in bl_entries.iter().take(10) {
            let avg: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
            log!(log_file, "  {} — {} failures, avg loss {:.4}", species, losses.len(), avg);
        }
    }

    // --- Self-modification: write the winning genome ---
    let best_config = best_ever.to_config(10);
    let best_gen = sorted_history.iter()
        .find(|e| e.genome.loss == best_ever.loss)
        .map(|e| e.gen)
        .unwrap_or(0);

    match best_config.save_genome(best_ever.loss, best_gen) {
        Ok(_) => {
            log!(log_file, "\n--- Genome Written ---");
            log!(log_file, "  Saved to genome.json (generation {})", best_gen);
            log!(log_file, "  The organism has evolved.");
            log!(log_file, "  Run `cargo run --release` to see the new creature.");
        }
        Err(e) => {
            log!(log_file, "\n  Warning: failed to save genome: {}", e);
        }
    }

    log!(log_file, "\n--- Final Demo with Best Config ---");
    let result = train_and_generate(&best_config, false);
    log!(log_file, "Final loss: {:.4}", result.final_loss);

    log!(log_file, "\nExperiment saved to: {}", log_path);
}
