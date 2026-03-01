/*
    MicroGPT Hydra: Multi-Head Evolutionary Engine

    A multi-objective evolutionary system where distinct sub-populations ("Heads")
    optimize for different aesthetic criteria in parallel, then exchange genetic
    material through a central "Body".

    Heads:
    1. The Weaver (Flow): Optimizes for pronounceability and linguistic rhythm.
    2. The Mirror (Symmetry): Optimizes for structural patterns and palindromes.
    3. The Spark (Creativity): Optimizes for novelty and deviation from training data.

    Cycle:
    1. Independent Evolution: Each head runs N generations isolated.
    2. The Gathering: Heads submit their champions to the Body.
    3. Cross-Pollination: The Body breeds champions and redistributes offspring back to heads.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const POPULATION_PER_HEAD: usize = 8;
const HEAD_GENERATIONS: usize = 3;  // Gens per cycle before exchange
const CYCLES: usize = 5;            // Total synchronization cycles
const INPUT_FILE: &str = "input.txt";
const TRAIN_STEPS: usize = 300;

// --- Genome: hyperparameters as DNA ---

#[derive(Clone, Debug)]
struct Genome {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    n_ctx: usize,
    n_ff_exp: usize,
    lr: f64,
    fitness: f64,
    names: Vec<String>,
}

impl Genome {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut g = Genome {
            n_emb: rng.gen_range(1..=8),
            n_head: rng.gen_range(1..=8),
            n_layer: rng.gen_range(1..=8),
            n_ctx: rng.gen_range(1..=8),
            n_ff_exp: rng.gen_range(1..=8),
            lr: rng.gen_range(0.001..0.02),
            fitness: 0.0,
            names: Vec::new(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let choice = rng.gen_range(0..6);
        match choice {
            0 => self.n_emb = rng.gen_range(1..=8),
            1 => self.n_head = rng.gen_range(1..=8),
            2 => self.n_layer = rng.gen_range(1..=8),
            3 => self.n_ctx = rng.gen_range(1..=8),
            4 => self.n_ff_exp = rng.gen_range(1..=8),
            5 => self.lr = (self.lr * rng.gen_range(0.7..1.3)).max(0.0001).min(0.1),
            _ => {},
        }
        self.enforce_constraints();
        self.fitness = 0.0;
        self.names.clear();
    }

    // Ensure embedding dimension is divisible by number of heads
    fn enforce_constraints(&mut self) {
        // Clamp all to 1-8
        self.n_emb = self.n_emb.max(1).min(8);
        self.n_layer = self.n_layer.max(1).min(8);
        self.n_ctx = self.n_ctx.max(1).min(8);
        self.n_ff_exp = self.n_ff_exp.max(1).min(8);
        
        // n_head must be <= n_emb
        self.n_head = self.n_head.max(1).min(self.n_emb);

        // Find largest divisor of n_emb <= current n_head
        while self.n_emb % self.n_head != 0 {
            self.n_head -= 1;
        }

        // Context must be at least 2 to have input->target pairs
        self.n_ctx = self.n_ctx.max(2);
    }

    // Train and evaluate based on a specific objective
    fn evaluate(&mut self, training_data: &HashSet<String>, objective: &Objective) {
        if self.fitness != 0.0 && !self.names.is_empty() {
            return;
        }

        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: TRAIN_STEPS,
            input_file: INPUT_FILE.to_string(),
            ..Default::default()
        };

        let result = train_and_generate(&config, true);
        
        // Calculate fitness based on the specific objective of this Head
        let score = match objective {
            Objective::Weaver => calculate_flow(&result.names),
            Objective::Mirror => calculate_symmetry(&result.names),
            Objective::Spark => calculate_creativity(&result.names, training_data),
        };

        self.names = result.names;
        self.fitness = score;
    }
}

// --- Hydra Heads: Distinct Objectives ---

#[derive(Clone, Copy, Debug)]
enum Objective {
    Weaver, // Flow
    Mirror, // Symmetry
    Spark,  // Creativity
}

struct HydraHead {
    objective: Objective,
    population: Vec<Genome>,
    name: String,
}

impl HydraHead {
    fn new(objective: Objective, name: &str) -> Self {
        HydraHead {
            objective,
            population: (0..POPULATION_PER_HEAD).map(|_| Genome::new_random()).collect(),
            name: name.to_string(),
        }
    }

    fn evolve(&mut self, training_data: &HashSet<String>) {
        // Parallel evaluation of the population
        let obj = self.objective;
        self.population.par_iter_mut().for_each(|genome| {
            genome.evaluate(training_data, &obj);
        });

        // Sort by fitness
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        println!("  [{}] Top Score: {:.4} (Genome: Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.5})", 
            self.name, 
            self.population[0].fitness, 
            self.population[0].n_emb, 
            self.population[0].n_head, 
            self.population[0].n_layer,
            self.population[0].n_ctx,
            self.population[0].n_ff_exp,
            self.population[0].lr
        );
    }

    fn breed(&mut self) {
        // Simple elitism + mutation
        let elitism = 2;
        let mut new_pop = Vec::with_capacity(POPULATION_PER_HEAD);
        
        // Keep elites
        for i in 0..elitism {
            new_pop.push(self.population[i].clone());
        }

        // Breed rest
        let mut rng = rand::thread_rng();
        while new_pop.len() < POPULATION_PER_HEAD {
            let parent = &self.population[rng.gen_range(0..elitism)];
            let mut child = parent.clone();
            child.mutate();
            new_pop.push(child);
        }
        self.population = new_pop;
    }

    // Accept immigrants from other heads
    fn inject_immigrants(&mut self, immigrants: Vec<Genome>) {
        let count = immigrants.len();
        // Replace the worst performing organisms with immigrants
        let start = self.population.len() - count;
        for (i, immigrant) in immigrants.into_iter().enumerate() {
            if start + i < self.population.len() {
                // Reset fitness so it gets re-evaluated under THIS head's objective
                let mut new_genome = immigrant;
                new_genome.fitness = 0.0; 
                new_genome.names.clear();
                self.population[start + i] = new_genome;
            }
        }
    }
}

// --- Fitness Functions ---

fn calculate_flow(names: &[String]) -> f64 {
    if names.is_empty() { return -100.0; }
    let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
    
    let mut total = 0.0;
    let mut valid = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { continue; }

        let mut score = 0.0;
        let mut cons_v = 0;
        let mut cons_c = 0;

        for c in name.chars() {
            if vowels.contains(&c) {
                cons_v += 1;
                cons_c = 0;
            } else {
                cons_c += 1;
                cons_v = 0;
            }
            // Penalize clusters
            if cons_v > 2 || cons_c > 2 { score -= 1.0; }
        }
        
        // Reward ideal length
        if name.len() >= 4 && name.len() <= 8 { score += 1.0; }
        
        total += score;
        valid += 1;
    }
    
    if valid == 0 { -100.0 } else { total / valid as f64 }
}

fn calculate_symmetry(names: &[String]) -> f64 {
    if names.is_empty() { return -100.0; }
    
    let mut total = 0.0;
    let mut valid = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 { continue; }

        let mut score = 0.0;
        let chars: Vec<char> = name.chars().collect();

        // Palindrome
        if chars.iter().eq(chars.iter().rev()) { score += 3.0; }

        // Repeating halves
        if name.len() >= 4 {
            let mid = name.len() / 2;
            if name[..mid] == name[mid..mid*2] { score += 2.0; }
        }

        // Rhyme-friendly endings
        if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') { score += 0.5; }

        total += score;
        valid += 1;
    }

    if valid == 0 { -100.0 } else { total / valid as f64 }
}

fn calculate_creativity(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() { return -100.0; }
    
    let mut total = 0.0;
    let mut valid = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 { continue; }

        // Novelty is the only goal here
        if training_data.contains(&name) {
            total -= 5.0; // Memorization is death
        } else {
            total += 2.0; // Novelty is life
        }
        
        // Reward length variance (we don't want all short names)
        if name.len() > 6 { total += 0.5; }

        valid += 1;
    }

    if valid == 0 { -100.0 } else { total / valid as f64 }
}

// --- Main Hydra Loop ---

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       MicroGPT Hydra: Multi-Head Evolution       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("Heads: Weaver (Flow), Mirror (Symmetry), Spark (Creativity)");
    println!("Pop per Head: {}, Cycles: {}, Gens/Cycle: {}", POPULATION_PER_HEAD, CYCLES, HEAD_GENERATIONS);

    // Load Data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    // Initialize Heads
    let mut heads = vec![
        HydraHead::new(Objective::Weaver, "Weaver"),
        HydraHead::new(Objective::Mirror, "Mirror"),
        HydraHead::new(Objective::Spark,  "Spark"),
    ];

    for cycle in 0..CYCLES {
        println!("\n=== Cycle {}/{} ===", cycle + 1, CYCLES);
        let cycle_start = Instant::now();

        // 1. Independent Evolution
        // Use rayon's par_iter_mut to run heads in parallel if possible, 
        // but since we are already parallelizing inside `evolve`, we might just iterate sequentially over heads.
        // Actually, let's keep it sequential for heads to avoid oversubscribing threads.
        for head in &mut heads {
            println!(">> {} is thinking...", head.name);
            for gen in 0..HEAD_GENERATIONS {
                head.evolve(&training_data);
                if gen < HEAD_GENERATIONS - 1 {
                    head.breed();
                }
            }
        }

        // 2. The Gathering (Cross-Pollination)
        println!("\n>> The Gathering: Heads exchange secrets...");
        
        // Collect elites from each head
        let mut exchange_pool = Vec::new();
        for head in &heads {
            // Take top 2 from each head
            exchange_pool.push(head.population[0].clone());
            exchange_pool.push(head.population[1].clone());
        }

        // Distribute pool to all heads (as immigrants)
        for i in 0..heads.len() {
            let mut incoming = Vec::new();
            for genome in &exchange_pool {
                incoming.push(genome.clone());
            }
            heads[i].inject_immigrants(incoming);
        }

        println!(">> Cycle Complete in {:.2?}", cycle_start.elapsed());
    }

    // Final Results
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║                 Hydra Ascended                   ║");
    println!("╚══════════════════════════════════════════════════╝");

    for head in &heads {
        let best = &head.population[0];
        println!("\n[{}] Champion:", head.name);
        println!("  Genome: Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.5}", 
            best.n_emb, best.n_head, best.n_layer, best.n_ctx, best.n_ff_exp, best.lr);
        println!("  Score:  {:.4}", best.fitness);
        println!("  Names:  {}", best.names.iter().take(5).cloned().collect::<Vec<_>>().join(", "));
    }
}
