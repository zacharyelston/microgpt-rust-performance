/*
    MicroGPT Aesthetic Evolution Engine

    A parallel evolutionary engine that optimizes for the *beauty*
    of generated names rather than raw loss. Treats MicroGPT
    hyperparameters as DNA and evolves them to maximize a fitness
    function based on:

      - Flow: pronounceability (vowel/consonant alternation)
      - Symmetry: palindromes, repeating sub-patterns, pleasant endings
      - Creativity: penalty for memorizing training data; reward novelty

    Uses Rayon for parallel evaluation across all CPU cores.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const POPULATION_SIZE: usize = 12;
const GENERATIONS: usize = 5;
const ELITISM: usize = 2;
const INPUT_FILE: &str = "input.txt";
const TRAIN_STEPS: usize = 300;

// --- Genome: hyperparameters as DNA ---

#[derive(Clone, Debug)]
struct AestheticCandidate {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    lr: f64,
    fitness: f64,
    names: Vec<String>,
}

impl AestheticCandidate {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut g = AestheticCandidate {
            n_emb: *[16, 24, 32].choose(&mut rng).unwrap(),
            n_head: *[2, 4].choose(&mut rng).unwrap(),
            n_layer: rng.gen_range(1..=3),
            lr: rng.gen_range(0.001..0.015),
            fitness: 0.0,
            names: Vec::new(),
        };
        g.enforce_model_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let choice = rng.gen_range(0..4);
        match choice {
            0 => self.n_emb = *[16, 24, 32, 40].choose(&mut rng).unwrap(),
            1 => self.n_head = *[2, 4].choose(&mut rng).unwrap(),
            2 => {
                self.n_layer = (self.n_layer as i32 + *[-1, 1].choose(&mut rng).unwrap())
                    .max(1)
                    .min(4) as usize
            }
            3 => self.lr = (self.lr * rng.gen_range(0.7..1.3)).max(0.0001).min(0.1),
            _ => {}
        }
        self.enforce_model_constraints();
        self.fitness = 0.0;
        self.names.clear();
    }

    // Ensure embedding dimension is divisible by number of heads
    fn enforce_model_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            self.n_head = 2;
        }
        if self.n_emb % self.n_head != 0 {
            self.n_emb = (self.n_emb / self.n_head) * self.n_head;
            if self.n_emb == 0 {
                self.n_emb = self.n_head;
            }
        }
    }

    // Train a MicroGPT and evaluate the aesthetic quality of its output
    fn evaluate_candidate(
        &mut self,
        training_data: &HashSet<String>,
        seed: Option<u64>,
        id: usize,
    ) {
        if self.fitness != 0.0 && !self.names.is_empty() {
            return;
        }

        let eval_seed = seed.map(|base| base ^ (id as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            lr: self.lr,
            steps: TRAIN_STEPS,
            input_file: INPUT_FILE.to_string(),
            seed: eval_seed,
            ..Default::default()
        };

        let result = train_and_generate(&config, true);
        let score = score_name_batch(&result.names, training_data);

        self.names = result.names;
        self.fitness = score;
    }
}

// --- Fitness: The Judge ---
// Evaluates generated names on three aesthetic dimensions.

fn score_name_batch(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() {
        return -100.0;
    }

    let mut total_score = 0.0;
    let mut valid_count = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) {
            continue;
        }

        let s_flow = score_pronounceability(&name);
        let s_sym = score_pattern_harmony(&name);
        let s_creat = score_novelty(&name, training_data);

        // Creativity weighted 2x — novelty matters most
        total_score += s_flow * 1.0 + s_sym * 1.2 + s_creat * 2.0;
        valid_count += 1;
    }

    if valid_count == 0 {
        return -100.0;
    }
    total_score / valid_count as f64
}

// Flow: penalize unpronounceable clusters (3+ consecutive vowels or consonants)
fn score_pronounceability(name: &str) -> f64 {
    let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
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
        if cons_v > 2 || cons_c > 2 {
            score -= 1.0;
        }
    }

    // Bonus for ideal name length (4-8 characters)
    if name.len() >= 4 && name.len() <= 8 {
        score += 0.5;
    }
    score
}

// Symmetry: reward palindromes, repeating halves, pleasant endings
fn score_pattern_harmony(name: &str) -> f64 {
    let mut score = 0.0;
    let chars: Vec<char> = name.chars().collect();

    // Perfect palindrome
    if name.len() > 3 && chars.iter().eq(chars.iter().rev()) {
        score += 2.0;
    }

    // Repeating first half (e.g., "mama")
    if name.len() >= 4 {
        let mid = name.len() / 2;
        if name[..mid] == name[mid..mid * 2] {
            score += 1.5;
        }
    }

    // Pleasant ending sounds
    if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') {
        score += 0.2;
    }
    score
}

// Creativity: heavy penalty for memorizing training data
fn score_novelty(name: &str, training_data: &HashSet<String>) -> f64 {
    if training_data.contains(name) {
        -5.0
    } else {
        1.0
    }
}

// --- Main Evolution Loop ---

fn main() {
    println!("--- Starting Aesthetic Evolution (Rust Edition) ---");
    println!(
        "Pop: {}, Gens: {}, Threads: Parallel",
        POPULATION_SIZE, GENERATIONS
    );

    let mut run_seed: Option<u64> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--seed" {
            i += 1;
            if i < args.len() {
                run_seed = args[i].parse::<u64>().ok();
            }
        }
        i += 1;
    }
    if let Some(seed) = run_seed {
        println!("Seed: {}", seed);
    }

    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args([
                "-o",
                INPUT_FILE,
                "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
            ])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    let mut population: Vec<AestheticCandidate> = (0..POPULATION_SIZE)
        .map(|_| AestheticCandidate::new_random())
        .collect();

    for gen in 0..GENERATIONS {
        let start_time = Instant::now();
        println!("\n=== Generation {}/{} ===", gen + 1, GENERATIONS);

        // Evaluate all organisms in parallel
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, genome)| {
                genome.evaluate_candidate(&training_data, run_seed, i + gen * POPULATION_SIZE);
            });

        for (i, g) in population.iter().enumerate() {
            println!(
                "Org {}: [Emb:{} Head:{} Lay:{} LR:{:.5}] -> Score: {:.4}",
                i + 1,
                g.n_emb,
                g.n_head,
                g.n_layer,
                g.lr,
                g.fitness
            );
            if !g.names.is_empty() {
                println!(
                    "    Sample: {}",
                    g.names
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }

        let gen_time = start_time.elapsed();
        println!("--- Generation Time: {:.2?} ---", gen_time);

        // Sort by fitness (higher = more beautiful)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best = &population[0];
        println!(
            "\n>> Gen {} Champion: [Emb:{} Head:{} Lay:{} LR:{:.5}]",
            gen + 1,
            best.n_emb,
            best.n_head,
            best.n_layer,
            best.lr
        );
        println!(">> Score: {:.4}", best.fitness);

        // Breed next generation: keep elites, mutate the rest
        if gen < GENERATIONS - 1 {
            let mut new_pop = Vec::with_capacity(POPULATION_SIZE);
            for i in 0..ELITISM {
                new_pop.push(population[i].clone());
            }

            let mut rng = rand::thread_rng();
            while new_pop.len() < POPULATION_SIZE {
                let parent = &population[rng.gen_range(0..ELITISM)];
                let mut child = parent.clone();
                child.mutate();
                new_pop.push(child);
            }
            population = new_pop;
        }
    }
}
