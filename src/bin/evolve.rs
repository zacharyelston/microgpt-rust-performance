/*
    MicroGPT Evolutionary Engine (Rust Edition)
    
    This binary implements the evolutionary algorithm directly in Rust,
    utilizing Rayon for parallel processing of the population.
    
    It treats MicroGPT hyperparameters as a Genome and evolves them
    to maximize aesthetic fitness of generated names.
*/

use microgpt_rust::{train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

// Configuration
const POPULATION_SIZE: usize = 12; // Matches Python parallel version
const GENERATIONS: usize = 5;
const ELITISM: usize = 2;
const INPUT_FILE: &str = "input.txt";
const TRAIN_STEPS: usize = 300;

#[derive(Clone, Debug)]
struct Genome {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    lr: f64,
    fitness: f64,
    loss: f64,
    names: Vec<String>,
}

impl Genome {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut g = Genome {
            n_emb: *[16, 24, 32].choose(&mut rng).unwrap(),
            n_head: *[2, 4].choose(&mut rng).unwrap(),
            n_layer: rng.gen_range(1..=3),
            lr: rng.gen_range(0.001..0.015),
            fitness: 0.0,
            loss: 0.0,
            names: Vec::new(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let choice = rng.gen_range(0..4);
        match choice {
            0 => self.n_emb = *[16, 24, 32, 40].choose(&mut rng).unwrap(),
            1 => self.n_head = *[2, 4].choose(&mut rng).unwrap(),
            2 => self.n_layer = (self.n_layer as i32 + *[-1, 1].choose(&mut rng).unwrap()).max(1).min(4) as usize,
            3 => self.lr = (self.lr * rng.gen_range(0.7..1.3)).max(0.0001).min(0.1),
            _ => {},
        }
        self.enforce_constraints();
        self.fitness = 0.0; // Reset fitness
        self.names.clear();
        self.loss = 0.0;
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            self.n_head = 2; // Fallback
        }
        if self.n_emb % self.n_head != 0 {
             // If still invalid, adjust emb
             self.n_emb = (self.n_emb / self.n_head) * self.n_head;
             if self.n_emb == 0 { self.n_emb = self.n_head; }
        }
    }

    fn evaluate(&mut self, training_data: &HashSet<String>, gen: usize, org_id: usize) {
        if self.fitness != 0.0 && !self.names.is_empty() {
            return; // Already evaluated
        }

        let log_file = format!("logs/gen_{}_org_{}.csv", gen, org_id);

        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            lr: self.lr,
            steps: TRAIN_STEPS,
            input_file: INPUT_FILE.to_string(),
            log_file: Some(log_file),
            ..Default::default()
        };

        // Run training and generation (this handles the heavy lifting)
        let (generated_names, final_loss) = train_and_generate(&config, true); // silent=true
        
        // Calculate fitness
        let score = calculate_fitness(&generated_names, training_data);
        
        self.names = generated_names;
        self.fitness = score;
        self.loss = final_loss;
    }
}

// --- Judge Logic ---

fn calculate_fitness(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() { return 0.0; }
    
    let mut total_score = 0.0;
    let count = names.len() as f64;

    for name in names {
        let name = name.trim().to_lowercase();
        // 0 points for invalid strings
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { 
            continue; 
        }

        let rating = rate_name(&name, training_data);
        total_score += rating as f64;
    }

    total_score / count
}

fn rate_name(name: &str, training_data: &HashSet<String>) -> i32 {
    // 0: Not a fit (Memorized or Unpronounceable)
    if training_data.contains(name) {
        return 0;
    }
    
    let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
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
        // Strict flow check: Fail if > 3 consecutive
        if cons_v > 3 || cons_c > 3 {
            return 0;
        }
    }

    // 1: It fits (Valid & Novel)
    // Check for "Greatness" to bump to 2
    
    // Palindrome
    let chars: Vec<char> = name.chars().collect();
    if name.len() > 3 && chars.iter().eq(chars.iter().rev()) {
        return 2;
    }

    // Repeating sub-patterns (e.g. "lala")
    if name.len() >= 4 {
        let mid = name.len() / 2;
        if name[..mid] == name[mid..mid*2] {
            return 2;
        }
    }
    
    // Aesthetic Endings (Subjective "Great")
    if name.ends_with("ia") || name.ends_with("ius") {
        return 2;
    }

    // Default Good Score
    1
}

// --- Main Evolution Loop ---

fn main() {
    println!("--- Starting Aesthetic Evolution (Rust Edition) ---");
    println!("Pop: {}, Gens: {}, Threads: Parallel", POPULATION_SIZE, GENERATIONS);

    // Ensure logs directory exists
    fs::create_dir_all("logs").ok();

    // Load training data once for judge
    let raw = fs::read_to_string(INPUT_FILE).unwrap_or_default();
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    // Init Population
    let mut population: Vec<Genome> = (0..POPULATION_SIZE).map(|_| Genome::new_random()).collect();

    for gen in 0..GENERATIONS {
        let start_time = Instant::now();
        println!("\n=== Generation {}/{} ===", gen + 1, GENERATIONS);

        // Parallel Evaluation
        // We use par_iter_mut to evaluate in parallel
        population.par_iter_mut().enumerate().for_each(|(i, genome)| {
             genome.evaluate(&training_data, gen + 1, i + 1);
        });

        // Print results (after parallel step to keep output clean)
        for (i, g) in population.iter().enumerate() {
            println!("Org {}: [Emb:{} Head:{} Lay:{} LR:{:.5}] -> Score: {:.4} (Loss: {:.4})", 
                i+1, g.n_emb, g.n_head, g.n_layer, g.lr, g.fitness, g.loss);
            if !g.names.is_empty() {
                println!("    Sample: {}", g.names.iter().take(3).cloned().collect::<Vec<_>>().join(", "));
            }
        }

        let gen_time = start_time.elapsed();
        println!("--- Generation Time: {:.2?} ---", gen_time);

        // Sort by fitness (descending)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        
        let best = &population[0];
        println!("\n>> Gen {} Champion: [Emb:{} Head:{} Lay:{} LR:{:.5}]", gen+1, best.n_emb, best.n_head, best.n_layer, best.lr);
        println!(">> Score: {:.4} (Loss: {:.4})", best.fitness, best.loss);

        // Elitism & Reproduction
        if gen < GENERATIONS - 1 {
            let mut new_pop = Vec::with_capacity(POPULATION_SIZE);
            // Elitism
            for i in 0..ELITISM {
                new_pop.push(population[i].clone());
            }
            
            // Fill rest
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
