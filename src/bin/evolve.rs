/*
    MicroGPT Evolutionary Engine (Rust Edition)
    
    This binary implements the evolutionary algorithm directly in Rust,
    utilizing Rayon for parallel processing of the population.
    
    It treats MicroGPT hyperparameters as a Genome and evolves them
    to maximize aesthetic fitness of generated names.
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

#[derive(Clone, Debug)]
struct Genome {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    lr: f64,
    fitness: f64,
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
        self.fitness = 0.0;
        self.names.clear();
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            self.n_head = 2;
        }
        if self.n_emb % self.n_head != 0 {
             self.n_emb = (self.n_emb / self.n_head) * self.n_head;
             if self.n_emb == 0 { self.n_emb = self.n_head; }
        }
    }

    fn evaluate(&mut self, training_data: &HashSet<String>) {
        if self.fitness != 0.0 && !self.names.is_empty() {
            return;
        }

        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            lr: self.lr,
            steps: TRAIN_STEPS,
            input_file: INPUT_FILE.to_string(),
            ..Default::default()
        };

        let result = train_and_generate(&config, true);
        let score = calculate_fitness(&result.names, training_data);
        
        self.names = result.names;
        self.fitness = score;
    }
}

// --- Judge Logic ---

fn calculate_fitness(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() { return -100.0; }
    
    let mut total_score = 0.0;
    let mut valid_count = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { continue; }

        let s_flow = score_flow(&name);
        let s_sym = score_symmetry(&name);
        let s_creat = score_creativity(&name, training_data);

        total_score += s_flow * 1.0 + s_sym * 1.2 + s_creat * 2.0;
        valid_count += 1;
    }

    if valid_count == 0 { return -100.0; }
    total_score / valid_count as f64
}

fn score_flow(name: &str) -> f64 {
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
    
    if name.len() >= 4 && name.len() <= 8 {
        score += 0.5;
    }
    score
}

fn score_symmetry(name: &str) -> f64 {
    let mut score = 0.0;
    let chars: Vec<char> = name.chars().collect();
    
    if name.len() > 3 && chars.iter().eq(chars.iter().rev()) {
        score += 2.0;
    }

    if name.len() >= 4 {
        let mid = name.len() / 2;
        if name[..mid] == name[mid..mid*2] {
            score += 1.5;
        }
    }

    if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') {
        score += 0.2;
    }
    score
}

fn score_creativity(name: &str, training_data: &HashSet<String>) -> f64 {
    if training_data.contains(name) {
        -5.0
    } else {
        1.0
    }
}

// --- Main Evolution Loop ---

fn main() {
    println!("--- Starting Aesthetic Evolution (Rust Edition) ---");
    println!("Pop: {}, Gens: {}, Threads: Parallel", POPULATION_SIZE, GENERATIONS);

    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    let mut population: Vec<Genome> = (0..POPULATION_SIZE).map(|_| Genome::new_random()).collect();

    for gen in 0..GENERATIONS {
        let start_time = Instant::now();
        println!("\n=== Generation {}/{} ===", gen + 1, GENERATIONS);

        population.par_iter_mut().enumerate().for_each(|(_i, genome)| {
             genome.evaluate(&training_data);
        });

        for (i, g) in population.iter().enumerate() {
            println!("Org {}: [Emb:{} Head:{} Lay:{} LR:{:.5}] -> Score: {:.4}", 
                i+1, g.n_emb, g.n_head, g.n_layer, g.lr, g.fitness);
            if !g.names.is_empty() {
                println!("    Sample: {}", g.names.iter().take(3).cloned().collect::<Vec<_>>().join(", "));
            }
        }

        let gen_time = start_time.elapsed();
        println!("--- Generation Time: {:.2?} ---", gen_time);

        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        
        let best = &population[0];
        println!("\n>> Gen {} Champion: [Emb:{} Head:{} Lay:{} LR:{:.5}]", gen+1, best.n_emb, best.n_head, best.n_layer, best.lr);
        println!(">> Score: {:.4}", best.fitness);

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
