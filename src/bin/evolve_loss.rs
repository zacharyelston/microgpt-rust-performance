/*
    MicroGPT Loss Evolution Engine
    
    Evolves hyperparameters to minimize training loss.
    Target: loss < 1.9
    
    Uses Rayon for parallel evaluation of the population.
    Designed to converge within reasonable compute time.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

const POPULATION_SIZE: usize = 6;
const MAX_GENERATIONS: usize = 15;
const ELITISM: usize = 2;
const TARGET_LOSS: f64 = 1.9;
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
}

impl Genome {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let n_emb = *[16, 24, 32].choose(&mut rng).unwrap();
        let n_head = *[2, 4].choose(&mut rng).unwrap();
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(1..=2),
            n_ctx: 16,
            n_ff_exp: *[2, 4].choose(&mut rng).unwrap(),
            lr: rng.gen_range(0.005..0.015),
            steps: *[300, 500, 750, 1000].choose(&mut rng).unwrap(),
            loss: f64::MAX,
            evaluated: false,
        };
        g.enforce_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..6) {
            0 => self.n_emb = *[16, 24, 32].choose(&mut rng).unwrap(),
            1 => self.n_head = *[2, 4].choose(&mut rng).unwrap(),
            2 => self.n_layer = rng.gen_range(1..=3),
            3 => self.lr = (self.lr * rng.gen_range(0.6..1.5)).clamp(0.001, 0.03),
            4 => self.steps = (self.steps as i32 + *[-250, 250, 500].choose(&mut rng).unwrap()).clamp(300, 2000) as usize,
            5 => self.n_ff_exp = *[2, 4].choose(&mut rng).unwrap(),
            _ => {},
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            self.n_head = 2;
        }
        if self.n_emb % self.n_head != 0 {
            self.n_emb = (self.n_emb / self.n_head) * self.n_head;
            if self.n_emb == 0 { self.n_emb = 16; }
        }
    }

    fn evaluate(&mut self) {
        if self.evaluated {
            return;
        }

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

        let result = train_and_generate(&config, true);
        self.loss = result.final_loss;
        self.evaluated = true;
    }

    fn desc(&self) -> String {
        format!("[Emb:{} Head:{} Lay:{} FF:{} LR:{:.4} Steps:{}]",
            self.n_emb, self.n_head, self.n_layer, self.n_ff_exp, self.lr, self.steps)
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

fn crossover(a: &Genome, b: &Genome) -> Genome {
    let mut rng = rand::thread_rng();
    let mut child = Genome {
        n_emb: if rng.gen() { a.n_emb } else { b.n_emb },
        n_head: if rng.gen() { a.n_head } else { b.n_head },
        n_layer: if rng.gen() { a.n_layer } else { b.n_layer },
        n_ctx: a.n_ctx,
        n_ff_exp: if rng.gen() { a.n_ff_exp } else { b.n_ff_exp },
        lr: if rng.gen() { a.lr } else { b.lr },
        steps: std::cmp::max(a.steps, b.steps),
        loss: f64::MAX,
        evaluated: false,
    };
    child.enforce_constraints();
    child
}

fn main() {
    println!("=== MicroGPT Loss Evolution Engine ===");
    println!("Target: loss < {:.1}", TARGET_LOSS);
    println!("Population: {}, Max Generations: {}", POPULATION_SIZE, MAX_GENERATIONS);
    println!();

    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    let mut population: Vec<Genome> = (0..POPULATION_SIZE).map(|_| Genome::new_random()).collect();
    let mut best_ever = Genome::new_random();
    best_ever.loss = f64::MAX;
    let total_start = Instant::now();

    for gen in 0..MAX_GENERATIONS {
        let gen_start = Instant::now();
        println!("--- Generation {}/{} ---", gen + 1, MAX_GENERATIONS);

        population.par_iter_mut().for_each(|genome| {
            genome.evaluate();
        });

        population.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

        for (i, g) in population.iter().enumerate() {
            let marker = if i < ELITISM { ">" } else { " " };
            println!("{} #{}: {} Loss: {:.4}", marker, i + 1, g.desc(), g.loss);
        }

        let gen_best = &population[0];
        if gen_best.loss < best_ever.loss {
            best_ever = gen_best.clone();
        }

        let elapsed = gen_start.elapsed().as_secs_f64();
        println!("  Best: {:.4} | All-time: {:.4} | {:.0}s\n", gen_best.loss, best_ever.loss, elapsed);

        if best_ever.loss < TARGET_LOSS {
            println!("========================================");
            println!("  TARGET REACHED! Loss {:.4} < {:.1}", best_ever.loss, TARGET_LOSS);
            println!("  Config: {}", best_ever.desc());
            println!("  Total time: {:.0}s", total_start.elapsed().as_secs_f64());
            println!("========================================");

            println!("\nFinal run with winning config (10 samples)...");
            let result = train_and_generate(&best_ever.to_config(10), false);
            println!("Final loss: {:.4}", result.final_loss);
            return;
        }

        if gen < MAX_GENERATIONS - 1 {
            let mut new_pop = Vec::with_capacity(POPULATION_SIZE);
            for i in 0..ELITISM {
                new_pop.push(population[i].clone());
            }

            let mut rng = rand::thread_rng();
            while new_pop.len() < POPULATION_SIZE {
                if rng.gen::<f64>() < 0.3 && POPULATION_SIZE >= 4 {
                    let p1 = &population[rng.gen_range(0..ELITISM)];
                    let p2 = &population[rng.gen_range(0..POPULATION_SIZE / 2)];
                    let mut child = crossover(p1, p2);
                    if rng.gen::<f64>() < 0.5 {
                        child.mutate();
                    }
                    new_pop.push(child);
                } else {
                    let parent = &population[rng.gen_range(0..ELITISM)];
                    let mut child = parent.clone();
                    child.mutate();
                    new_pop.push(child);
                }
            }
            population = new_pop;
        }
    }

    println!("========================================");
    println!("  Evolution complete ({} generations)", MAX_GENERATIONS);
    println!("  Best loss: {:.4} (target was < {:.1})", best_ever.loss, TARGET_LOSS);
    println!("  Config: {}", best_ever.desc());
    println!("  Total time: {:.0}s", total_start.elapsed().as_secs_f64());
    println!("========================================");

    println!("\nFinal run with best config (10 samples)...");
    let result = train_and_generate(&best_ever.to_config(10), false);
    println!("Final loss: {:.4}", result.final_loss);
}
