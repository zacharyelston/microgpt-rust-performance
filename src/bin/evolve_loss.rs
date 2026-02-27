/*
    MicroGPT Loss Evolution Engine
    
    Evolves hyperparameters to minimize training loss.
    Always runs all generations to learn what works.
    Prints analysis at the end showing hyperparameter trends.
    
    Uses Rayon for parallel evaluation of the population.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

const POPULATION_SIZE: usize = 6;
const NUM_GENERATIONS: usize = 5;
const ELITISM: usize = 2;
const TARGET_LOSS: f64 = 1.2;
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

#[derive(Clone, Debug)]
struct HistoryEntry {
    gen: usize,
    genome: Genome,
}

fn main() {
    println!("=== MicroGPT Loss Evolution Engine ===");
    println!("Target: loss < {:.1}", TARGET_LOSS);
    println!("Population: {}, Generations: {} (always runs all)", POPULATION_SIZE, NUM_GENERATIONS);
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
    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut gen_bests: Vec<(usize, f64)> = Vec::new();
    let total_start = Instant::now();
    let mut target_gen: Option<usize> = None;

    for gen in 0..NUM_GENERATIONS {
        let gen_start = Instant::now();
        println!("--- Generation {}/{} ---", gen + 1, NUM_GENERATIONS);

        population.par_iter_mut().for_each(|genome| {
            genome.evaluate();
        });

        population.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

        for (i, g) in population.iter().enumerate() {
            let marker = if i < ELITISM { ">" } else { " " };
            println!("{} #{}: {} Loss: {:.4}", marker, i + 1, g.desc(), g.loss);
            history.push(HistoryEntry { gen: gen + 1, genome: g.clone() });
        }

        let gen_best = &population[0];
        if gen_best.loss < best_ever.loss {
            best_ever = gen_best.clone();
        }
        gen_bests.push((gen + 1, gen_best.loss));

        if target_gen.is_none() && best_ever.loss < TARGET_LOSS {
            target_gen = Some(gen + 1);
            println!("  ** Target {:.1} reached! Continuing to evolve... **", TARGET_LOSS);
        }

        let elapsed = gen_start.elapsed().as_secs_f64();
        println!("  Best: {:.4} | All-time: {:.4} | {:.0}s\n", gen_best.loss, best_ever.loss, elapsed);

        if gen < NUM_GENERATIONS - 1 {
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

    let total_time = total_start.elapsed().as_secs_f64();
    let total_evals = history.len();

    println!("========================================");
    println!("          EVOLUTION COMPLETE");
    println!("========================================");
    println!("  Generations: {}", NUM_GENERATIONS);
    println!("  Total evaluations: {}", total_evals);
    println!("  Total time: {:.0}s", total_time);
    println!("  Best loss: {:.4}", best_ever.loss);
    println!("  Best config: {}", best_ever.desc());
    if let Some(g) = target_gen {
        println!("  Target {:.1} first reached: generation {}", TARGET_LOSS, g);
    } else {
        println!("  Target {:.1} NOT reached", TARGET_LOSS);
    }

    println!("\n--- Loss Progression ---");
    for (gen, loss) in &gen_bests {
        let bar_len = ((4.0 - loss) * 15.0).max(0.0).min(60.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  Gen {:2}: {:.4} {}", gen, loss, bar);
    }

    println!("\n--- What We Learned ---");

    let top_n = std::cmp::min(10, history.len());
    let mut sorted_history = history.clone();
    sorted_history.sort_by(|a, b| a.genome.loss.partial_cmp(&b.genome.loss).unwrap());

    println!("\nTop {} configs across all generations:", top_n);
    for (i, entry) in sorted_history.iter().take(top_n).enumerate() {
        println!("  {:2}. Gen {} | Loss {:.4} | {}", i + 1, entry.gen, entry.genome.loss, entry.genome.desc());
    }

    let top_configs: Vec<&Genome> = sorted_history.iter().take(top_n).map(|e| &e.genome).collect();

    let avg_emb: f64 = top_configs.iter().map(|g| g.n_emb as f64).sum::<f64>() / top_n as f64;
    let avg_head: f64 = top_configs.iter().map(|g| g.n_head as f64).sum::<f64>() / top_n as f64;
    let avg_layer: f64 = top_configs.iter().map(|g| g.n_layer as f64).sum::<f64>() / top_n as f64;
    let avg_ff: f64 = top_configs.iter().map(|g| g.n_ff_exp as f64).sum::<f64>() / top_n as f64;
    let avg_lr: f64 = top_configs.iter().map(|g| g.lr).sum::<f64>() / top_n as f64;
    let avg_steps: f64 = top_configs.iter().map(|g| g.steps as f64).sum::<f64>() / top_n as f64;

    let all_avg_emb: f64 = history.iter().map(|e| e.genome.n_emb as f64).sum::<f64>() / total_evals as f64;
    let all_avg_layer: f64 = history.iter().map(|e| e.genome.n_layer as f64).sum::<f64>() / total_evals as f64;
    let all_avg_lr: f64 = history.iter().map(|e| e.genome.lr).sum::<f64>() / total_evals as f64;
    let all_avg_steps: f64 = history.iter().map(|e| e.genome.steps as f64).sum::<f64>() / total_evals as f64;

    println!("\nHyperparameter trends (top {} vs all {}):", top_n, total_evals);
    println!("  {:12} {:>10} {:>10}", "Param", "Top Avg", "All Avg");
    println!("  {:12} {:>10.1} {:>10.1}", "Embedding", avg_emb, all_avg_emb);
    println!("  {:12} {:>10.1} {:>10.1}", "Heads", avg_head, top_configs.iter().map(|g| g.n_head as f64).sum::<f64>() / top_n as f64);
    println!("  {:12} {:>10.1} {:>10.1}", "Layers", avg_layer, all_avg_layer);
    println!("  {:12} {:>10.1} {:>10.1}", "FF Mult", avg_ff, history.iter().map(|e| e.genome.n_ff_exp as f64).sum::<f64>() / total_evals as f64);
    println!("  {:12} {:>10.4} {:>10.4}", "Learn Rate", avg_lr, all_avg_lr);
    println!("  {:12} {:>10.0} {:>10.0}", "Steps", avg_steps, all_avg_steps);

    println!("\nInsights:");
    if avg_layer > all_avg_layer + 0.2 {
        println!("  - More layers correlate with lower loss (top avg {:.1} vs overall {:.1})", avg_layer, all_avg_layer);
    }
    if avg_emb > all_avg_emb + 2.0 {
        println!("  - Larger embeddings help (top avg {:.0} vs overall {:.0})", avg_emb, all_avg_emb);
    } else if avg_emb < all_avg_emb - 2.0 {
        println!("  - Smaller embeddings performed better (top avg {:.0} vs overall {:.0})", avg_emb, all_avg_emb);
    }
    if avg_steps > all_avg_steps + 50.0 {
        println!("  - More training steps improve results (top avg {:.0} vs overall {:.0})", avg_steps, all_avg_steps);
    }
    if avg_lr > all_avg_lr * 1.15 {
        println!("  - Higher learning rates favored (top avg {:.4} vs overall {:.4})", avg_lr, all_avg_lr);
    } else if avg_lr < all_avg_lr * 0.85 {
        println!("  - Lower learning rates favored (top avg {:.4} vs overall {:.4})", avg_lr, all_avg_lr);
    }

    println!("\n--- Final Demo with Best Config ---");
    let result = train_and_generate(&best_ever.to_config(10), false);
    println!("Final loss: {:.4}", result.final_loss);
}
