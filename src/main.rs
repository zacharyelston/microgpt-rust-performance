/*
   MicroGPT: A Living Transformer

   A minimal GPT that evolves. When a genome.json exists (written by
   the evolution engine), this program becomes the evolved creature.
   Without it, it runs with default parameters — the primordial form.
*/

use microgpt_rust::{train_and_generate, TrainingConfig};

const INPUT_URL: &str = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (mut cfg, origin) = match TrainingConfig::load_genome() {
        Some((genome, loss, gen)) => {
            println!("=== MicroGPT [Evolved] ===");
            println!("Loaded genome from generation {} (loss: {:.4})", gen, loss);
            println!();
            (genome, "evolved")
        }
        None => {
            println!("=== MicroGPT [Primordial] ===");
            println!("No genome found. Running with default parameters.");
            println!("Run `cargo run --release --bin evolve_loss` to evolve.");
            println!();
            (TrainingConfig::default(), "default")
        }
    };

    cfg.gen_samples = 10;

    let mut silent = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-s" | "--steps" => { i += 1; if i < args.len() { cfg.steps = args[i].parse().unwrap_or(cfg.steps); } }
            "-l" | "--lr" => { i += 1; if i < args.len() { cfg.lr = args[i].parse().unwrap_or(cfg.lr); } }
            "-e" | "--emb" => { i += 1; if i < args.len() { cfg.n_emb = args[i].parse().unwrap_or(cfg.n_emb); } }
            "-h" | "--head" => { i += 1; if i < args.len() { cfg.n_head = args[i].parse().unwrap_or(cfg.n_head); } }
            "-y" | "--layer" => { i += 1; if i < args.len() { cfg.n_layer = args[i].parse().unwrap_or(cfg.n_layer); } }
            "-c" | "--ctx" => { i += 1; if i < args.len() { cfg.n_ctx = args[i].parse().unwrap_or(cfg.n_ctx); } }
            "-f" | "--ff" => { i += 1; if i < args.len() { cfg.n_ff_exp = args[i].parse().unwrap_or(cfg.n_ff_exp); } }
            "--silent" => { silent = true; }
            _ => {}
        }
        i += 1;
    }

    if std::fs::metadata(&cfg.input_file).is_err() {
        let _ = std::process::Command::new("curl").args(["-o", &cfg.input_file, INPUT_URL]).output();
    }

    let result = train_and_generate(&cfg, silent);

    println!("\nOrganism: {} | Loss: {:.4} | Params: {}",
        origin, result.final_loss, result.num_params);
}
