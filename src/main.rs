/*
    MicroGPT: A Living Transformer — Main Binary

    This is the organism. When genome.json exists (written by the
    evolution engine), it loads evolved hyperparameters and becomes
    a different creature. Without it, it runs in primordial form
    with hardcoded defaults.

    The lifecycle:
      1. Run evolve_loss to discover optimal DNA
      2. Evolution writes genome.json
      3. This binary reads it and becomes the evolved version
      4. Delete genome.json to return to primordial state
*/

use microgpt_rust::{train_and_generate, TrainingConfig};

const INPUT_URL: &str = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Check for evolved genome — this is where self-modification happens
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

    // CLI overrides — these take precedence over both defaults and genome
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
            "-d" | "--data" => { i += 1; if i < args.len() { cfg.input_file = args[i].clone(); } }
            "--silent" => { silent = true; }
            _ => {}
        }
        i += 1;
    }

    // Ensure training data exists
    if std::fs::metadata(&cfg.input_file).is_err() {
        let _ = std::process::Command::new("curl").args(["-o", &cfg.input_file, INPUT_URL]).output();
    }

    let result = train_and_generate(&cfg, silent);

    println!("\nOrganism: {} | Loss: {:.4} | Params: {}",
        origin, result.final_loss, result.num_params);
}
