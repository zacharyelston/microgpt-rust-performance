/* 
   MicroGPT: The Art of Symmetry
   A minimal, aesthetic implementation of a Transformer in Rust.
*/

use microgpt_rust::{train_and_generate, TrainingConfig};

const INPUT_URL: &str = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = TrainingConfig::default();
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

    train_and_generate(&cfg, silent);
}
