/* 
   MicroGPT: The Art of Symmetry
   A minimal, aesthetic implementation of a Transformer in Rust.
*/

use rand::Rng;
use std::{collections::HashSet, io::Write};
use microgpt_rust::{GPT, Val, softmax};

// --- 0. The Configuration ---

struct Config {
    // Model
    n_emb: usize,
    n_ctx: usize,
    n_layer: usize,
    n_head: usize,
    n_ff_exp: usize,
    
    // Training
    steps: usize,
    lr: f64,
    adam_beta1: f64,
    adam_beta2: f64,
    adam_eps: f64,
    checkpoint_interval: usize,

    // Generation
    gen_samples: usize,

    // Constants
    rms_eps: f64,
    init_scale: f64,

    // Data
    input_file: &'static str,
    input_url: &'static str,
}

const CFG: Config = Config {
    n_emb: 16,
    n_ctx: 16,
    n_layer: 1,
    n_head: 4,
    n_ff_exp: 4,
    steps: 200,
    lr: 0.005,
    adam_beta1: 0.85,
    adam_beta2: 0.99,
    adam_eps: 1e-8,
    checkpoint_interval: 20,

    gen_samples: 5,
    rms_eps: 1e-5,
    init_scale: 0.1,

    input_file: "input.txt",
    input_url: "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
};

// --- IV. The Training Loop ---

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut steps, mut lr, mut n_emb, mut n_head, mut n_layer, mut n_ctx, mut n_ff) = (CFG.steps, CFG.lr, CFG.n_emb, CFG.n_head, CFG.n_layer, CFG.n_ctx, CFG.n_ff_exp);
    let mut silent = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-s" | "--steps" => { i += 1; if i < args.len() { steps = args[i].parse().unwrap_or(steps); } }
            "-l" | "--lr" => { i += 1; if i < args.len() { lr = args[i].parse().unwrap_or(lr); } }
            "-e" | "--emb" => { i += 1; if i < args.len() { n_emb = args[i].parse().unwrap_or(n_emb); } }
            "-h" | "--head" => { i += 1; if i < args.len() { n_head = args[i].parse().unwrap_or(n_head); } }
            "-y" | "--layer" => { i += 1; if i < args.len() { n_layer = args[i].parse().unwrap_or(n_layer); } }
            "-c" | "--ctx" => { i += 1; if i < args.len() { n_ctx = args[i].parse().unwrap_or(n_ctx); } }
            "-f" | "--ff" => { i += 1; if i < args.len() { n_ff = args[i].parse().unwrap_or(n_ff); } }
            "--silent" => { silent = true; }
            _ => {}
        }
        i += 1;
    }

    if std::fs::metadata(CFG.input_file).is_err() {
        let _ = std::process::Command::new("curl").args(["-o", CFG.input_file, CFG.input_url]).output();
    }
    let raw = std::fs::read_to_string(CFG.input_file).unwrap_or_else(|_| "emma\nolivia\nava\n".to_string());
    let chars: Vec<char> = { let mut c: Vec<_> = raw.chars().collect::<HashSet<_>>().into_iter().filter(|c| !c.is_whitespace()).collect(); c.sort(); c };
    let vocab = chars.len() + 1;
    
    let model = GPT::new(vocab, n_ctx, n_emb, n_layer, n_head, n_ff, CFG.init_scale, CFG.rms_eps);
    let params = model.params();
    if !silent {
        println!("MicroGPT: {} params, training for {} steps (lr={}, emb={}, head={}, layer={}, ctx={}, ff={})", params.len(), steps, lr, n_emb, n_head, n_layer, n_ctx, n_ff);
    }
    
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    let docs: Vec<&str> = raw.lines().collect();
    
    for step in 0..steps {
        let doc = docs[step % docs.len()];
        let tokens: Vec<usize> = std::iter::once(vocab-1)
            .chain(doc.chars().map(|c| chars.iter().position(|&x| x == c).unwrap()))
            .chain(std::iter::once(vocab-1)).collect();

        let mut loss = Val::new(0.);
        let (mut kc, mut vc) = (vec![vec![]; n_layer], vec![vec![]; n_layer]);
        
        for p in 0..tokens.len()-1 {
            let logits = model.forward(tokens[p], p, &mut kc, &mut vc);
            let probs = softmax(&logits);
            loss = &loss - &probs[tokens[p+1]].log();
        }
        loss = &loss * &Val::new((tokens.len() as f64 - 1.).recip());
        let loss_val = loss.data();
        
        for p in &params { p.zero(); }
        loss.backward();

        let lr_t = lr * (1. - step as f64 / steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m[i] = CFG.adam_beta1 * m[i] + (1. - CFG.adam_beta1) * g;
            v[i] = CFG.adam_beta2 * v[i] + (1. - CFG.adam_beta2) * g * g;
            let m_hat = m[i] / (1. - CFG.adam_beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1. - CFG.adam_beta2.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + CFG.adam_eps);
        }
        if !silent && step % CFG.checkpoint_interval == 0 { print!("step {:4} | loss {:.4}\r", step, loss_val); std::io::stdout().flush().unwrap(); }
    }
    
    if !silent { println!("\n--- Generation ---"); }
    for _ in 0..CFG.gen_samples {
        let (mut kc, mut vc) = (vec![vec![]; n_layer], vec![vec![]; n_layer]);
        let mut tok = vocab - 1;
        if !silent { print!("> "); }
        for p in 0..n_ctx {
            let logits = model.forward(tok, p, &mut kc, &mut vc);
            let probs = softmax(&logits);
            let mut c = 0.;
            let r: f64 = rand::thread_rng().gen();
            let mut next = vocab - 1;
            for (i, v) in probs.iter().enumerate() {
                c += v.data();
                if r < c { next = i; break; }
            }
            tok = next;
            if tok == vocab - 1 { break; }
            print!("{}", chars[tok]);
        }
        println!();
    }
}
