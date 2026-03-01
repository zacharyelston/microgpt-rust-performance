use microgpt_rust::{train_and_generate, TrainingConfig};

const NUM_TRIALS: usize = 3;

fn main() {
    let datasets = [
        ("Real Words", "data/words.txt"),
        ("Non-Words", "data/nonwords.txt"),
        ("Names", "input.txt"),
    ];

    let (cfg_base, origin) = match TrainingConfig::load_genome() {
        Some((genome, loss, gen)) => {
            println!("Using evolved genome (gen {}, loss {:.4})", gen, loss);
            (genome, "evolved")
        }
        None => {
            println!("Using default config (no genome.json found)");
            (TrainingConfig::default(), "default")
        }
    };

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║         MicroGPT A/B Test: Words vs Noise       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("Config [{}]: Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.4} Steps:{}",
        origin, cfg_base.n_emb, cfg_base.n_head, cfg_base.n_layer,
        cfg_base.n_ctx, cfg_base.n_ff_exp, cfg_base.lr, cfg_base.steps);
    println!("Trials per dataset: {}", NUM_TRIALS);
    println!();

    let mut results = Vec::new();

    for (name, path) in &datasets {
        if std::fs::metadata(path).is_err() {
            println!("  [skip] {} — file not found: {}", name, path);
            continue;
        }

        let data = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = data.lines().filter(|l| !l.is_empty()).collect();
        let vocab_chars: std::collections::HashSet<char> = data.chars().collect();
        let avg_word_len = lines.iter().map(|l| l.len()).sum::<usize>() as f64 / lines.len() as f64;

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Dataset:    {}", name);
        println!("  File:       {}", path);
        println!("  Entries:    {}", lines.len());
        println!("  Vocab:      {} chars", vocab_chars.len());
        println!("  Avg length: {:.1}", avg_word_len);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut trial_losses = Vec::new();
        let mut best_samples = Vec::new();
        let mut best_loss = f64::MAX;

        for t in 0..NUM_TRIALS {
            let mut cfg = cfg_base.clone();
            cfg.input_file = path.to_string();
            cfg.gen_samples = 15;

            let result = train_and_generate(&cfg, true);
            println!("  Trial {}: loss {:.4}", t + 1, result.final_loss);
            trial_losses.push(result.final_loss);

            if result.final_loss < best_loss {
                best_loss = result.final_loss;
                best_samples = result.names.clone();
            }
        }

        let avg_loss = trial_losses.iter().sum::<f64>() / trial_losses.len() as f64;
        let min_loss = trial_losses.iter().cloned().fold(f64::MAX, f64::min);
        let max_loss = trial_losses.iter().cloned().fold(f64::MIN, f64::max);

        println!();
        println!("  Avg loss: {:.4}  (range {:.4} - {:.4})", avg_loss, min_loss, max_loss);
        println!("  Best samples:");
        for (i, s) in best_samples.iter().enumerate() {
            println!("    {:>2}. {}", i + 1, s);
        }
        println!();

        results.push((*name, avg_loss, min_loss, max_loss, best_samples));
    }

    println!("╔══════════════════════════════════════════════════╗");
    println!("║                   A/B Results                   ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    for (name, avg, min, max, samples) in &results {
        let avg_len = if samples.is_empty() { 0.0 } else {
            samples.iter().map(|s| s.len() as f64).sum::<f64>() / samples.len() as f64
        };
        println!("  {:<12} Avg loss: {:.4}  (best {:.4}, worst {:.4})",
            name, avg, min, max);
        println!("               Avg gen length: {:.1}  Top 5: {}",
            avg_len,
            samples.iter().take(5).cloned().collect::<Vec<_>>().join(", "));
        println!();
    }

    if results.len() >= 2 {
        let words_loss = results[0].1;
        let noise_loss = results[1].1;

        println!("  ─── Analysis ───");
        println!();

        if words_loss < noise_loss {
            let diff = ((noise_loss - words_loss) / noise_loss * 100.0).abs();
            println!("  Real words are {:.1}% easier to learn than random noise.", diff);
            println!("  The model detects structure in natural language.");
        } else {
            let diff = ((words_loss - noise_loss) / noise_loss * 100.0).abs();
            println!("  Random noise is {:.1}% easier to learn than real words.", diff);
            println!("  Real words have more complex patterns to capture.");
        }

        if results.len() >= 3 {
            let names_loss = results[2].1;
            println!();
            if names_loss < words_loss {
                println!("  Names ({:.4}) learned better than common words ({:.4}).",
                    names_loss, words_loss);
                println!("  Names have stronger patterns (common suffixes like -ly, -er, -on).");
            } else {
                println!("  Common words ({:.4}) learned better than names ({:.4}).",
                    words_loss, names_loss);
            }
        }

        println!();
        println!("  Word-likeness signal: the gap between real and noise loss");
        println!("  tells us how much learnable structure exists in the data.");
        println!("  A model trained on words should generate word-like outputs;");
        println!("  one trained on noise should produce random letter soup.");
    }
}
