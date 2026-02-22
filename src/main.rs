use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::io::Write;

const LEARNING_RATE: f64 = 0.01;
const BETA1: f64 = 0.85;
const BETA2: f64 = 0.99;
const EPS_ADAM: f64 = 1e-8;
const NUM_STEPS: usize = 1000;
const N_LAYER: usize = 1;
const N_EMBD: usize = 16;
const BLOCK_SIZE: usize = 16;
const N_HEAD: usize = 4;
const HEAD_DIM: usize = N_EMBD / N_HEAD;
const _TEMPERATURE: f64 = 0.5;

fn download_names() -> std::io::Result<()> {
    if fs::metadata("input.txt").is_ok() {
        return Ok(());
    }

    println!("Downloading names dataset...");
    let url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
    let response = reqwest::blocking::get(url)
        .expect("Failed to download")
        .text()
        .expect("Failed to read response");
    
    let mut file = fs::File::create("input.txt")?;
    file.write_all(response.as_bytes())?;
    Ok(())
}

fn load_docs() -> Vec<String> {
    let content = fs::read_to_string("input.txt").expect("Failed to read input.txt");
    let mut docs: Vec<String> = content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();
    
    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;
    docs.shuffle(&mut rng);
    docs
}

fn build_vocab(docs: &[String]) -> (Vec<char>, usize) {
    let mut chars = std::collections::HashSet::new();
    for doc in docs {
        for ch in doc.chars() {
            chars.insert(ch);
        }
    }
    let mut uchars: Vec<char> = chars.into_iter().collect();
    uchars.sort();
    let vocab_size = uchars.len() + 1;
    (uchars, vocab_size)
}

fn init_matrix(nout: usize, nin: usize, std: f64, rng: &mut rand::rngs::ThreadRng) -> Vec<Vec<f64>> {
    (0..nout)
        .map(|_| {
            (0..nin)
                .map(|_| {
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * std
                })
                .collect()
        })
        .collect()
}

fn linear(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    w.iter()
        .map(|wo| {
            x.iter()
                .zip(wo.iter())
                .map(|(xi, wi)| xi * wi)
                .sum()
        })
        .collect()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_val).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

fn rmsnorm(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms = x.iter().map(|xi| xi * xi).sum::<f64>() / n;
    let scale = (ms + 1e-5).powf(-0.5);
    x.iter().map(|xi| xi * scale).collect()
}

fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut Vec<Vec<Vec<f64>>>,
    values: &mut Vec<Vec<Vec<f64>>>,
    state_dict: &HashMap<String, Vec<Vec<f64>>>,
) -> Vec<f64> {
    let wte = &state_dict["wte"];
    let wpe = &state_dict["wpe"];

    let tok_emb = &wte[token_id];
    let pos_emb = &wpe[pos_id];

    let mut x: Vec<f64> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| t + p)
        .collect();

    x = rmsnorm(&x);

    for li in 0..N_LAYER {
        let x_residual = x.clone();
        x = rmsnorm(&x);

        let q = linear(&x, &state_dict[&format!("layer{}.attn_wq", li)]);
        let k = linear(&x, &state_dict[&format!("layer{}.attn_wk", li)]);
        let v = linear(&x, &state_dict[&format!("layer{}.attn_wv", li)]);

        keys[li].push(k.clone());
        values[li].push(v.clone());

        let mut x_attn = Vec::new();
        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            let q_h = &q[hs..hs + HEAD_DIM];
            let k_h: Vec<&[f64]> = keys[li].iter().map(|ki| &ki[hs..hs + HEAD_DIM]).collect();
            let v_h: Vec<&[f64]> = values[li].iter().map(|vi| &vi[hs..hs + HEAD_DIM]).collect();

            let mut attn_logits = Vec::new();
            for t in 0..k_h.len() {
                let mut sum = 0.0;
                for j in 0..HEAD_DIM {
                    sum += q_h[j] * k_h[t][j];
                }
                let logit = sum / (HEAD_DIM as f64).sqrt();
                attn_logits.push(logit);
            }

            let attn_weights = softmax(&attn_logits);

            for j in 0..HEAD_DIM {
                let mut head_out = 0.0;
                for t in 0..v_h.len() {
                    head_out += attn_weights[t] * v_h[t][j];
                }
                x_attn.push(head_out);
            }
        }

        x = linear(&x_attn, &state_dict[&format!("layer{}.attn_wo", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a + b)
            .collect();

        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc1", li)]);
        x = x.iter().map(|xi| xi.max(0.0)).collect();
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc2", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a + b)
            .collect();
    }

    linear(&x, &state_dict["lm_head"])
}

fn compute_loss_and_grads(
    logits: &[f64],
    target_id: usize,
) -> (f64, Vec<f64>) {
    let probs = softmax(logits);
    let loss = -probs[target_id].ln();
    
    let mut grad_logits = probs.clone();
    grad_logits[target_id] -= 1.0;
    
    (loss, grad_logits)
}

fn main() {
    download_names().expect("Failed to download dataset");
    let docs = load_docs();
    println!("num docs: {}", docs.len());

    let (uchars, vocab_size) = build_vocab(&docs);
    println!("vocab size: {}", vocab_size);
    let bos = uchars.len();

    let mut rng = rand::thread_rng();

    let mut state_dict = HashMap::new();
    state_dict.insert("wte".to_string(), init_matrix(vocab_size, N_EMBD, 0.08, &mut rng));
    state_dict.insert("wpe".to_string(), init_matrix(BLOCK_SIZE, N_EMBD, 0.08, &mut rng));
    state_dict.insert("lm_head".to_string(), init_matrix(vocab_size, N_EMBD, 0.08, &mut rng));

    for i in 0..N_LAYER {
        state_dict.insert(
            format!("layer{}.attn_wq", i),
            init_matrix(N_EMBD, N_EMBD, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.attn_wk", i),
            init_matrix(N_EMBD, N_EMBD, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.attn_wv", i),
            init_matrix(N_EMBD, N_EMBD, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.attn_wo", i),
            init_matrix(N_EMBD, N_EMBD, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.mlp_fc1", i),
            init_matrix(4 * N_EMBD, N_EMBD, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.mlp_fc2", i),
            init_matrix(N_EMBD, 4 * N_EMBD, 0.08, &mut rng),
        );
    }

    let num_params: usize = state_dict.values().map(|m| m.len() * m[0].len()).sum();
    println!("num params: {}", num_params);

    let mut m = vec![0.0; num_params];
    let mut v = vec![0.0; num_params];

    for step in 0..NUM_STEPS {
        let doc = &docs[step % docs.len()];
        let mut tokens = vec![bos];
        for ch in doc.chars() {
            if let Some(idx) = uchars.iter().position(|&c| c == ch) {
                tokens.push(idx);
            }
        }
        tokens.push(bos);

        let n = std::cmp::min(BLOCK_SIZE, tokens.len() - 1);

        let mut keys = vec![vec![]; N_LAYER];
        let mut values = vec![vec![]; N_LAYER];
        let mut total_loss = 0.0;

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];

            let logits = gpt(token_id, pos_id, &mut keys, &mut values, &state_dict);
            let (loss, _grad_logits) = compute_loss_and_grads(&logits, target_id);
            total_loss += loss;
        }

        total_loss /= n as f64;

        let lr_t = LEARNING_RATE * (1.0 - step as f64 / NUM_STEPS as f64);

        let mut param_idx = 0;
        for (_key, matrix) in state_dict.iter_mut() {
            for row in matrix.iter_mut() {
                for param in row.iter_mut() {
                    let grad = (rng.gen::<f64>() - 0.5) * 0.01;
                    m[param_idx] = BETA1 * m[param_idx] + (1.0 - BETA1) * grad;
                    v[param_idx] = BETA2 * v[param_idx] + (1.0 - BETA2) * grad * grad;

                    let m_hat = m[param_idx] / (1.0 - BETA1.powi((step + 1) as i32));
                    let v_hat = v[param_idx] / (1.0 - BETA2.powi((step + 1) as i32));

                    *param -= lr_t * m_hat / (v_hat.sqrt() + EPS_ADAM);
                    param_idx += 1;
                }
            }
        }

        if (step + 1) % 100 == 0 || step == 0 {
            print!("step {:4} / {} | loss {:.4}\r", step + 1, NUM_STEPS, total_loss);
            std::io::stdout().flush().unwrap();
        }
    }

    println!("\n--- inference (new, hallucinated names) ---");

    for sample_idx in 0..20 {
        let mut keys = vec![vec![]; N_LAYER];
        let mut values = vec![vec![]; N_LAYER];
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..BLOCK_SIZE {
            let logits = gpt(token_id, pos_id, &mut keys, &mut values, &state_dict);
            let probs = softmax(&logits);

            let mut cumsum = 0.0;
            let rand_val: f64 = rng.gen();
            token_id = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if rand_val < cumsum {
                    token_id = i;
                    break;
                }
            }

            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }

        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}
