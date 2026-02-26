use rand::Rng;
use std::{cell::RefCell, collections::{HashMap, HashSet}, ops::{Add, Mul, Neg, Sub}, rc::Rc, io::Write};
use std::fs;
use std::time::Instant;
use std::f64::consts::PI;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
const TEMPERATURE: f64 = 0.5;
const BATCH_SIZE: usize = 4;  // Mini-batch size
const PATIENCE: usize = 50;  // Early stopping patience
const MIN_LR: f64 = 0.0001;  // Minimum learning rate

type ValueRef = Rc<RefCell<Value>>;

#[derive(Clone)]
struct Value {
    data: f64,
    grad: f64,
    children: Vec<(ValueRef, f64)>,
}

#[cfg(feature = "parallel")]
unsafe impl Send for Value {}
#[cfg(feature = "parallel")]
unsafe impl Sync for Value {}

impl Value {
    fn new(data: f64) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: Vec::new(),
        }))
    }

    fn with_children(data: f64, children: Vec<(ValueRef, f64)>) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children,
        }))
    }
}

fn download_names() -> std::io::Result<()> {
    if fs::metadata("input.txt").is_ok() {
        return Ok(());
    }

    println!("Downloading names dataset...");
    let url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
    let _output = std::process::Command::new("curl")
        .args(["-s", "-o", "input.txt", url])
        .output()
        .expect("Failed to download");
    let text = fs::read_to_string("input.txt").expect("Failed to read input.txt");
    
    let mut file = fs::File::create("input.txt")?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

fn add(a: &ValueRef, b: &ValueRef) -> ValueRef {
    let a_val = a.borrow();
    let b_val = b.borrow();
    let data = a_val.data + b_val.data;
    drop(a_val);
    drop(b_val);
    Value::with_children(data, vec![(a.clone(), 1.0), (b.clone(), 1.0)])
}

fn mul(a: &ValueRef, b: &ValueRef) -> ValueRef {
    let a_val = a.borrow();
    let b_val = b.borrow();
    let data = a_val.data * b_val.data;
    let local_grad_a = b_val.data;
    let local_grad_b = a_val.data;
    drop(a_val);
    drop(b_val);
    Value::with_children(data, vec![(a.clone(), local_grad_a), (b.clone(), local_grad_b)])
}

fn pow(a: &ValueRef, exp: f64) -> ValueRef {
    let a_val = a.borrow();
    let data = a_val.data.powf(exp);
    let local_grad = exp * a_val.data.powf(exp - 1.0);
    drop(a_val);
    Value::with_children(data, vec![(a.clone(), local_grad)])
}

fn log(a: &ValueRef) -> ValueRef {
    let a_val = a.borrow();
    let data = a_val.data.ln();
    let local_grad = 1.0 / a_val.data;
    drop(a_val);
    Value::with_children(data, vec![(a.clone(), local_grad)])
}

fn exp(a: &ValueRef) -> ValueRef {
    let a_val = a.borrow();
    let data = a_val.data.exp();
    let local_grad = data;
    drop(a_val);
    Value::with_children(data, vec![(a.clone(), local_grad)])
}

fn relu(a: &ValueRef) -> ValueRef {
    let a_val = a.borrow();
    let data = a_val.data.max(0.0);
    let local_grad = if a_val.data > 0.0 { 1.0 } else { 0.0 };
    drop(a_val);
    Value::with_children(data, vec![(a.clone(), local_grad)])
}

fn backward(loss: &ValueRef) {
    loss.borrow_mut().grad = 1.0;
    
    // Use a simple stack-based reverse traversal instead of topological sort
    let mut stack = vec![loss.clone()];
    let mut processed = std::collections::HashSet::new();
    
    while let Some(node_ref) = stack.pop() {
        let ptr = node_ref.as_ptr() as usize;
        if processed.contains(&ptr) {
            continue;
        }
        processed.insert(ptr);
        
        let node = node_ref.borrow();
        let node_grad = node.grad;
        let children = node.children.clone();
        drop(node);
        
        for (child_ref, local_grad) in children {
            child_ref.borrow_mut().grad += local_grad * node_grad;
            stack.push(child_ref);
        }
    }
}

fn reset_grad(v: &ValueRef) {
    v.borrow_mut().grad = 0.0;
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

fn init_matrix(nout: usize, nin: usize, std: f64, rng: &mut rand::rngs::ThreadRng) -> Vec<Vec<ValueRef>> {
    (0..nout)
        .map(|_| {
            (0..nin)
                .map(|_| {
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    Value::new(z * std)
                })
                .collect()
        })
        .collect()
}

fn linear(x: &[ValueRef], w: &[Vec<ValueRef>]) -> Vec<ValueRef> {
    w.iter()
        .map(|wo| {
            // Fuse all operations into a single node to reduce graph size
            let mut data = 0.0;
            let mut children = Vec::new();
            
            for (xi, wi) in x.iter().zip(wo.iter()) {
                data += xi.borrow().data * wi.borrow().data;
                children.push((xi.clone(), wi.borrow().data));
                children.push((wi.clone(), xi.borrow().data));
            }
            
            Value::with_children(data, children)
        })
        .collect()
}

fn softmax(logits: &[ValueRef]) -> Vec<ValueRef> {
    let max_val = logits
        .iter()
        .map(|l| l.borrow().data)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_id = Value::new(max_val);
    
    let exps: Vec<ValueRef> = logits
        .iter()
        .map(|l| {
            let shifted = add(l, &mul(&max_id, &Value::new(-1.0)));
            exp(&shifted)
        })
        .collect();
    
    let mut total = Value::new(0.0);
    for exp_val in &exps {
        total = add(&total, exp_val);
    }
    
    let inv_total = pow(&total, -1.0);
    exps.iter().map(|e| mul(e, &inv_total)).collect()
}

fn rmsnorm(x: &[ValueRef]) -> Vec<ValueRef> {
    let n = x.len() as f64;
    let mut ms = Value::new(0.0);
    for xi in x {
        let xi_sq = mul(xi, xi);
        ms = add(&ms, &xi_sq);
    }
    let n_id = Value::new(n);
    ms = mul(&ms, &pow(&n_id, -1.0));
    let eps_id = Value::new(1e-5);
    ms = add(&ms, &eps_id);
    let scale = pow(&ms, -0.5);

    x.iter().map(|xi| mul(xi, &scale)).collect()
}

fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut Vec<Vec<Vec<ValueRef>>>,
    values: &mut Vec<Vec<Vec<ValueRef>>>,
    state_dict: &HashMap<String, Vec<Vec<ValueRef>>>,
) -> Vec<ValueRef> {
    let wte = &state_dict["wte"];
    let wpe = &state_dict["wpe"];

    let tok_emb = wte[token_id].clone();
    let pos_emb = wpe[pos_id].clone();

    let mut x: Vec<ValueRef> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| add(t, p))
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
            let k_h: Vec<Vec<ValueRef>> = keys[li]
                .iter()
                .map(|ki| ki[hs..hs + HEAD_DIM].to_vec())
                .collect();
            let v_h: Vec<Vec<ValueRef>> = values[li]
                .iter()
                .map(|vi| vi[hs..hs + HEAD_DIM].to_vec())
                .collect();

            let mut attn_logits = Vec::new();
            for t in 0..k_h.len() {
                let mut sum = Value::new(0.0);
                for j in 0..HEAD_DIM {
                    let prod = mul(&q_h[j], &k_h[t][j]);
                    sum = add(&sum, &prod);
                }
                let head_dim_id = Value::new(HEAD_DIM as f64);
                let sqrt_head_dim = pow(&head_dim_id, 0.5);
                let logit = mul(&sum, &pow(&sqrt_head_dim, -1.0));
                attn_logits.push(logit);
            }

            let attn_weights = softmax(&attn_logits);

            for j in 0..HEAD_DIM {
                let mut head_out = Value::new(0.0);
                for t in 0..v_h.len() {
                    let prod = mul(&attn_weights[t], &v_h[t][j]);
                    head_out = add(&head_out, &prod);
                }
                x_attn.push(head_out);
            }
        }

        x = linear(&x_attn, &state_dict[&format!("layer{}.attn_wo", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| add(a, b))
            .collect();

        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc1", li)]);
        x = x.iter().map(|xi| relu(xi)).collect();
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc2", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| add(a, b))
            .collect();
    }

    linear(&x, &state_dict["lm_head"])
}

fn collect_params(state_dict: &HashMap<String, Vec<Vec<ValueRef>>>) -> Vec<ValueRef> {
    let mut params = Vec::new();
    for key in [
        "wte", "wpe", "lm_head",
        "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo",
        "layer0.mlp_fc1", "layer0.mlp_fc2",
    ].iter() {
        if let Some(matrix) = state_dict.get(*key) {
            for row in matrix {
                for param in row {
                    params.push(param.clone());
                }
            }
        }
    }
    params
}

fn reset_all_grads(params: &[ValueRef]) {
    for param in params {
        reset_grad(param);
    }
}

fn count_graph_nodes(v: &ValueRef) -> usize {
    let mut visited = std::collections::HashSet::new();
    let mut count = 0;
    
    fn traverse(v: &ValueRef, visited: &mut std::collections::HashSet<usize>, count: &mut usize) {
        let ptr = v.as_ptr() as usize;
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            *count += 1;
            let v_borrow = v.borrow();
            let children = v_borrow.children.clone();
            drop(v_borrow);
            for (child, _) in children {
                traverse(&child, visited, count);
            }
        }
    }
    
    traverse(v, &mut visited, &mut count);
    count
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut steps, mut lr, mut n_emb, mut n_head, mut n_layer, mut n_ctx, mut n_ff) = (NUM_STEPS, LEARNING_RATE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, 4);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-s" | "--steps" => { i += 1; if i < args.len() { steps = args[i].parse().unwrap_or(steps); } }
            "-l" | "--lr" => { i += 1; if i < args.len() { lr = args[i].parse().unwrap_or(lr); } }
            "-e" | "--emb" => { i += 1; if i < args.len() { n_emb = args[i].parse().unwrap_or(n_emb); } }
            "-h" | "--head" => { i += 1; if i < args.len() { n_head = args[i].parse().unwrap_or(n_head); } }
            "-y" | "--layer" => { i += 1; if i < args.len() { n_layer = args[i].parse().unwrap_or(n_layer); } }
            "-c" | "--ctx" => { i += 1; if i < args.len() { n_ctx = args[i].parse().unwrap_or(n_ctx); } }
            "-b" | "--batch" => { i += 1; if i < args.len() { n_ff = args[i].parse().unwrap_or(n_ff); } }
            _ => {}
        }
        i += 1;
    }
    download_names().expect("Failed to download dataset");
    let docs = load_docs();
    println!("num docs: {}", docs.len());

    let (uchars, vocab_size) = build_vocab(&docs);
    println!("vocab size: {}", vocab_size);
    let bos = uchars.len();

    let mut rng = rand::thread_rng();

    let mut state_dict = HashMap::new();
    state_dict.insert("wte".to_string(), init_matrix(vocab_size, n_emb, 0.08, &mut rng));
    state_dict.insert("wpe".to_string(), init_matrix(n_ctx, n_emb, 0.08, &mut rng));
    state_dict.insert("lm_head".to_string(), init_matrix(vocab_size, n_emb, 0.08, &mut rng));

    for i in 0..n_layer {
        state_dict.insert(
            format!("layer{}.attn_wq", i),
            init_matrix(n_emb, n_emb, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.attn_wk", i),
            init_matrix(n_emb, n_emb, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.attn_wv", i),
            init_matrix(n_emb, n_emb, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.attn_wo", i),
            init_matrix(n_emb, n_emb, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.mlp_fc1", i),
            init_matrix(4 * n_emb, n_emb, 0.08, &mut rng),
        );
        state_dict.insert(
            format!("layer{}.mlp_fc2", i),
            init_matrix(n_emb, 4 * n_emb, 0.08, &mut rng),
        );
    }

    let params = collect_params(&state_dict);
    let num_params = params.len();
    println!("num params: {}", num_params);

    let mut m = vec![0.0; num_params];
    let mut v = vec![0.0; num_params];

    let mut total_forward = 0.0;
    let mut total_backward = 0.0;
    let mut total_softmax = 0.0;
    let mut total_linear = 0.0;
    let mut total_update = 0.0;
    let mut total_graph_size = 0usize;

    // Early stopping variables
    let mut best_loss = f64::INFINITY;
    let mut patience_counter = 0;
    let mut best_step = 0;

    for step in 0..steps {
        reset_all_grads(&params);

        // Mini-batch: process multiple documents per step
        let start_idx = (step * n_ff) % docs.len();
        let batch_docs: Vec<String> = (0..n_ff)
            .map(|i| docs[(start_idx + i) % docs.len()].clone())
            .collect();

        let mut batch_loss = Value::new(0.0);

        for doc in batch_docs {
            let mut tokens = vec![bos];
            for ch in doc.chars() {
                if let Some(idx) = uchars.iter().position(|&c| c == ch) {
                    tokens.push(idx);
                }
            }
            tokens.push(bos);

            let n = std::cmp::min(n_ctx, tokens.len() - 1);

            let mut keys = vec![vec![]; n_layer];
            let mut values = vec![vec![]; n_layer];
            let mut losses = Vec::new();

            let t_forward = Instant::now();
            for pos_id in 0..n {
                let token_id = tokens[pos_id];
                let target_id = tokens[pos_id + 1];

                let logits = gpt(token_id, pos_id, &mut keys, &mut values, &state_dict);
                let t_softmax = Instant::now();
                let probs = softmax(&logits);
                total_softmax += t_softmax.elapsed().as_secs_f64();
                
                let loss_t = log(&probs[target_id]);
                let neg_loss = mul(&loss_t, &Value::new(-1.0));
                losses.push(neg_loss);
            }
            total_forward += t_forward.elapsed().as_secs_f64();

            let mut doc_loss = Value::new(0.0);
            for loss_id in &losses {
                doc_loss = add(&doc_loss, loss_id);
            }
            let n_id = Value::new(n as f64);
            doc_loss = mul(&doc_loss, &pow(&n_id, -1.0));
            
            // Accumulate batch loss
            batch_loss = add(&batch_loss, &doc_loss);
        }

        // Average batch loss
        let batch_size_id = Value::new(n_ff as f64);
        batch_loss = mul(&batch_loss, &pow(&batch_size_id, -1.0));

        let loss_data = batch_loss.borrow().data;
        
        let t_backward = Instant::now();
        backward(&batch_loss);
        total_backward += t_backward.elapsed().as_secs_f64();

        // Cosine annealing learning rate schedule
        let cosine_factor = 0.5 * (1.0 + (step as f64 * PI / steps as f64).cos());
        let lr_t = MIN_LR + (lr - MIN_LR) * cosine_factor;

        let t_update = Instant::now();
        for (i, param) in params.iter().enumerate() {
            let grad = param.borrow().grad;
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad;
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad * grad;

            let m_hat = m[i] / (1.0 - BETA1.powi((step + 1) as i32));
            let v_hat = v[i] / (1.0 - BETA2.powi((step + 1) as i32));

            let update = lr_t * m_hat / (v_hat.sqrt() + EPS_ADAM);
            param.borrow_mut().data -= update;
        }
        total_update += t_update.elapsed().as_secs_f64();

        if step == 0 {
            total_graph_size = count_graph_nodes(&batch_loss);
        }

        // Early stopping logic
        if loss_data < best_loss {
            best_loss = loss_data;
            best_step = step;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        if (step + 1) % 100 == 0 || step == 0 {
            print!("step {:4} / {} | loss {:.4} | lr {:.6} | patience {}/{}\r", 
                   step + 1, steps, loss_data, lr_t, patience_counter, PATIENCE);
            std::io::stdout().flush().unwrap();
        }

        // Early stopping check
        if patience_counter >= PATIENCE {
            println!("\n--- Early Stopped ---");
            println!("No improvement for {} steps. Best loss: {:.4} at step {}", 
                     PATIENCE, best_loss, best_step + 1);
            break;
        }
    }

    println!("\n--- Profiling Results ---");
    println!("Forward pass:  {:.2}s ({:.1}%)", total_forward, total_forward / (total_forward + total_backward + total_update) * 100.0);
    println!("Backward pass: {:.2}s ({:.1}%)", total_backward, total_backward / (total_forward + total_backward + total_update) * 100.0);
    println!("Softmax:       {:.2}s ({:.1}%)", total_softmax, total_softmax / (total_forward + total_backward + total_update) * 100.0);
    println!("Param update:  {:.2}s ({:.1}%)", total_update, total_update / (total_forward + total_backward + total_update) * 100.0);
    println!("Avg graph size: {} nodes", total_graph_size);

    println!("\n--- inference (new, hallucinated names) ---");

    for sample_idx in 0..20 {
        let mut keys = vec![vec![]; n_layer];
        let mut values = vec![vec![]; n_layer];
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..n_ctx {
            let logits = gpt(token_id, pos_id, &mut keys, &mut values, &state_dict);
            let probs = softmax(&logits);
            let prob_vals: Vec<f64> = probs.iter().map(|p| p.borrow().data).collect();

            let mut cumsum = 0.0;
            let rand_val: f64 = rng.gen();
            token_id = 0;
            for (i, &p) in prob_vals.iter().enumerate() {
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
