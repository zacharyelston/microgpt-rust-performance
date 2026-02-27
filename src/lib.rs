/*
    MicroGPT: A Living Transformer — Shared Library

    This file contains the entire neural network stack, built from scratch:

      I.   Val & Autograd — scalar-level automatic differentiation
      II.  Operators — arithmetic with gradient tracking via macro
      III. GPT Model — embeddings, attention, MLP, normalization
      IV.  Training & Generation — training loop, Adam optimizer, sampling
      V.   Genome I/O — save/load evolved hyperparameters to genome.json

    No ML frameworks. Every gradient is computed by hand through the
    computation graph. This is the organism's body — the evolution
    engine shapes it by choosing what to build.
*/

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    cell::RefCell,
    collections::HashSet,
    io::Write,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

// ============================================================
// I. The Atom: Value & Autograd
//
// A Val wraps a scalar with its gradient and a record of how it
// was produced (the "tape"). Every arithmetic operation on Vals
// builds a directed acyclic graph. Calling backward() on a loss
// node walks this graph in reverse topological order, accumulating
// gradients via the chain rule.
// ============================================================

#[derive(Clone)]
pub struct Val(pub Rc<RefCell<Node>>);

pub struct Node {
    pub data: f64,
    pub grad: f64,
    // Each entry: (parent Val, local derivative d(self)/d(parent))
    pub prev: Vec<(Val, f64)>,
}

impl Val {
    pub fn new(data: f64) -> Self {
        Val(Rc::new(RefCell::new(Node {
            data,
            grad: 0.,
            prev: vec![],
        })))
    }
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn zero(&self) {
        self.0.borrow_mut().grad = 0.;
    }

    // Reverse-mode automatic differentiation.
    // Topologically sorts the graph, then propagates gradients backward.
    pub fn backward(&self) {
        let mut order = vec![];
        let mut visited = HashSet::new();
        fn build(v: &Val, vis: &mut HashSet<usize>, ord: &mut Vec<Val>) {
            if vis.insert(v.ptr()) {
                for (child, _) in &v.0.borrow().prev {
                    build(child, vis, ord);
                }
                ord.push(v.clone());
            }
        }
        build(self, &mut visited, &mut order);
        self.0.borrow_mut().grad = 1.0; // d(loss)/d(loss) = 1
        for v in order.iter().rev() {
            let n = v.0.borrow();
            let g = n.grad;
            // Chain rule: propagate gradient to each parent
            for (child, local) in &n.prev {
                child.0.borrow_mut().grad += local * g;
            }
        }
    }

    // Unique identity for deduplication in topological sort
    pub fn ptr(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    // Unary operations — each records the local derivative for backprop
    pub fn pow(&self, p: f64) -> Val {
        let (d, i) = (self.data(), self.clone());
        // d/dx(x^p) = p * x^(p-1)
        Val(Rc::new(RefCell::new(Node {
            data: d.powf(p),
            grad: 0.,
            prev: vec![(i, p * d.powf(p - 1.))],
        })))
    }
    pub fn exp(&self) -> Val {
        let (d, i) = (self.data().exp(), self.clone());
        // d/dx(e^x) = e^x
        Val(Rc::new(RefCell::new(Node {
            data: d,
            grad: 0.,
            prev: vec![(i, d)],
        })))
    }
    pub fn log(&self) -> Val {
        let (d, i) = (self.data(), self.clone());
        // d/dx(ln(x)) = 1/x
        Val(Rc::new(RefCell::new(Node {
            data: d.ln(),
            grad: 0.,
            prev: vec![(i, 1. / d)],
        })))
    }
    pub fn relu(&self) -> Val {
        let (d, i) = (self.data(), self.clone());
        // d/dx(relu(x)) = 1 if x > 0, else 0
        Val(Rc::new(RefCell::new(Node {
            data: d.max(0.),
            grad: 0.,
            prev: vec![(i, if d > 0. { 1. } else { 0. })],
        })))
    }
}

// ============================================================
// II. The Algebra: Operators
//
// The op! macro defines binary operations for all combinations
// of Val and &Val (owned × owned, owned × ref, ref × ref, etc).
// Each operation records both operands and their local derivatives
// into the computation graph.
// ============================================================

#[macro_export]
macro_rules! op {
    ($t:ident, $f:ident, $op:tt, $dg_self:expr, $dg_other:expr) => {
        impl $t<&Val> for &Val { type Output = Val;
            fn $f(self, o: &Val) -> Val {
                Val(Rc::new(RefCell::new(Node {
                    data: self.data() $op o.data(), grad: 0.,
                    prev: vec![(self.clone(), $dg_self(self, o)), (o.clone(), $dg_other(self, o))]
                })))
            }
        }
        impl $t<Val> for Val { type Output = Val; fn $f(self, o: Val) -> Val { &self $op &o } }
        impl $t<&Val> for Val { type Output = Val; fn $f(self, o: &Val) -> Val { &self $op o } }
        impl $t<Val> for &Val { type Output = Val; fn $f(self, o: Val) -> Val { self $op &o } }
    };
}

// d/da(a+b) = 1, d/db(a+b) = 1
op!(Add, add, +, |_,_| 1., |_,_| 1.);
// d/da(a-b) = 1, d/db(a-b) = -1
op!(Sub, sub, -, |_,_| 1., |_,_| -1.);
// d/da(a*b) = b, d/db(a*b) = a
op!(Mul, mul, *, |_,o: &Val| o.data(), |s: &Val,_| s.data());

impl Neg for &Val {
    type Output = Val;
    fn neg(self) -> Val {
        self * &Val::new(-1.)
    }
}
impl Neg for Val {
    type Output = Val;
    fn neg(self) -> Val {
        &self * &Val::new(-1.)
    }
}

// ============================================================
// III. The Architecture: GPT
//
// A minimal transformer: token embeddings + positional embeddings,
// N layers of (multi-head attention + feed-forward MLP), then a
// linear projection to vocabulary logits. Uses RMSNorm (no bias
// terms) and causal KV caching for autoregressive generation.
// ============================================================

pub type Vec1 = Vec<Val>;
pub type Mat2 = Vec<Vec1>;

// Initialize a matrix with random values scaled by `scale`
pub fn mat(r: usize, c: usize, scale: f64, rng: &mut StdRng) -> Mat2 {
    (0..r)
        .map(|_| {
            (0..c)
                .map(|_| Val::new(rng.gen_range(-1.0..1.0) * scale))
                .collect()
        })
        .collect()
}

// Matrix-vector multiply: each row of w dotted with x
pub fn linear(x: &[Val], w: &Mat2) -> Vec1 {
    w.iter()
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(w, x)| w * x)
                .fold(Val::new(0.), |a, b| a + b)
        })
        .collect()
}

// Numerically stable softmax: subtract max before exp to prevent overflow
pub fn softmax(x: &[Val]) -> Vec1 {
    let max = x.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec1 = x.iter().map(|v| (v - &Val::new(max)).exp()).collect();
    let sum = exps.iter().fold(Val::new(0.), |a, b| a + b);
    let inv = sum.pow(-1.);
    exps.iter().map(|v| v * &inv).collect()
}

// Root Mean Square Normalization (no learned scale/bias)
pub fn rmsnorm(x: &[Val], eps: f64) -> Vec1 {
    let ss = x.iter().map(|v| v * v).fold(Val::new(0.), |a, b| a + b);
    let n = Val::new((x.len() as f64).recip());
    let scale = (ss * &n + &Val::new(eps)).pow(-0.5);
    x.iter().map(|v| v * &scale).collect()
}

#[allow(clippy::upper_case_acronyms)]
pub struct GPT {
    pub wte: Mat2,      // Token embeddings: [vocab × d_model]
    pub wpe: Mat2,      // Positional embeddings: [context × d_model]
    pub lm_head: Mat2,  // Output projection: [vocab × d_model]
    pub wq: Vec<Mat2>,  // Query projections: per-layer [d_model × d_model]
    pub wk: Vec<Mat2>,  // Key projections
    pub wv: Vec<Mat2>,  // Value projections
    pub wo: Vec<Mat2>,  // Output projections (after attention)
    pub fc1: Vec<Mat2>, // MLP first layer: [ff_dim × d_model]
    pub fc2: Vec<Mat2>, // MLP second layer: [d_model × ff_dim]
    pub n_head: usize,
    pub rms_eps: f64,
}

impl GPT {
    pub fn new(
        v: usize,
        ctx: usize,
        d: usize,
        l: usize,
        h: usize,
        ff: usize,
        init_scale: f64,
        rms_eps: f64,
        rng: &mut StdRng,
    ) -> Self {
        GPT {
            wte: mat(v, d, init_scale, rng),
            wpe: mat(ctx, d, init_scale, rng),
            lm_head: mat(v, d, init_scale, rng),
            wq: (0..l).map(|_| mat(d, d, init_scale, rng)).collect(),
            wk: (0..l).map(|_| mat(d, d, init_scale, rng)).collect(),
            wv: (0..l).map(|_| mat(d, d, init_scale, rng)).collect(),
            wo: (0..l).map(|_| mat(d, d, init_scale, rng)).collect(),
            fc1: (0..l).map(|_| mat(ff * d, d, init_scale, rng)).collect(),
            fc2: (0..l).map(|_| mat(d, ff * d, init_scale, rng)).collect(),
            n_head: h,
            rms_eps,
        }
    }

    // Collect all trainable parameters into a flat vector for the optimizer
    pub fn params(&self) -> Vec<Val> {
        let mut p = vec![];
        for m in [&self.wte, &self.wpe, &self.lm_head] {
            for r in m {
                p.extend(r.clone());
            }
        }
        for ms in [&self.wq, &self.wk, &self.wv, &self.wo, &self.fc1, &self.fc2] {
            for m in ms {
                for r in m {
                    p.extend(r.clone());
                }
            }
        }
        p
    }

    // Forward pass for a single token at a given position.
    // Uses causal KV cache: k and v accumulate across positions,
    // so attention naturally only sees past tokens.
    pub fn forward(&self, t: usize, pos: usize, k: &mut [Vec<Vec1>], v: &mut [Vec<Vec1>]) -> Vec1 {
        // Combine token embedding with positional embedding
        let mut x: Vec1 = self.wte[t]
            .iter()
            .zip(&self.wpe[pos])
            .map(|(t, p)| t + p)
            .collect();
        let hd = x.len() / self.n_head; // Dimension per head

        for i in 0..self.wq.len() {
            // Pre-norm: RMSNorm before attention
            let xn = rmsnorm(&x, self.rms_eps);
            let q_vec = linear(&xn, &self.wq[i]);
            k[i].push(linear(&xn, &self.wk[i]));
            v[i].push(linear(&xn, &self.wv[i]));

            // Multi-head attention: split Q/K/V into heads, compute scaled dot-product
            let mut att = vec![];
            for h in 0..self.n_head {
                let rng = h * hd..(h + 1) * hd;
                let q_h = &q_vec[rng.clone()];
                let scale = Val::new((hd as f64).sqrt().recip());

                // Attention scores: dot(q, each cached k) / sqrt(d_head)
                let scores: Vec1 = k[i]
                    .iter()
                    .map(|k_t| {
                        q_h.iter()
                            .zip(&k_t[rng.clone()])
                            .map(|(q, k)| q * k)
                            .fold(Val::new(0.), |a, b| a + b)
                            * &scale
                    })
                    .collect();

                // Weighted sum of cached values
                let w = softmax(&scores);
                let mut out = (0..hd).map(|_| Val::new(0.)).collect::<Vec1>();
                for (t, wt) in w.iter().enumerate() {
                    let v_h = &v[i][t][rng.clone()];
                    for (j, val) in v_h.iter().enumerate() {
                        out[j] = &out[j] + &(wt * val);
                    }
                }
                att.extend(out);
            }
            // Residual connection: x = x + Wo @ attention_output
            x = x
                .iter()
                .zip(linear(&att, &self.wo[i]))
                .map(|(x, a)| x + a)
                .collect();

            // Feed-forward MLP with residual: x = x + W2 @ relu(W1 @ norm(x))
            let xn = rmsnorm(&x, self.rms_eps);
            let h = linear(&xn, &self.fc1[i])
                .iter()
                .map(|v| v.relu())
                .collect::<Vec1>();
            x = x
                .iter()
                .zip(linear(&h, &self.fc2[i]))
                .map(|(x, m)| x + m)
                .collect();
        }
        // Final norm + project to vocabulary logits
        linear(&rmsnorm(&x, self.rms_eps), &self.lm_head)
    }
}

// ============================================================
// IV. Training & Generation
//
// The training loop processes one name per step: tokenize it,
// run each token through the model, compute cross-entropy loss,
// backpropagate, and update with Adam. Generation samples from
// the trained model autoregressively until the end token.
// ============================================================

const GENOME_FILE: &str = "genome.json";

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub n_emb: usize,       // Embedding dimension (d_model)
    pub n_ctx: usize,       // Maximum context / sequence length
    pub n_layer: usize,     // Number of transformer layers
    pub n_head: usize,      // Number of attention heads
    pub n_ff_exp: usize,    // Feed-forward expansion factor (ff_dim = n_ff_exp * n_emb)
    pub steps: usize,       // Number of training steps
    pub lr: f64,            // Peak learning rate
    pub adam_beta1: f64,    // Adam first moment decay
    pub adam_beta2: f64,    // Adam second moment decay
    pub adam_eps: f64,      // Adam epsilon for numerical stability
    pub gen_samples: usize, // Number of names to generate after training
    pub rms_eps: f64,       // RMSNorm epsilon
    pub init_scale: f64,    // Weight initialization scale
    pub input_file: String, // Path to training data
    pub checkpoint_interval: usize,
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
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
            gen_samples: 5,
            rms_eps: 1e-5,
            init_scale: 0.1,
            input_file: "input.txt".to_string(),
            checkpoint_interval: 20,
            seed: None,
        }
    }
}

// ============================================================
// V. Genome I/O — Self-Modification
//
// The evolution engine writes the winning hyperparameters to
// genome.json. The main binary reads it on startup. This is
// how the program evolves itself: evolution discovers better
// DNA, writes it to disk, and the organism becomes something new.
// ============================================================

impl TrainingConfig {
    // Serialize the evolved hyperparameters to genome.json
    pub fn save_genome(&self, loss: f64, generation: usize) -> std::io::Result<()> {
        let json = format!(
            "{{\n  \"n_emb\": {},\n  \"n_ctx\": {},\n  \"n_layer\": {},\n  \"n_head\": {},\n  \"n_ff_exp\": {},\n  \"steps\": {},\n  \"lr\": {},\n  \"loss\": {},\n  \"generation\": {},\n  \"evolved\": true\n}}",
            self.n_emb, self.n_ctx, self.n_layer, self.n_head, self.n_ff_exp,
            self.steps, self.lr, loss, generation
        );
        std::fs::write(GENOME_FILE, json)
    }

    // Load evolved hyperparameters from genome.json.
    // Returns None if the file doesn't exist (primordial state).
    pub fn load_genome() -> Option<(TrainingConfig, f64, usize)> {
        let data = std::fs::read_to_string(GENOME_FILE).ok()?;
        let mut cfg = TrainingConfig::default();
        let mut loss = 0.0;
        let mut gen = 0;
        for line in data.lines() {
            let line = line.trim().trim_end_matches(',');
            if let Some((key, val)) = line.split_once(':') {
                let key = key.trim().trim_matches('"');
                let val = val.trim();
                match key {
                    "n_emb" => cfg.n_emb = val.parse().unwrap_or(cfg.n_emb),
                    "n_ctx" => cfg.n_ctx = val.parse().unwrap_or(cfg.n_ctx),
                    "n_layer" => cfg.n_layer = val.parse().unwrap_or(cfg.n_layer),
                    "n_head" => cfg.n_head = val.parse().unwrap_or(cfg.n_head),
                    "n_ff_exp" => cfg.n_ff_exp = val.parse().unwrap_or(cfg.n_ff_exp),
                    "steps" => cfg.steps = val.parse().unwrap_or(cfg.steps),
                    "lr" => cfg.lr = val.parse().unwrap_or(cfg.lr),
                    "loss" => loss = val.parse().unwrap_or(0.0),
                    "generation" => gen = val.parse().unwrap_or(0),
                    _ => {}
                }
            }
        }
        Some((cfg, loss, gen))
    }
}

#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub names: Vec<String>,
    pub final_loss: f64,
    pub num_params: usize,
}

// Read the training dataset (one name per line)
pub fn load_training_data(input_file: &str) -> String {
    std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        panic!("Failed to read training data from '{}': {}", input_file, e);
    })
}

// Extract unique characters from the dataset to build the vocabulary.
// Token 0..N-1 = characters, token N = start/end delimiter.
pub fn build_vocab(raw: &str) -> Vec<char> {
    let mut c: Vec<_> = raw
        .chars()
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|c| !c.is_whitespace())
        .collect();
    c.sort();
    c
}

// Main training and generation function used by all binaries.
// Trains the model on the dataset, then generates sample names.
pub fn train_and_generate(cfg: &TrainingConfig, silent: bool) -> TrainingResult {
    let raw = load_training_data(&cfg.input_file);
    let chars = build_vocab(&raw);
    let vocab = chars.len() + 1; // +1 for the start/end delimiter token

    let mut rng = StdRng::seed_from_u64(cfg.seed.unwrap_or_else(|| rand::thread_rng().gen()));
    let model = GPT::new(
        vocab,
        cfg.n_ctx,
        cfg.n_emb,
        cfg.n_layer,
        cfg.n_head,
        cfg.n_ff_exp,
        cfg.init_scale,
        cfg.rms_eps,
        &mut rng,
    );
    let params = model.params();
    let num_params = params.len();

    if !silent {
        println!("MicroGPT: {} params, training for {} steps (lr={}, emb={}, head={}, layer={}, ctx={}, ff={})",
            num_params, cfg.steps, cfg.lr, cfg.n_emb, cfg.n_head, cfg.n_layer, cfg.n_ctx, cfg.n_ff_exp);
    }

    // Adam optimizer state: first moment (m) and second moment (v)
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    let docs: Vec<&str> = raw.lines().collect();
    let mut final_loss = 0.0;

    for step in 0..cfg.steps {
        // Tokenize one name: [START, char1, char2, ..., END]
        let doc = docs[step % docs.len()];
        let mut tokens: Vec<usize> = std::iter::once(vocab - 1)
            .chain(
                doc.chars()
                    .map(|c| chars.iter().position(|&x| x == c).unwrap()),
            )
            .chain(std::iter::once(vocab - 1))
            .collect();
        // Truncate to context window to prevent positional embedding overflow
        if tokens.len() > cfg.n_ctx {
            tokens.truncate(cfg.n_ctx);
        }

        // Forward pass: compute cross-entropy loss over all next-token predictions
        let mut loss = Val::new(0.);
        let (mut kc, mut vc) = (vec![vec![]; cfg.n_layer], vec![vec![]; cfg.n_layer]);

        for p in 0..tokens.len() - 1 {
            let logits = model.forward(tokens[p], p, &mut kc, &mut vc);
            let probs = softmax(&logits);
            loss = &loss - &probs[tokens[p + 1]].log(); // Negative log-likelihood
        }
        loss = &loss * &Val::new((tokens.len() as f64 - 1.).recip()); // Average over sequence
        final_loss = loss.data();

        // Backward pass: compute gradients
        for p in &params {
            p.zero();
        }
        loss.backward();

        // Adam update with linear learning rate decay
        let lr_t = cfg.lr * (1. - step as f64 / cfg.steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m[i] = cfg.adam_beta1 * m[i] + (1. - cfg.adam_beta1) * g;
            v[i] = cfg.adam_beta2 * v[i] + (1. - cfg.adam_beta2) * g * g;
            let m_hat = m[i] / (1. - cfg.adam_beta1.powi(step as i32 + 1)); // Bias correction
            let v_hat = v[i] / (1. - cfg.adam_beta2.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + cfg.adam_eps);
        }

        if !silent && step % cfg.checkpoint_interval == 0 {
            print!("step {:4} | loss {:.4}\r", step, final_loss);
            std::io::stdout().flush().unwrap();
        }
    }

    if !silent {
        println!("\n--- Generation ---");
    }

    // Autoregressive generation: sample tokens until END or context limit
    let mut results = Vec::new();
    for _ in 0..cfg.gen_samples {
        let (mut kc, mut vc) = (vec![vec![]; cfg.n_layer], vec![vec![]; cfg.n_layer]);
        let mut tok = vocab - 1; // Start with delimiter token
        let mut name = String::new();

        for p in 0..cfg.n_ctx {
            let logits = model.forward(tok, p, &mut kc, &mut vc);
            let probs = softmax(&logits);
            // Sample from the probability distribution
            let mut c = 0.;
            let r: f64 = rng.gen();
            let mut next = vocab - 1;
            for (i, v) in probs.iter().enumerate() {
                c += v.data();
                if r < c {
                    next = i;
                    break;
                }
            }
            tok = next;
            if tok == vocab - 1 {
                break;
            } // END token — name is complete
            name.push(chars[tok]);
        }
        if !silent {
            println!("> {}", name);
        }
        results.push(name);
    }

    TrainingResult {
        names: results,
        final_loss,
        num_params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_roundtrip_json() {
        let tmp = std::env::temp_dir().join(format!("microgpt_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&tmp).unwrap();

        let cfg = TrainingConfig {
            n_emb: 24,
            n_ctx: 12,
            n_layer: 2,
            n_head: 4,
            n_ff_exp: 3,
            steps: 321,
            lr: 0.0042,
            ..Default::default()
        };
        cfg.save_genome(1.23, 7).unwrap();
        let (loaded, loss, gen) = TrainingConfig::load_genome().unwrap();

        assert_eq!(loaded.n_emb, 24);
        assert_eq!(loaded.n_ctx, 12);
        assert_eq!(loaded.n_layer, 2);
        assert_eq!(loaded.n_head, 4);
        assert_eq!(loaded.n_ff_exp, 3);
        assert_eq!(loaded.steps, 321);
        assert!((loaded.lr - 0.0042).abs() < 1e-12);
        assert!((loss - 1.23).abs() < 1e-12);
        assert_eq!(gen, 7);

        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
