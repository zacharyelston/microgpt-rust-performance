/* 
   MicroGPT: The Art of Symmetry
   Shared Library
*/

use rand::Rng;
use std::{cell::RefCell, collections::HashSet, io::Write, ops::{Add, Mul, Neg, Sub}, rc::Rc};

// --- I. The Atom: Value & Autograd ---

#[derive(Clone)]
pub struct Val(pub Rc<RefCell<Node>>);
pub struct Node { pub data: f64, pub grad: f64, pub prev: Vec<(Val, f64)> }

impl Val {
    pub fn new(data: f64) -> Self { Val(Rc::new(RefCell::new(Node { data, grad: 0., prev: vec![] }))) }
    pub fn data(&self) -> f64 { self.0.borrow().data }
    pub fn grad(&self) -> f64 { self.0.borrow().grad }
    pub fn zero(&self) { self.0.borrow_mut().grad = 0.; }

    pub fn backward(&self) {
        let mut order = vec![];
        let mut visited = HashSet::new();
        fn build(v: &Val, vis: &mut HashSet<usize>, ord: &mut Vec<Val>) {
            if vis.insert(v.ptr()) {
                for (child, _) in &v.0.borrow().prev { build(child, vis, ord); }
                ord.push(v.clone());
            }
        }
        build(self, &mut visited, &mut order);
        self.0.borrow_mut().grad = 1.0;
        for v in order.iter().rev() {
            let n = v.0.borrow();
            let g = n.grad;
            for (child, local) in &n.prev { child.0.borrow_mut().grad += local * g; }
        }
    }

    pub fn ptr(&self) -> usize { Rc::as_ptr(&self.0) as usize }
    
    pub fn pow(&self, p: f64) -> Val {
        let (d, i) = (self.data(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d.powf(p), grad: 0., prev: vec![(i, p * d.powf(p-1.))] })))
    }
    pub fn exp(&self) -> Val {
        let (d, i) = (self.data().exp(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d, grad: 0., prev: vec![(i, d)] })))
    }
    pub fn log(&self) -> Val {
        let (d, i) = (self.data(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d.ln(), grad: 0., prev: vec![(i, 1./d)] })))
    }
    pub fn relu(&self) -> Val {
        let (d, i) = (self.data(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d.max(0.), grad: 0., prev: vec![(i, if d>0.{1.}else{0.})] })))
    }
}

// --- II. The Algebra: Operators ---

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

op!(Add, add, +, |_,_| 1., |_,_| 1.);
op!(Sub, sub, -, |_,_| 1., |_,_| -1.);
op!(Mul, mul, *, |_,o: &Val| o.data(), |s: &Val,_| s.data());

impl Neg for &Val { type Output = Val; fn neg(self) -> Val { self * &Val::new(-1.) } }
impl Neg for Val { type Output = Val; fn neg(self) -> Val { &self * &Val::new(-1.) } }

// --- III. The Architecture: GPT ---

pub type Vec1 = Vec<Val>;
pub type Mat2 = Vec<Vec1>;

pub fn mat(r: usize, c: usize, scale: f64) -> Mat2 {
    let mut rng = rand::thread_rng();
    (0..r).map(|_| (0..c).map(|_| Val::new(rng.gen_range(-1.0..1.0) * scale)).collect()).collect()
}

pub fn linear(x: &[Val], w: &Mat2) -> Vec1 {
    w.iter().map(|row| row.iter().zip(x).map(|(w, x)| w * x).fold(Val::new(0.), |a, b| a + b)).collect()
}

pub fn softmax(x: &[Val]) -> Vec1 {
    let max = x.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec1 = x.iter().map(|v| (v - &Val::new(max)).exp()).collect();
    let sum = exps.iter().fold(Val::new(0.), |a, b| a + b);
    let inv = sum.pow(-1.);
    exps.iter().map(|v| v * &inv).collect()
}

pub fn rmsnorm(x: &[Val], eps: f64) -> Vec1 {
    let ss = x.iter().map(|v| v * v).fold(Val::new(0.), |a, b| a + b);
    let n = Val::new((x.len() as f64).recip());
    let scale = (ss * &n + &Val::new(eps)).pow(-0.5);
    x.iter().map(|v| v * &scale).collect()
}

#[allow(clippy::upper_case_acronyms)]
pub struct GPT {
    pub wte: Mat2, pub wpe: Mat2, pub lm_head: Mat2,
    pub wq: Vec<Mat2>, pub wk: Vec<Mat2>, pub wv: Vec<Mat2>, pub wo: Vec<Mat2>,
    pub fc1: Vec<Mat2>, pub fc2: Vec<Mat2>,
    pub n_head: usize,
    pub rms_eps: f64,
}

impl GPT {
    pub fn new(v: usize, ctx: usize, d: usize, l: usize, h: usize, ff: usize, init_scale: f64, rms_eps: f64) -> Self {
        GPT {
            wte: mat(v, d, init_scale), wpe: mat(ctx, d, init_scale), lm_head: mat(v, d, init_scale),
            wq: (0..l).map(|_| mat(d, d, init_scale)).collect(), wk: (0..l).map(|_| mat(d, d, init_scale)).collect(),
            wv: (0..l).map(|_| mat(d, d, init_scale)).collect(), wo: (0..l).map(|_| mat(d, d, init_scale)).collect(),
            fc1: (0..l).map(|_| mat(ff*d, d, init_scale)).collect(), fc2: (0..l).map(|_| mat(d, ff*d, init_scale)).collect(),
            n_head: h,
            rms_eps,
        }
    }
    
    pub fn params(&self) -> Vec<Val> {
        let mut p = vec![];
        for m in [&self.wte, &self.wpe, &self.lm_head] { for r in m { p.extend(r.clone()); } }
        for ms in [&self.wq, &self.wk, &self.wv, &self.wo, &self.fc1, &self.fc2] {
            for m in ms { for r in m { p.extend(r.clone()); } }
        }
        p
    }

    pub fn forward(&self, t: usize, pos: usize, k: &mut [Vec<Vec1>], v: &mut [Vec<Vec1>]) -> Vec1 {
        let mut x: Vec1 = self.wte[t].iter().zip(&self.wpe[pos]).map(|(t, p)| t + p).collect();
        let hd = x.len() / self.n_head; 

        for i in 0..self.wq.len() {
            let xn = rmsnorm(&x, self.rms_eps);
            let q_vec = linear(&xn, &self.wq[i]);
            k[i].push(linear(&xn, &self.wk[i]));
            v[i].push(linear(&xn, &self.wv[i]));
            
            let mut att = vec![];
            for h in 0..self.n_head {
                let rng = h*hd..(h+1)*hd;
                let q_h = &q_vec[rng.clone()];
                let scale = Val::new((hd as f64).sqrt().recip());
                
                let scores: Vec1 = k[i].iter().map(|k_t| 
                    q_h.iter().zip(&k_t[rng.clone()]).map(|(q, k)| q * k).fold(Val::new(0.), |a, b| a + b) * &scale
                ).collect();
                
                let w = softmax(&scores);
                let mut out = (0..hd).map(|_| Val::new(0.)).collect::<Vec1>();
                for (t, wt) in w.iter().enumerate() {
                    let v_h = &v[i][t][rng.clone()];
                    for (j, val) in v_h.iter().enumerate() { out[j] = &out[j] + &(wt * val); }
                }
                att.extend(out);
            }
            x = x.iter().zip(linear(&att, &self.wo[i])).map(|(x, a)| x + a).collect();
            
            let xn = rmsnorm(&x, self.rms_eps);
            let h = linear(&xn, &self.fc1[i]).iter().map(|v| v.relu()).collect::<Vec1>();
            x = x.iter().zip(linear(&h, &self.fc2[i])).map(|(x, m)| x + m).collect();
        }
        linear(&rmsnorm(&x, self.rms_eps), &self.lm_head)
    }
}

// --- IV. Training & Generation ---

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub n_emb: usize,
    pub n_ctx: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff_exp: usize,
    pub steps: usize,
    pub lr: f64,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub adam_eps: f64,
    pub gen_samples: usize,
    pub rms_eps: f64,
    pub init_scale: f64,
    pub input_file: String,
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            n_emb: 16, n_ctx: 16, n_layer: 1, n_head: 4, n_ff_exp: 4,
            steps: 200, lr: 0.005,
            adam_beta1: 0.85, adam_beta2: 0.99, adam_eps: 1e-8,
            gen_samples: 5, rms_eps: 1e-5, init_scale: 0.1,
            input_file: "input.txt".to_string(),
            checkpoint_interval: 20,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub names: Vec<String>,
    pub final_loss: f64,
    pub num_params: usize,
}

pub fn load_training_data(input_file: &str) -> String {
    std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        panic!("Failed to read training data from '{}': {}", input_file, e);
    })
}

pub fn build_vocab(raw: &str) -> Vec<char> {
    let mut c: Vec<_> = raw.chars().collect::<HashSet<_>>().into_iter().filter(|c| !c.is_whitespace()).collect();
    c.sort();
    c
}

pub fn train_and_generate(cfg: &TrainingConfig, silent: bool) -> TrainingResult {
    let raw = load_training_data(&cfg.input_file);
    let chars = build_vocab(&raw);
    let vocab = chars.len() + 1;
    
    let model = GPT::new(vocab, cfg.n_ctx, cfg.n_emb, cfg.n_layer, cfg.n_head, cfg.n_ff_exp, cfg.init_scale, cfg.rms_eps);
    let params = model.params();
    let num_params = params.len();
    
    if !silent {
        println!("MicroGPT: {} params, training for {} steps (lr={}, emb={}, head={}, layer={}, ctx={}, ff={})",
            num_params, cfg.steps, cfg.lr, cfg.n_emb, cfg.n_head, cfg.n_layer, cfg.n_ctx, cfg.n_ff_exp);
    }
    
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    let docs: Vec<&str> = raw.lines().collect();
    let mut final_loss = 0.0;
    
    for step in 0..cfg.steps {
        let doc = docs[step % docs.len()];
        let mut tokens: Vec<usize> = std::iter::once(vocab-1)
            .chain(doc.chars().map(|c| chars.iter().position(|&x| x == c).unwrap()))
            .chain(std::iter::once(vocab-1)).collect();
        if tokens.len() > cfg.n_ctx {
            tokens.truncate(cfg.n_ctx);
        }

        let mut loss = Val::new(0.);
        let (mut kc, mut vc) = (vec![vec![]; cfg.n_layer], vec![vec![]; cfg.n_layer]);
        
        for p in 0..tokens.len()-1 {
            let logits = model.forward(tokens[p], p, &mut kc, &mut vc);
            let probs = softmax(&logits);
            loss = &loss - &probs[tokens[p+1]].log();
        }
        loss = &loss * &Val::new((tokens.len() as f64 - 1.).recip());
        final_loss = loss.data();
        
        for p in &params { p.zero(); }
        loss.backward();

        let lr_t = cfg.lr * (1. - step as f64 / cfg.steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m[i] = cfg.adam_beta1 * m[i] + (1. - cfg.adam_beta1) * g;
            v[i] = cfg.adam_beta2 * v[i] + (1. - cfg.adam_beta2) * g * g;
            let m_hat = m[i] / (1. - cfg.adam_beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1. - cfg.adam_beta2.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + cfg.adam_eps);
        }

        if !silent && step % cfg.checkpoint_interval == 0 {
            print!("step {:4} | loss {:.4}\r", step, final_loss);
            std::io::stdout().flush().unwrap();
        }
    }
    
    if !silent { println!("\n--- Generation ---"); }

    let mut results = Vec::new();
    for _ in 0..cfg.gen_samples {
        let (mut kc, mut vc) = (vec![vec![]; cfg.n_layer], vec![vec![]; cfg.n_layer]);
        let mut tok = vocab - 1;
        let mut name = String::new();
        
        for p in 0..cfg.n_ctx {
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
            name.push(chars[tok]);
        }
        if !silent { println!("> {}", name); }
        results.push(name);
    }

    TrainingResult { names: results, final_loss, num_params }
}
