/* 
   MicroGPT: The Art of Symmetry
   A minimal, aesthetic implementation of a Transformer in Rust.
*/

use rand::Rng;
use std::{cell::RefCell, collections::HashSet, ops::{Add, Mul, Neg, Sub}, rc::Rc, io::Write};

// --- I. The Atom: Value & Autograd ---

#[derive(Clone)]
struct Val(Rc<RefCell<Node>>);
struct Node { data: f64, grad: f64, prev: Vec<(Val, f64)> }

impl Val {
    fn new(data: f64) -> Self { Val(Rc::new(RefCell::new(Node { data, grad: 0., prev: vec![] }))) }
    fn data(&self) -> f64 { self.0.borrow().data }
    fn grad(&self) -> f64 { self.0.borrow().grad }
    fn zero(&self) { self.0.borrow_mut().grad = 0.; }

    fn backward(&self) {
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

    fn ptr(&self) -> usize { Rc::as_ptr(&self.0) as usize }
    
    fn pow(&self, p: f64) -> Val {
        let (d, i) = (self.data(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d.powf(p), grad: 0., prev: vec![(i, p * d.powf(p-1.))] })))
    }
    fn exp(&self) -> Val {
        let (d, i) = (self.data().exp(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d, grad: 0., prev: vec![(i, d)] })))
    }
    fn log(&self) -> Val {
        let (d, i) = (self.data(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d.ln(), grad: 0., prev: vec![(i, 1./d)] })))
    }
    fn relu(&self) -> Val {
        let (d, i) = (self.data(), self.clone());
        Val(Rc::new(RefCell::new(Node { data: d.max(0.), grad: 0., prev: vec![(i, if d>0.{1.}else{0.})] })))
    }
}

// --- II. The Algebra: Operators ---

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

type Vec1 = Vec<Val>;
type Mat2 = Vec<Vec1>;

fn mat(r: usize, c: usize) -> Mat2 {
    let mut rng = rand::thread_rng();
    (0..r).map(|_| (0..c).map(|_| Val::new(rng.gen_range(-1.0..1.0) * 0.2)).collect()).collect()
}
fn linear(x: &[Val], w: &Mat2) -> Vec1 {
    w.iter().map(|row| row.iter().zip(x).map(|(w, x)| w * x).fold(Val::new(0.), |a, b| a + b)).collect()
}
fn softmax(x: &[Val]) -> Vec1 {
    let max = x.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec1 = x.iter().map(|v| (v - &Val::new(max)).exp()).collect();
    let sum = exps.iter().fold(Val::new(0.), |a, b| a + b);
    let inv = sum.pow(-1.);
    exps.iter().map(|v| v * &inv).collect()
}
fn rmsnorm(x: &[Val]) -> Vec1 {
    let ss = x.iter().map(|v| v * v).fold(Val::new(0.), |a, b| a + b);
    let n = Val::new((x.len() as f64).recip());
    let scale = (ss * &n + &Val::new(1e-5)).pow(-0.5);
    x.iter().map(|v| v * &scale).collect()
}

struct GPT {
    wte: Mat2, wpe: Mat2, lm_head: Mat2,
    wq: Vec<Mat2>, wk: Vec<Mat2>, wv: Vec<Mat2>, wo: Vec<Mat2>,
    fc1: Vec<Mat2>, fc2: Vec<Mat2>,
}

impl GPT {
    fn new(v: usize, ctx: usize, d: usize, l: usize) -> Self {
        GPT {
            wte: mat(v, d), wpe: mat(ctx, d), lm_head: mat(v, d),
            wq: (0..l).map(|_| mat(d, d)).collect(), wk: (0..l).map(|_| mat(d, d)).collect(),
            wv: (0..l).map(|_| mat(d, d)).collect(), wo: (0..l).map(|_| mat(d, d)).collect(),
            fc1: (0..l).map(|_| mat(4*d, d)).collect(), fc2: (0..l).map(|_| mat(d, 4*d)).collect(),
        }
    }
    
    fn params(&self) -> Vec<Val> {
        let mut p = vec![];
        for m in [&self.wte, &self.wpe, &self.lm_head] { for r in m { p.extend(r.clone()); } }
        for ms in [&self.wq, &self.wk, &self.wv, &self.wo, &self.fc1, &self.fc2] {
            for m in ms { for r in m { p.extend(r.clone()); } }
        }
        p
    }

    fn forward(&self, t: usize, pos: usize, k: &mut Vec<Vec<Vec1>>, v: &mut Vec<Vec<Vec1>>) -> Vec1 {
        let mut x: Vec1 = self.wte[t].iter().zip(&self.wpe[pos]).map(|(t, p)| t + p).collect();
        let hd = x.len() / 4; 

        for i in 0..self.wq.len() {
            let xn = rmsnorm(&x);
            let q_vec = linear(&xn, &self.wq[i]);
            k[i].push(linear(&xn, &self.wk[i]));
            v[i].push(linear(&xn, &self.wv[i]));
            
            let mut att = vec![];
            for h in 0..4 {
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
            
            let xn = rmsnorm(&x);
            let h = linear(&xn, &self.fc1[i]).iter().map(|v| v.relu()).collect::<Vec1>();
            x = x.iter().zip(linear(&h, &self.fc2[i])).map(|(x, m)| x + m).collect();
        }
        linear(&rmsnorm(&x), &self.lm_head)
    }
}

// --- IV. The Training Loop ---

fn main() {
    if std::fs::metadata("input.txt").is_err() {
        let _ = std::process::Command::new("curl").args(["-o", "input.txt", "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"]).output();
    }
    let raw = std::fs::read_to_string("input.txt").unwrap_or_else(|_| "emma\nolivia\nava\n".to_string());
    let chars: Vec<char> = { let mut c: Vec<_> = raw.chars().collect::<HashSet<_>>().into_iter().filter(|c| !c.is_whitespace()).collect(); c.sort(); c };
    let vocab = chars.len() + 1;
    let (n_emb, n_ctx, n_layer) = (16, 16, 1);
    
    let model = GPT::new(vocab, n_ctx, n_emb, n_layer);
    let params = model.params();
    println!("MicroGPT: {} params", params.len());
    
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    let (steps, lr) = (200, 0.01);
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
            m[i] = 0.85 * m[i] + 0.15 * g;
            v[i] = 0.99 * v[i] + 0.01 * g * g;
            let m_hat = m[i] / (1. - 0.85f64.powi(step as i32 + 1));
            let v_hat = v[i] / (1. - 0.99f64.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + 1e-8);
        }
        if step % 20 == 0 { print!("step {:4} | loss {:.4}\r", step, loss_val); std::io::stdout().flush().unwrap(); }
    }
    
    println!("\n--- Generation ---");
    for _ in 0..5 {
        let (mut kc, mut vc) = (vec![vec![]; n_layer], vec![vec![]; n_layer]);
        let mut tok = vocab - 1;
        print!("> ");
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
