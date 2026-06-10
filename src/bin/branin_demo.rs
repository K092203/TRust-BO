// Tandem Residual-GP 動作検証デモ (Branin, 2D)。
// GP は本番モジュール src/gp.rs を #[path] で取り込み、同一コードを検証する。
#[path = "../gp.rs"]
mod gp;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

fn randn(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    let y = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}
fn normal_cdf(z: f64) -> f64 { 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2)) }
fn normal_pdf(z: f64) -> f64 { (-0.5 * z * z).exp() / (2.0 * PI).sqrt() }

/// EI (最小化形式): (f_best - mu - xi)Φ(z) + σφ(z)
fn ei_min(mu: f64, sigma: f64, f_best: f64, xi: f64) -> f64 {
    if sigma < 1e-12 { return 0.0; }
    let z = (f_best - mu - xi) / sigma;
    (f_best - mu - xi) * normal_cdf(z) + sigma * normal_pdf(z)
}

// ── 手書き tanh MLP (demo 専用、burn 不使用) ─────────────────────────────────
struct Mlp {
    sizes: Vec<usize>,
    w: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<f64>>,
    mw: Vec<Vec<Vec<f64>>>, vw: Vec<Vec<Vec<f64>>>,
    mb: Vec<Vec<f64>>, vb: Vec<Vec<f64>>,
    t: i32,
}
impl Mlp {
    fn new(sizes: Vec<usize>, rng: &mut StdRng) -> Self {
        let nl = sizes.len() - 1;
        let mut w = Vec::with_capacity(nl);
        let mut b = Vec::with_capacity(nl);
        for l in 0..nl {
            let (ni, no) = (sizes[l], sizes[l + 1]);
            let bound = 1.0 / (ni as f64).sqrt();
            let mut wl = vec![vec![0.0; ni]; no];
            for o in 0..no { for i in 0..ni { wl[o][i] = rng.gen_range(-bound..bound); } }
            w.push(wl);
            b.push(vec![0.0; no]);
        }
        let mw: Vec<Vec<Vec<f64>>> = w.iter().map(|l| l.iter().map(|r| vec![0.0; r.len()]).collect()).collect();
        let vw = mw.clone();
        let mb: Vec<Vec<f64>> = b.iter().map(|r| vec![0.0; r.len()]).collect();
        let vb = mb.clone();
        Mlp { sizes, w, b, mw, vw, mb, vb, t: 0 }
    }
    fn forward(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let nl = self.sizes.len() - 1;
        let mut acts = Vec::with_capacity(nl + 1);
        acts.push(x.to_vec());
        for l in 0..nl {
            let no = self.sizes[l + 1];
            let mut out = vec![0.0; no];
            for o in 0..no {
                let mut s = self.b[l][o];
                for i in 0..self.sizes[l] { s += self.w[l][o][i] * acts[l][i]; }
                out[o] = if l < nl - 1 { s.tanh() } else { s };
            }
            acts.push(out);
        }
        acts
    }
    fn predict(&self, x: &[f64]) -> f64 { *self.forward(x).last().unwrap().first().unwrap() }
    fn train(&mut self, xs: &[Vec<f64>], ys: &[f64], epochs: usize, lr: f64) {
        let nl = self.sizes.len() - 1;
        let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        for _ in 0..epochs {
            let mut gw: Vec<Vec<Vec<f64>>> =
                self.w.iter().map(|l| l.iter().map(|r| vec![0.0; r.len()]).collect()).collect();
            let mut gb: Vec<Vec<f64>> = self.b.iter().map(|r| vec![0.0; r.len()]).collect();
            for (x, &y) in xs.iter().zip(ys) {
                let acts = self.forward(x);
                let pred = acts[nl][0];
                let mut deltas: Vec<Vec<f64>> = vec![Vec::new(); nl];
                deltas[nl - 1] = vec![pred - y];
                for l in (0..nl).rev() {
                    for o in 0..self.sizes[l + 1] {
                        let d = deltas[l][o];
                        gb[l][o] += d;
                        for i in 0..self.sizes[l] { gw[l][o][i] += d * acts[l][i]; }
                    }
                    if l > 0 {
                        let ni = self.sizes[l];
                        let mut dp = vec![0.0; ni];
                        for i in 0..ni {
                            let mut s = 0.0;
                            for o in 0..self.sizes[l + 1] { s += self.w[l][o][i] * deltas[l][o]; }
                            let a = acts[l][i];
                            dp[i] = s * (1.0 - a * a);
                        }
                        deltas[l - 1] = dp;
                    }
                }
            }
            let n = xs.len() as f64;
            self.t += 1;
            let c1 = 1.0 - b1.powi(self.t);
            let c2 = 1.0 - b2.powi(self.t);
            for l in 0..nl {
                for o in 0..self.sizes[l + 1] {
                    let g = gb[l][o] / n;
                    self.mb[l][o] = b1 * self.mb[l][o] + (1.0 - b1) * g;
                    self.vb[l][o] = b2 * self.vb[l][o] + (1.0 - b2) * g * g;
                    self.b[l][o] -= lr * (self.mb[l][o] / c1) / ((self.vb[l][o] / c2).sqrt() + eps);
                    for i in 0..self.sizes[l] {
                        let g = gw[l][o][i] / n;
                        self.mw[l][o][i] = b1 * self.mw[l][o][i] + (1.0 - b1) * g;
                        self.vw[l][o][i] = b2 * self.vw[l][o][i] + (1.0 - b2) * g * g;
                        self.w[l][o][i] -=
                            lr * (self.mw[l][o][i] / c1) / ((self.vw[l][o][i] / c2).sqrt() + eps);
                    }
                }
            }
        }
    }
}

struct Ensemble { members: Vec<Mlp> }
impl Ensemble {
    fn train(d: usize, m: usize, xs: &[Vec<f64>], ys: &[f64], rng: &mut StdRng) -> Self {
        let n = xs.len();
        let mut members = Vec::with_capacity(m);
        for _ in 0..m {
            let mut bx = Vec::with_capacity(n);
            let mut by = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.gen_range(0..n);
                bx.push(xs[idx].clone());
                by.push(ys[idx]);
            }
            let mut net = Mlp::new(vec![d, 16, 16, 1], rng);
            net.train(&bx, &by, 120, 0.01);
            members.push(net);
        }
        Ensemble { members }
    }
    fn predict(&self, x: &[f64]) -> (f64, f64) {
        let p: Vec<f64> = self.members.iter().map(|m| m.predict(x)).collect();
        let mean = p.iter().sum::<f64>() / p.len() as f64;
        let var = p.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / p.len() as f64;
        (mean, var)
    }
}

// ── Trust Region (TuRBO 型) ──────────────────────────────────────────────────
struct Tr { length: f64, lmin: f64, lmax: f64, succ: usize, fail: usize }
impl Tr {
    fn new() -> Self { Tr { length: 0.8, lmin: 0.01, lmax: 1.6, succ: 0, fail: 0 } }
    fn update(&mut self, improved: bool) {
        if improved { self.succ += 1; self.fail = 0; } else { self.fail += 1; self.succ = 0; }
        if self.succ == 3 { self.length = (2.0 * self.length).min(self.lmax); self.succ = 0; }
        else if self.fail == 3 { self.length /= 2.0; self.fail = 0; }
    }
}

// ── Branin ([0,1]^2 → 実定義域) ──────────────────────────────────────────────
fn branin(u: &[f64]) -> f64 {
    let x1 = -5.0 + 15.0 * u[0];
    let x2 = 15.0 * u[1];
    let (a, b, c, r, s, t) = (1.0, 5.1 / (4.0 * PI * PI), 5.0 / PI, 6.0, 10.0, 1.0 / (8.0 * PI));
    a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
}

// ── CEM (acquisition 最大化、局所境界内) ─────────────────────────────────────
fn cem_maximize<F: Fn(&[f64]) -> f64>(
    f: F, center: &[f64], half: f64, lb: &[f64], ub: &[f64], rng: &mut StdRng,
) -> Vec<f64> {
    let d = center.len();
    let mut mean = center.to_vec();
    let mut std = vec![half; d];
    let mut best_x = center.to_vec();
    let mut best_v = f(center);
    for _ in 0..5 {
        let mut samples: Vec<(f64, Vec<f64>)> = Vec::with_capacity(64);
        for _ in 0..64 {
            let x: Vec<f64> = (0..d)
                .map(|j| (mean[j] + std[j].max(1e-6) * randn(rng)).clamp(lb[j], ub[j]))
                .collect();
            let v = f(&x);
            if v > best_v { best_v = v; best_x = x.clone(); }
            samples.push((v, x));
        }
        samples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let ne = 13; // 64 * 0.2
        for j in 0..d {
            let m = samples[..ne].iter().map(|s| s.1[j]).sum::<f64>() / ne as f64;
            let var = samples[..ne].iter().map(|s| (s.1[j] - m).powi(2)).sum::<f64>() / ne as f64;
            mean[j] = m;
            std[j] = var.sqrt() + 1e-6;
        }
    }
    best_x
}

fn main() {
    let d = 2;
    let n_local = 50;
    let max_evals = 160;
    let mut rng = StdRng::seed_from_u64(42);

    let mut xs: Vec<Vec<f64>> = vec![];
    let mut ys: Vec<f64> = vec![];
    let mut best_u = vec![0.0; d];
    let mut best_raw = f64::INFINITY;
    let mut eval = |u: Vec<f64>, xs: &mut Vec<Vec<f64>>, ys: &mut Vec<f64>,
                    best_u: &mut Vec<f64>, best_raw: &mut f64| {
        let v = branin(&u);
        if v < *best_raw { *best_raw = v; *best_u = u.clone(); }
        xs.push(u);
        ys.push(v);
    };

    for _ in 0..12 {
        let u: Vec<f64> = (0..d).map(|_| rng.gen::<f64>()).collect();
        eval(u, &mut xs, &mut ys, &mut best_u, &mut best_raw);
    }
    println!("init best = {:.6}", best_raw as f32);

    let mut tr = Tr::new();
    let mut local = false;
    let mut stagnation = 0;
    let mut round = 0;

    while xs.len() < max_evals {
        round += 1;
        // 標準化 + アンサンブル訓練
        let my = ys.iter().sum::<f64>() / ys.len() as f64;
        let sy = (ys.iter().map(|v| (v - my).powi(2)).sum::<f64>() / ys.len() as f64)
            .sqrt()
            .max(1e-6);
        let ys_std: Vec<f64> = ys.iter().map(|v| (v - my) / sy).collect();
        let ens = Ensemble::train(d, 5, &xs, &ys_std, &mut rng);
        let f_best_std = (best_raw - my) / sy;

        if !local {
            // GLOBAL: TR 内ランダム 400 点 → EI 上位 4 点を評価
            let l = tr.length;
            let lo: Vec<f64> = (0..d).map(|j| (best_u[j] - l / 2.0).max(0.0)).collect();
            let hi: Vec<f64> = (0..d).map(|j| (best_u[j] + l / 2.0).min(1.0)).collect();
            let mut cands: Vec<(f64, Vec<f64>)> = (0..400)
                .map(|_| {
                    let u: Vec<f64> = (0..d)
                        .map(|j| rng.gen_range(lo[j]..hi[j].max(lo[j] + 1e-9)))
                        .collect();
                    let (mu, var) = ens.predict(&u);
                    (ei_min(mu, (var + 1e-6).sqrt(), f_best_std, 0.01), u)
                })
                .collect();
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let top_ei = cands[0].0;
            let prev = best_raw;
            for k in 0..4.min(cands.len()) {
                let u = cands[k].1.clone();
                eval(u, &mut xs, &mut ys, &mut best_u, &mut best_raw);
            }
            tr.update(best_raw < prev - 1e-3 * prev.abs().max(1e-9));
            if top_ei < 1e-5 { stagnation += 1; } else { stagnation = 0; }
            if tr.length <= tr.lmin || stagnation >= 5 {
                local = true;
            }
        } else {
            // LOCAL: 残差 Micro-GP + CEM で 1 点精緻化
            let mut idx: Vec<(f64, usize)> = xs
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (x.iter().zip(&best_u).map(|(a, b)| (a - b).powi(2)).sum::<f64>(), i)
                })
                .collect();
            idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let active: Vec<usize> =
                idx.into_iter().take(n_local.min(xs.len())).map(|p| p.1).collect();
            let xl: Vec<Vec<f64>> = active.iter().map(|&i| xs[i].clone()).collect();
            let rl: Vec<f64> = active
                .iter()
                .map(|&i| ys_std[i] - ens.predict(&xs[i]).0)
                .collect();

            match gp::fit_micro_gp(&xl, &rl, &mut rng) {
                Ok(g) => {
                    let half = tr.length.max(0.02) / 2.0;
                    let lb: Vec<f64> = (0..d).map(|j| (best_u[j] - half).max(0.0)).collect();
                    let ub: Vec<f64> = (0..d).map(|j| (best_u[j] + half).min(1.0)).collect();
                    let acq = |u: &[f64]| -> f64 {
                        let mu_mlp = ens.predict(u).0;
                        let (mu_gp, var) = gp::gp_predict(&g, u);
                        ei_min(mu_mlp + mu_gp, var.sqrt(), f_best_std, 0.0)
                    };
                    let next = cem_maximize(acq, &best_u.clone(), half, &lb, &ub, &mut rng);
                    eval(next, &mut xs, &mut ys, &mut best_u, &mut best_raw);
                }
                Err(e) => {
                    // フォールバック: local 境界内ランダム 1 点
                    println!("  (GP fit failed: {e:?} — random fallback)");
                    let half = tr.length.max(0.02) / 2.0;
                    let u: Vec<f64> = (0..d)
                        .map(|j| {
                            let lo = (best_u[j] - half).max(0.0);
                            let hi = (best_u[j] + half).min(1.0);
                            rng.gen_range(lo..hi.max(lo + 1e-9))
                        })
                        .collect();
                    eval(u, &mut xs, &mut ys, &mut best_u, &mut best_raw);
                }
            }
        }

        println!(
            "round {:>3} | {} | best = {:>10.6} | L = {:.4} | evals = {}",
            round,
            if local { "LOCAL " } else { "GLOBAL" },
            best_raw as f32,
            tr.length as f32,
            xs.len()
        );
    }

    println!("\nfinal best = {:.6}  (Branin true min = 0.397887)", best_raw as f32);
    let x1 = -5.0 + 15.0 * best_u[0];
    let x2 = 15.0 * best_u[1];
    println!("final point = ({:.4}, {:.4})", x1 as f32, x2 as f32);
}
