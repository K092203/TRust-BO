use std::cmp::Ordering;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::acquisition;
use crate::gp;
use crate::surrogate::Ensemble;
use crate::types::ProposeConfig;

/// CEM で候補プールを生成する。
/// bounds_lo / bounds_hi でサンプルをクランプ (TR 境界)。
/// sigma_init: 初期標準偏差 (TR の場合 L/6、非 TR の場合 0.2)。
/// Phase 3 から: bounds に TR 境界を渡す。Phase 2 では [0,1]^n で呼び出す。
pub fn cem_pool(
    ensemble: &Ensemble,
    n_dims: usize,
    init_mu: &[f32],
    best_norm: f32,
    sigma_init: f32,
    bounds_lo: &[f32],
    bounds_hi: &[f32],
    config: &ProposeConfig,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut mu = init_mu.to_vec();
    let mut sigma = vec![sigma_init; n_dims];

    let elite_k = ((config.n_cem_samples as f32 * config.elite_fraction) as usize).max(1);
    // halt_eps = L × 1e-3; sigma_init = L/6  →  halt_eps = sigma_init × 6e-3
    let halt_eps = (sigma_init * 6e-3).max(1e-8);

    let mut best_pool: Vec<Vec<f32>> = vec![init_mu.to_vec()];

    for _ in 0..config.n_cem_iters {
        let candidates: Vec<Vec<f32>> = (0..config.n_cem_samples)
            .map(|_| {
                (0..n_dims)
                    .map(|j| {
                        let u1 = rng.gen::<f32>().max(1e-7);
                        let u2 = rng.gen::<f32>();
                        let z =
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        (mu[j] + sigma[j] * z).clamp(bounds_lo[j], bounds_hi[j])
                    })
                    .collect()
            })
            .collect();

        let (means, stds) = ensemble.predict(&candidates, n_dims);
        let scores =
            acquisition::score(&means, &stds, best_norm, config.beta, &config.acquisition);

        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        best_pool = indexed[..elite_k.min(indexed.len())]
            .iter()
            .map(|(i, _)| candidates[*i].clone())
            .collect();

        for j in 0..n_dims {
            let mu_new = best_pool.iter().map(|c| c[j]).sum::<f32>() / elite_k as f32;
            let var = best_pool.iter().map(|c| (c[j] - mu_new).powi(2)).sum::<f32>()
                / elite_k as f32;
            mu[j] = mu_new;
            sigma[j] = var.sqrt().max(1e-8);
        }

        if sigma.iter().cloned().fold(0f32, f32::max) < halt_eps {
            break;
        }
    }

    best_pool
}

/// Phase 2 (local) 用 CEM。combined 予測 mu = mu_MLP + mu_GP残差、sigma = GP 分散。
/// f64 (GP) ↔ f32 (MLP/CEM) の変換はこの関数内に集約する。
pub fn cem_pool_gp(
    ensemble: &Ensemble,
    micro_gp: &gp::MicroGp,
    n_dims: usize,
    init_mu: &[f32],
    best_norm: f32,
    sigma_init: f32,
    bounds_lo: &[f32],
    bounds_hi: &[f32],
    config: &ProposeConfig,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut mu = init_mu.to_vec();
    let mut sigma = vec![sigma_init; n_dims];

    let elite_k = ((config.n_cem_samples as f32 * config.elite_fraction) as usize).max(1);
    let halt_eps = (sigma_init * 6e-3).max(1e-8);

    let mut best_pool: Vec<Vec<f32>> = vec![init_mu.to_vec()];

    for _ in 0..config.n_cem_iters {
        let candidates: Vec<Vec<f32>> = (0..config.n_cem_samples)
            .map(|_| {
                (0..n_dims)
                    .map(|j| {
                        let u1 = rng.gen::<f32>().max(1e-7);
                        let u2 = rng.gen::<f32>();
                        let z =
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        (mu[j] + sigma[j] * z).clamp(bounds_lo[j], bounds_hi[j])
                    })
                    .collect()
            })
            .collect();

        let (means, stds) = combined_predict(ensemble, micro_gp, &candidates, n_dims);
        let scores = acquisition::ei(&means, &stds, best_norm);

        let mut indexed: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| s.is_finite())
            .collect();
        if indexed.is_empty() {
            break; // 全スコア非有限 → 現在の elite を返す (呼び出し側で backfill)
        }
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        best_pool = indexed[..elite_k.min(indexed.len())]
            .iter()
            .map(|(i, _)| candidates[*i].clone())
            .collect();

        let k = best_pool.len() as f32;
        for j in 0..n_dims {
            let mu_new = best_pool.iter().map(|c| c[j]).sum::<f32>() / k;
            let var = best_pool.iter().map(|c| (c[j] - mu_new).powi(2)).sum::<f32>() / k;
            mu[j] = mu_new;
            sigma[j] = var.sqrt().max(1e-8);
        }

        if sigma.iter().cloned().fold(0f32, f32::max) < halt_eps {
            break;
        }
    }

    best_pool
}

/// combined 予測: mu = mu_MLP + mu_GP残差, sigma = sqrt(var_GP)。
pub fn combined_predict(
    ensemble: &Ensemble,
    micro_gp: &gp::MicroGp,
    candidates: &[Vec<f32>],
    n_dims: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (mlp_means, _) = ensemble.predict(candidates, n_dims);
    let mut means = Vec::with_capacity(candidates.len());
    let mut stds = Vec::with_capacity(candidates.len());
    for (c, m) in candidates.iter().zip(mlp_means) {
        let x64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
        let (mu_gp, var_gp) = gp::gp_predict(micro_gp, &x64);
        means.push(m + mu_gp as f32);
        stds.push((var_gp.max(1e-12).sqrt()) as f32);
    }
    (means, stds)
}
