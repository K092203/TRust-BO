use std::cmp::Ordering;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::acquisition;
use crate::gp;
use crate::surrogate::Ensemble;
use crate::types::{ProposeConfig, ProposeMoConfig};

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

    // RAASP 型次元マスク (Papenmeier et al. ICML 2025 の知見の CEM 適応):
    // 有効時は各次元を確率 p でのみ摂動 (残りは mu に固定、最低 1 次元は摂動保証)。
    // モーメント更新は「その次元を実際に摂動した elite」のみで行い (不足時は旧値保持)、
    // 非摂動値混入による σ の人工収縮 (≈√p 倍/反復) を防ぐ。
    // マスクされた次元はガウス乱数を消費しない。フラグ off は従来と完全同一パス。
    let mask_p = if config.cem_dim_mask {
        (20.0 / n_dims as f32).min(1.0)
    } else {
        1.0
    };

    let mut best_pool: Vec<Vec<f32>> = vec![init_mu.to_vec()];
    for _ in 0..config.n_cem_iters {
        // perturbed[i][j]: 候補 i が次元 j を摂動したか (mask off 時は使わない)
        let mut perturbed: Vec<Vec<bool>> = Vec::new();
        let candidates: Vec<Vec<f32>> = (0..config.n_cem_samples)
            .map(|_| {
                if !config.cem_dim_mask {
                    return (0..n_dims)
                        .map(|j| {
                            let u1 = rng.gen::<f32>().max(1e-7);
                            let u2 = rng.gen::<f32>();
                            let z = (-2.0 * u1.ln()).sqrt()
                                * (2.0 * std::f32::consts::PI * u2).cos();
                            (mu[j] + sigma[j] * z).clamp(bounds_lo[j], bounds_hi[j])
                        })
                        .collect();
                }
                let mut mask: Vec<bool> = (0..n_dims).map(|_| rng.gen::<f32>() < mask_p).collect();
                if !mask.iter().any(|&m| m) {
                    // 最低 1 次元は摂動する (全固定候補は情報ゼロ)
                    let j = rng.gen_range(0..n_dims);
                    mask[j] = true;
                }
                let cand = (0..n_dims)
                    .map(|j| {
                        if !mask[j] {
                            return mu[j].clamp(bounds_lo[j], bounds_hi[j]);
                        }
                        let u1 = rng.gen::<f32>().max(1e-7);
                        let u2 = rng.gen::<f32>();
                        let z =
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        (mu[j] + sigma[j] * z).clamp(bounds_lo[j], bounds_hi[j])
                    })
                    .collect();
                perturbed.push(mask);
                cand
            })
            .collect();

        let (means, stds) = ensemble.predict(&candidates, n_dims);
        let scores = if config.acquisition == "ts" {
            acquisition::ts(&means, &stds, &mut rng)
        } else {
            acquisition::score(&means, &stds, best_norm, config.beta, &config.acquisition)
        };

        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let elite_idx: Vec<usize> = indexed[..elite_k.min(indexed.len())]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        best_pool = elite_idx.iter().map(|&i| candidates[i].clone()).collect();

        if !config.cem_dim_mask {
            for j in 0..n_dims {
                let mu_new = best_pool.iter().map(|c| c[j]).sum::<f32>() / elite_k as f32;
                let var = best_pool.iter().map(|c| (c[j] - mu_new).powi(2)).sum::<f32>()
                    / elite_k as f32;
                mu[j] = mu_new;
                sigma[j] = var.sqrt().max(1e-8);
            }
        } else {
            for j in 0..n_dims {
                // 次元 j を実際に摂動した elite のみでモーメント更新 (2 点未満は旧値保持)
                let vals: Vec<f32> = elite_idx
                    .iter()
                    .filter(|&&i| perturbed[i][j])
                    .map(|&i| candidates[i][j])
                    .collect();
                if vals.len() >= 2 {
                    let mu_new = vals.iter().sum::<f32>() / vals.len() as f32;
                    let var = vals.iter().map(|v| (v - mu_new).powi(2)).sum::<f32>()
                        / vals.len() as f32;
                    mu[j] = mu_new;
                    sigma[j] = var.sqrt().max(1e-8);
                }
            }
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

/// 多目的 EHVI を最大化する CEM (2 目的、正規化空間)。
///
/// `ensembles[k]` は目的 k の正規化値で学習済みサロゲート。
/// `front_norm` / `ref_norm` も正規化空間の値であること。
/// 1 スタートぶんの elite 候補プールを返す。
#[allow(clippy::too_many_arguments)]
pub fn cem_pool_ehvi(
    ensembles: &[Ensemble],
    n_dims: usize,
    init_mu: &[f32],
    sigma_init: f32,
    front_norm: &[[f64; 2]],
    ref_norm: [f64; 2],
    config: &ProposeMoConfig,
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
                        (mu[j] + sigma[j] * z).clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        let scores = ehvi_scores(ensembles, &candidates, n_dims, front_norm, ref_norm);

        let mut indexed: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| s.is_finite())
            .collect();
        if indexed.is_empty() {
            break;
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

/// 候補ごとの EHVI を計算する (各目的のサロゲートで予測 → 2D EHVI)。
pub fn ehvi_scores(
    ensembles: &[Ensemble],
    candidates: &[Vec<f32>],
    n_dims: usize,
    front_norm: &[[f64; 2]],
    ref_norm: [f64; 2],
) -> Vec<f32> {
    let (m0, s0) = ensembles[0].predict(candidates, n_dims);
    let (m1, s1) = ensembles[1].predict(candidates, n_dims);
    let means: Vec<[f64; 2]> = (0..candidates.len())
        .map(|i| [m0[i] as f64, m1[i] as f64])
        .collect();
    let stds: Vec<[f64; 2]> = (0..candidates.len())
        .map(|i| [s0[i] as f64, s1[i] as f64])
        .collect();
    acquisition::ehvi_2d_batch(&means, &stds, front_norm, ref_norm)
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
