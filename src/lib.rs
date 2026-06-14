mod acquisition;
mod batch;
mod candidate;
mod cem;
mod gp;
mod hypervolume;
mod normalize;
mod pareto;
mod surrogate;
mod tr;
mod types;

use std::cmp::Ordering;

use types::{ProposeConfig, ProposeMoConfig, ProposeMoOutput, ProposeOutput, TrustRegionState};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[pyclass]
pub struct Engine {}

#[pymethods]
impl Engine {
    #[new]
    pub fn new() -> Self {
        Engine {}
    }

    pub fn propose(
        &self,
        params: Vec<Vec<f32>>,
        values: Vec<f32>,
        feasibility: Vec<bool>,
        _constraint_values: Vec<Vec<f32>>,
        tr_states_json: String,
        config_json: String,
        seed: u64,
    ) -> PyResult<String> {
        let config: ProposeConfig = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid config_json: {e}")))?;
        let prev_tr_states: Vec<TrustRegionState> = serde_json::from_str(&tr_states_json)
            .map_err(|e| PyValueError::new_err(format!("invalid tr_states_json: {e}")))?;

        match config.acquisition.as_str() {
            "ucb" | "ei" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown acquisition '{other}': expected 'ucb' or 'ei'"
                )))
            }
        }

        // feasible/infeasible を分離 (infeasible は feasibility surrogate に使う)
        let (feasible_params, feasible_values): (Vec<Vec<f32>>, Vec<f32>) = params
            .iter()
            .zip(values.iter())
            .zip(feasibility.iter())
            .filter(|(_, &f)| f)
            .map(|((p, v), _)| (p.clone(), *v))
            .unzip();

        // warm path 発動条件: feasible 件数が n_init を超えていること
        if feasible_params.len() < config.n_init {
            return cold_start_output(config.batch_size, config.n_dims, seed);
        }

        // 全履歴中の最良点 (raw 最大化方向値)
        let global_best_idx = feasible_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let global_best_val = feasible_values[global_best_idx];
        let global_best_params = &feasible_params[global_best_idx];

        let (norm_values, _, _) = normalize::zscore(&feasible_values);
        let best_norm = norm_values[global_best_idx];

        // ── 共有サロゲート (全データ使用、TuRBO-M でも単一モデル) ─────────────
        let warm_states = if config.model_states.is_empty() {
            None
        } else {
            Some(config.model_states.as_slice())
        };
        let (ensemble, surrogate_loss, new_model_states) = surrogate::Ensemble::train(
            &feasible_params,
            &norm_values,
            config.n_dims,
            config.ensemble_size,
            seed,
            config.epochs,
            config.learning_rate,
            warm_states,
        );

        // feasibility surrogate: infeasible データがある場合に訓練 (ウォームスタート対応)
        let has_constraints = feasibility.iter().any(|&f| !f);
        let (feasibility_ensemble, new_feas_model_states): (Option<surrogate::Ensemble>, Vec<String>) =
            if has_constraints {
                let feas_labels: Vec<f32> =
                    feasibility.iter().map(|&f| if f { 1.0 } else { 0.0 }).collect();
                let feas_warm = if config.feas_model_states.is_empty() {
                    None
                } else {
                    Some(config.feas_model_states.as_slice())
                };
                let (ens, _, states) = surrogate::Ensemble::train(
                    &params,
                    &feas_labels,
                    config.n_dims,
                    config.ensemble_size,
                    seed.wrapping_add(0x4b7a_2fcd),
                    config.epochs,
                    config.learning_rate,
                    feas_warm,
                );
                (Some(ens), states)
            } else {
                (None, vec![])
            };

        // ── TR 状態管理 ─────────────────────────────────────────────────────────
        // n_trs: config の値を使うが、データ数を上限とする
        let effective_n_trs = config.n_trs.max(1).min(feasible_params.len());

        // Tandem Phase 2: 単一 TR のみ。local 中は TR を凍結（更新・再起動なし）。
        let phase2_armed = config.enable_phase2 && effective_n_trs == 1;
        let sticky_local = phase2_armed && config.phase == "local";
        let mut tr_exhausted = false;

        let mut tr_states: Vec<TrustRegionState> = if sticky_local
            && !prev_tr_states.is_empty()
        {
            prev_tr_states.clone()
        } else {
            // prev の TR 数が config と一致しない場合は全 TR を再初期化
            let need_init =
                prev_tr_states.is_empty() || prev_tr_states.len() != effective_n_trs;

            if need_init {
                if effective_n_trs == 1 {
                    // 後方互換: 単一 TR は global best を中心に初期化
                    vec![tr::init_tr(global_best_params, global_best_val, config.l_init, config.init_warmup)]
                } else {
                    // Multi-TR: Greedy Farthest-Point で多様な初期点を選択
                    tr::init_multi_tr(
                        &feasible_params,
                        &feasible_values,
                        effective_n_trs,
                        config.l_init,
                    )
                }
            } else {
                // 各 TR を独立に更新
                prev_tr_states
                    .iter()
                    .enumerate()
                    .map(|(k, state)| {
                        let updated = if effective_n_trs == 1 {
                            // 後方互換: 単一 TR はグローバルベスト帰属
                            tr::update_tr(state, global_best_val, global_best_params, &config)
                        } else {
                            // Multi-TR: 空間的帰属 (自身の TR 境界内データで成否判定)
                            tr::update_tr_spatial(
                                state,
                                &feasible_params,
                                &feasible_values,
                                &config,
                            )
                        };
                        if !updated.active {
                            if k == 0 {
                                if phase2_armed {
                                    // Phase 2 遷移シグナル: 再起動を保留し inactive のまま返す。
                                    // ガードで local に入れない場合は後段で restart する。
                                    tr_exhausted = true;
                                    updated
                                } else {
                                    // TR_0 は常に global best に戻る (搾取継続)
                                    tr::restart_tr(global_best_params, global_best_val, &config)
                                }
                            } else {
                                // TR_k (k>0) は密度考慮 Halton 多様点へ移動 (探索継続)
                                tr::restart_tr_explore(
                                    global_best_params,
                                    global_best_val,
                                    k,
                                    &config,
                                    seed.wrapping_add(k as u64 * 0xdeadbeef),
                                    &feasible_params,
                                )
                            }
                        } else {
                            updated
                        }
                    })
                    .collect()
            }
        };

        // ── Tandem Phase 2: phase 決定と local 分岐 ─────────────────────────────
        let min_evals = if config.phase2_min_evals > 0 {
            config.phase2_min_evals
        } else {
            3 * config.n_init
        };
        let enter_local = phase2_armed
            && feasible_params.len() >= min_evals
            && (sticky_local || tr_exhausted || config.stagnation_count >= 5);

        // ガードで local に入れなかった inactive TR は従来どおり再起動 (既存挙動維持)
        if tr_exhausted && !enter_local {
            tr_states[0] = tr::restart_tr(global_best_params, global_best_val, &config);
        }

        if enter_local {
            // 残差 Micro-GP を global best 近傍点でフィット。失敗時は global へフォールバック。
            let n_local = if config.phase2_local_points > 0 {
                config.phase2_local_points
            } else {
                50.max(config.n_dims + 2)
            };
            let mut order: Vec<usize> = (0..feasible_params.len()).collect();
            order.sort_by(|&a, &b| {
                let da: f32 = feasible_params[a]
                    .iter()
                    .zip(global_best_params)
                    .map(|(x, c)| (x - c).powi(2))
                    .sum();
                let db: f32 = feasible_params[b]
                    .iter()
                    .zip(global_best_params)
                    .map(|(x, c)| (x - c).powi(2))
                    .sum();
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            });
            let take = n_local.min(order.len());
            let sel: &[usize] = &order[..take];
            let sel_params: Vec<Vec<f32>> =
                sel.iter().map(|&i| feasible_params[i].clone()).collect();
            let (mlp_means, _) = ensemble.predict(&sel_params, config.n_dims);
            let xs_local: Vec<Vec<f64>> = sel_params
                .iter()
                .map(|p| p.iter().map(|&v| v as f64).collect())
                .collect();
            let residuals: Vec<f64> = sel
                .iter()
                .zip(&mlp_means)
                .map(|(&i, &m)| (norm_values[i] - m) as f64)
                .collect();

            let mut gp_rng = StdRng::seed_from_u64(seed.wrapping_add(0x9e37_79b9));
            if let Ok(micro) = gp::fit_micro_gp(&xs_local, &residuals, &mut gp_rng) {
                let l_frozen = tr_states[0].side_length.max(0.02);
                let half = l_frozen / 2.0;
                let lo: Vec<f32> =
                    global_best_params.iter().map(|c| (c - half).max(0.0)).collect();
                let hi: Vec<f32> =
                    global_best_params.iter().map(|c| (c + half).min(1.0)).collect();

                let pool = cem::cem_pool_gp(
                    &ensemble,
                    &micro,
                    config.n_dims,
                    global_best_params,
                    best_norm,
                    l_frozen / 6.0,
                    &lo,
                    &hi,
                    &config,
                    seed.wrapping_add(7),
                );

                let (pool_means, pool_stds) =
                    cem::combined_predict(&ensemble, &micro, &pool, config.n_dims);
                let mut acq_scores = acquisition::ei(&pool_means, &pool_stds, best_norm);
                if let Some(ref feas_surr) = feasibility_ensemble {
                    let (feas_means, _) = feas_surr.predict(&pool, config.n_dims);
                    for (s, f) in acq_scores.iter_mut().zip(feas_means.iter()) {
                        *s *= f.clamp(0.0, 1.0);
                    }
                }
                for s in acq_scores.iter_mut() {
                    if !s.is_finite() {
                        *s = f32::MIN; // NaN/inf 候補は実質除外
                    }
                }

                // local 領域に合わせた除外半径。不足分は一様乱数で backfill し
                // batch_size 件を必ず返す。
                let radius = (l_frozen * 0.1).min(0.1);
                let selected =
                    batch::greedy_select(&pool, &acq_scores, config.batch_size, radius);

                let mut candidates: Vec<Vec<f32>> =
                    selected.iter().map(|&i| pool[i].clone()).collect();
                let mut out_scores: Vec<f32> =
                    selected.iter().map(|&i| acq_scores[i]).collect();
                let mut out_means: Vec<f32> =
                    selected.iter().map(|&i| pool_means[i]).collect();
                let mut out_stds: Vec<f32> =
                    selected.iter().map(|&i| pool_stds[i]).collect();
                let mut bf_rng = StdRng::seed_from_u64(seed.wrapping_add(0xb0f1));
                while candidates.len() < config.batch_size {
                    candidates.push(
                        (0..config.n_dims)
                            .map(|j| bf_rng.gen_range(lo[j]..hi[j].max(lo[j] + 1e-6)))
                            .collect(),
                    );
                    out_scores.push(0.0);
                    out_means.push(0.0);
                    out_stds.push(0.0);
                }

                let output = ProposeOutput {
                    candidates,
                    tr_states,
                    acq_scores: out_scores,
                    pred_means: out_means,
                    pred_stds: out_stds,
                    surrogate_loss,
                    mode: "tandem_gp".to_string(),
                    model_states: new_model_states,
                    feas_model_states: new_feas_model_states,
                    phase: "local".to_string(),
                    stagnation_count: config.stagnation_count,
                };
                return serde_json::to_string(&output)
                    .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")));
            }
            // GP fit 失敗: global パスへフォールバック (phase="global" で次回再挑戦)。
            // 凍結保留していた inactive TR はここで再起動して通常更新に戻す。
            if !tr_states[0].active {
                tr_states[0] =
                    tr::restart_tr(global_best_params, global_best_val, &config);
            }
        }

        // ── CEM: 各 TR が独立して境界内でサンプリング ───────────────────────────
        let n_complete = feasible_params.len();
        let effective_n_cem_iters = if n_complete < 3 * config.n_init {
            config.n_cem_iters.min(16)
        } else {
            config.n_cem_iters
        };
        let cem_config = types::ProposeConfig {
            n_cem_iters: effective_n_cem_iters,
            ..config.clone()
        };

        let mut pool: Vec<Vec<f32>> = vec![];
        let mut pool_tr_idx: Vec<usize> = vec![]; // どの TR が生成した候補か

        for (tr_idx, tr_state) in tr_states.iter().enumerate() {
            let (lo, hi) = tr::tr_bounds(&tr_state.center, tr_state.side_length);
            let sigma_init = tr_state.side_length / 6.0;

            // TR 境界内の点を値の高い順にソート → CEM スタート候補に使う
            let mut in_tr_sorted: Vec<usize> = feasible_params
                .iter()
                .enumerate()
                .filter(|(_, p)| tr::in_bounds(p, &lo, &hi))
                .map(|(i, _)| i)
                .collect();
            in_tr_sorted.sort_unstable_by(|&a, &b| {
                norm_values[b]
                    .partial_cmp(&norm_values[a])
                    .unwrap_or(Ordering::Equal)
            });

            // スタート点: TR 内の top-3、なければ TR center のみ
            let start_points: Vec<Vec<f32>> = if in_tr_sorted.is_empty() {
                vec![tr_state.center.clone()]
            } else {
                in_tr_sorted[..3usize.min(in_tr_sorted.len())]
                    .iter()
                    .map(|&i| feasible_params[i].clone())
                    .collect()
            };

            let tr_pool_start = pool.len();
            for (k, start_mu) in start_points.iter().enumerate() {
                let elites = cem::cem_pool(
                    &ensemble,
                    config.n_dims,
                    start_mu,
                    best_norm,
                    sigma_init,
                    &lo,
                    &hi,
                    &cem_config,
                    // tr_idx=0, k=0,1,2 → seed+1,2,3  (n_trs=1 との後方互換を保つ)
                    seed.wrapping_add(1 + (tr_idx as u64 * 100 + k as u64)),
                );
                pool.extend(elites);
            }
            let tr_pool_end = pool.len();
            for _ in tr_pool_start..tr_pool_end {
                pool_tr_idx.push(tr_idx);
            }
        }

        // ── Acquisition スコア計算 ────────────────────────────────────────────────
        let (pool_means, pool_stds) = ensemble.predict(&pool, config.n_dims);
        let mut acq_scores = acquisition::score(
            &pool_means,
            &pool_stds,
            best_norm,
            config.beta,
            &config.acquisition,
        );

        // 制約付き: EI × P(feasible) で infeasible 領域を抑制
        if let Some(ref feas_surr) = feasibility_ensemble {
            let (feas_means, _) = feas_surr.predict(&pool, config.n_dims);
            for (s, f) in acq_scores.iter_mut().zip(feas_means.iter()) {
                *s *= f.clamp(0.0, 1.0);
            }
        }

        // ── バッチ選択 ────────────────────────────────────────────────────────────
        let selected = if effective_n_trs == 1 {
            // 後方互換: 既存の greedy_select をそのまま使用
            batch::greedy_select(&pool, &acq_scores, config.batch_size, 0.1)
        } else {
            // TuRBO-M: スロット飢餓防止
            // Phase 1: 各 TR から最低 1 候補を確定採用 (飢餓防止の最小保証)
            // Phase 2: 残り枠をマージ済み pool から Greedy で充填
            // per_tr_min=1 にして exploration TR への過剰配分を防ぐ
            let per_tr_min = 1_usize;
            let mut excluded = vec![false; pool.len()];
            let mut selected = Vec::with_capacity(config.batch_size);

            for tr_k in 0..effective_n_trs {
                let mut count = 0;
                while count < per_tr_min {
                    let best = (0..pool.len())
                        .filter(|&i| pool_tr_idx[i] == tr_k && !excluded[i])
                        .max_by(|&a, &b| {
                            acq_scores[a]
                                .partial_cmp(&acq_scores[b])
                                .unwrap_or(Ordering::Equal)
                        });
                    let Some(idx) = best else { break };

                    selected.push(idx);
                    excluded[idx] = true;

                    // 近傍点を除外 (多様性確保)
                    for j in 0..pool.len() {
                        if !excluded[j] {
                            let dist: f32 = pool[idx]
                                .iter()
                                .zip(&pool[j])
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>()
                                .sqrt();
                            if dist < 0.1 {
                                excluded[j] = true;
                            }
                        }
                    }
                    count += 1;
                }
            }

            // Phase 2: 残り枠を全体 Greedy (excluded 引き継ぎ) で充填
            let remaining = config.batch_size.saturating_sub(selected.len());
            if remaining > 0 {
                let extra = batch::greedy_select_partial(
                    &pool,
                    &acq_scores,
                    remaining,
                    0.1,
                    &excluded,
                );
                selected.extend(extra);
            }

            selected
        };

        let mode = if effective_n_trs > 1 {
            format!("tr_cem_m{effective_n_trs}")
        } else {
            "tr_cem".to_string()
        };

        let mut out_candidates: Vec<Vec<f32>> =
            selected.iter().map(|&i| pool[i].clone()).collect();
        let mut out_scores: Vec<f32> = selected.iter().map(|&i| acq_scores[i]).collect();
        let mut out_means: Vec<f32> = selected.iter().map(|&i| pool_means[i]).collect();
        let mut out_stds: Vec<f32> = selected.iter().map(|&i| pool_stds[i]).collect();
        // enable_phase2 時は batch_size 件保証 (重複データ等で greedy が不足した場合の backfill)。
        // デフォルト off では既存挙動 (不足を許容) を維持する。
        if config.enable_phase2 && !tr_states.is_empty() {
            let (lo, hi) = tr::tr_bounds(&tr_states[0].center, tr_states[0].side_length);
            let mut bf_rng = StdRng::seed_from_u64(seed.wrapping_add(0xb0f2));
            while out_candidates.len() < config.batch_size {
                out_candidates.push(
                    (0..config.n_dims)
                        .map(|j| bf_rng.gen_range(lo[j]..hi[j].max(lo[j] + 1e-6)))
                        .collect(),
                );
                out_scores.push(0.0);
                out_means.push(0.0);
                out_stds.push(0.0);
            }
        }

        // EI 停滞検出 (Phase 2 遷移シグナルの一つ)。ei 以外の獲得関数では無効。
        let max_acq = acq_scores
            .iter()
            .cloned()
            .filter(|s| s.is_finite())
            .fold(f32::MIN, f32::max);
        let stagnation_count = if config.acquisition == "ei" && max_acq < 1e-5 {
            config.stagnation_count + 1
        } else {
            0
        };

        let output = ProposeOutput {
            candidates: out_candidates,
            tr_states,
            acq_scores: out_scores,
            pred_means: out_means,
            pred_stds: out_stds,
            surrogate_loss,
            mode,
            model_states: new_model_states,
            feas_model_states: new_feas_model_states,
            phase: "global".to_string(),
            stagnation_count,
        };

        serde_json::to_string(&output)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))
    }

    /// 多目的提案 (Phase K-2: 2 目的 EHVI)。すべて最小化空間を前提とする。
    ///
    /// `objective_values[i]` は点 i の目的ベクトル (長さ n_obj)。infeasible 点の値は
    /// 無視される (フロント・学習から除外)。`ref_point` は最小化空間の参照点で、
    /// 空の場合は観測 nadir + マージンから動的に決める。
    pub fn propose_mo(
        &self,
        params: Vec<Vec<f32>>,
        objective_values: Vec<Vec<f32>>,
        feasibility: Vec<bool>,
        ref_point: Vec<f32>,
        config_json: String,
        seed: u64,
    ) -> PyResult<String> {
        let config: ProposeMoConfig = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid config_json: {e}")))?;

        if config.n_obj != 2 {
            return Err(PyValueError::new_err(format!(
                "propose_mo は現在 2 目的のみ対応 (n_obj={})",
                config.n_obj
            )));
        }

        // feasible 点のみ抽出 (params, 目的ベクトル)
        let feasible: Vec<(&Vec<f32>, &Vec<f32>)> = params
            .iter()
            .zip(objective_values.iter())
            .zip(feasibility.iter())
            .filter(|(_, &f)| f)
            .map(|((p, v), _)| (p, v))
            .collect();

        if feasible.len() < config.n_init {
            return cold_start_mo_output(config.batch_size, config.n_dims, seed);
        }

        let feasible_params: Vec<Vec<f32>> = feasible.iter().map(|(p, _)| (*p).clone()).collect();
        // 目的ごとの生値ベクトル
        let obj0: Vec<f32> = feasible.iter().map(|(_, v)| v[0]).collect();
        let obj1: Vec<f32> = feasible.iter().map(|(_, v)| v[1]).collect();

        // ── 目的ごとに z-score 正規化 + サロゲート学習 ─────────────────────────
        let (norm0, mean0, std0) = normalize::zscore(&obj0);
        let (norm1, mean1, std1) = normalize::zscore(&obj1);

        let warm = |k: usize| -> Option<&[String]> {
            config.model_states.get(k).map(|v| v.as_slice())
        };

        let (ens0, _, states0) = surrogate::Ensemble::train(
            &feasible_params, &norm0, config.n_dims, config.ensemble_size,
            seed, config.epochs, config.learning_rate, warm(0),
        );
        let (ens1, _, states1) = surrogate::Ensemble::train(
            &feasible_params, &norm1, config.n_dims, config.ensemble_size,
            seed.wrapping_add(0x51_2a_7f), config.epochs, config.learning_rate, warm(1),
        );
        let ensembles = [ens0, ens1];

        // ── Pareto フロント (生・最小化空間) ───────────────────────────────────
        let costs: Vec<Vec<f32>> = (0..feasible_params.len())
            .map(|i| vec![obj0[i], obj1[i]])
            .collect();
        let nd_idx = pareto::nondominated_indices(&costs);
        let front_raw: Vec<[f64; 2]> = pareto::sorted_front_2d(&costs);

        // ideal / nadir → 動的参照点 (ref_point が空なら自動)
        let ideal = [
            front_raw.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            front_raw.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        ];
        let nadir = [
            front_raw.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
            front_raw.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
        ];
        let margin = config.ref_margin as f64;
        let ref_raw: [f64; 2] = if ref_point.len() == 2 {
            [ref_point[0] as f64, ref_point[1] as f64]
        } else {
            [
                nadir[0] + (margin * (nadir[0] - ideal[0])).max(1e-3),
                nadir[1] + (margin * (nadir[1] - ideal[1])).max(1e-3),
            ]
        };

        // 正規化空間へ変換 (単調変換ゆえ昇順 x は保たれるが、安全のため再ソート)
        let to_norm = |p: [f64; 2]| -> [f64; 2] {
            [
                (p[0] - mean0 as f64) / std0 as f64,
                (p[1] - mean1 as f64) / std1 as f64,
            ]
        };
        let mut front_norm: Vec<[f64; 2]> = front_raw.iter().map(|&p| to_norm(p)).collect();
        front_norm.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));
        let ref_norm = to_norm(ref_raw);

        // ── CEM スタート点: Pareto 点 (フロント沿い分散) + 中心 (探索) ──────────
        let mut starts: Vec<Vec<f32>> = Vec::new();
        let n_front_starts = config.n_cem_starts.saturating_sub(1).max(1);
        for &i in nd_idx.iter().take(n_front_starts) {
            starts.push(feasible_params[i].clone());
        }
        starts.push(vec![0.5f32; config.n_dims]);

        // ── 各スタートで EHVI-CEM → プール集約 ─────────────────────────────────
        let mut pool: Vec<Vec<f32>> = Vec::new();
        for (s, start_mu) in starts.iter().enumerate() {
            let elites = cem::cem_pool_ehvi(
                &ensembles,
                config.n_dims,
                start_mu,
                config.sigma_init,
                &front_norm,
                ref_norm,
                &config,
                seed.wrapping_add(1 + s as u64 * 0x9e3779b9),
            );
            pool.extend(elites);
        }

        // ── EHVI スコア + 多様性付き Greedy 選択 ───────────────────────────────
        let mut acq = cem::ehvi_scores(&ensembles, &pool, config.n_dims, &front_norm, ref_norm);
        for s in acq.iter_mut() {
            if !s.is_finite() {
                *s = f32::MIN;
            }
        }
        let selected = batch::greedy_select(&pool, &acq, config.batch_size, 0.1);

        let mut candidates: Vec<Vec<f32>> = selected.iter().map(|&i| pool[i].clone()).collect();
        let mut ehvi_scores: Vec<f32> = selected.iter().map(|&i| acq[i]).collect();

        // 不足分を一様乱数で backfill (batch_size 件保証)
        let mut bf_rng = StdRng::seed_from_u64(seed.wrapping_add(0xb0f3));
        while candidates.len() < config.batch_size {
            candidates.push((0..config.n_dims).map(|_| bf_rng.gen::<f32>()).collect());
            ehvi_scores.push(0.0);
        }

        let hv = hypervolume::hypervolume_2d(&front_raw, ref_raw) as f32;
        let output = ProposeMoOutput {
            candidates,
            ehvi_scores,
            model_states: vec![states0, states1],
            pareto_size: nd_idx.len(),
            hypervolume: hv,
            mode: "ehvi_cem".to_string(),
        };
        serde_json::to_string(&output)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))
    }
}

fn cold_start_mo_output(batch_size: usize, n_dims: usize, seed: u64) -> PyResult<String> {
    let candidates = candidate::halton(batch_size, n_dims, seed);
    let n = candidates.len();
    let output = ProposeMoOutput {
        candidates,
        ehvi_scores: vec![0.0; n],
        model_states: vec![],
        pareto_size: 0,
        hypervolume: 0.0,
        mode: "cold_start".to_string(),
    };
    serde_json::to_string(&output)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))
}

fn cold_start_output(batch_size: usize, n_dims: usize, seed: u64) -> PyResult<String> {
    let candidates = candidate::halton(batch_size, n_dims, seed);
    let n = candidates.len();
    let output = ProposeOutput {
        candidates,
        tr_states: vec![],
        acq_scores: vec![0.0; n],
        pred_means: vec![0.0; n],
        pred_stds: vec![0.0; n],
        surrogate_loss: 0.0,
        mode: "cold_start".to_string(),
        model_states: vec![],
        feas_model_states: vec![],
        phase: "global".to_string(),
        stagnation_count: 0,
    };
    serde_json::to_string(&output)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))
}

#[pymodule]
fn _lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
