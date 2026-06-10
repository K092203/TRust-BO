use std::cmp::Ordering;

use crate::candidate;
use crate::types::{ProposeConfig, TrustRegionState};

/// TR 超立方体の境界 [lo, hi] を計算する。[0,1] にクランプ。
pub fn tr_bounds(center: &[f32], side_length: f32) -> (Vec<f32>, Vec<f32>) {
    let half = side_length / 2.0;
    let lo: Vec<f32> = center.iter().map(|&c| (c - half).max(0.0)).collect();
    let hi: Vec<f32> = center.iter().map(|&c| (c + half).min(1.0)).collect();
    (lo, hi)
}

/// TR 境界内に点があるか判定する。
pub fn in_bounds(p: &[f32], lo: &[f32], hi: &[f32]) -> bool {
    p.iter().zip(lo).zip(hi).all(|((x, l), h)| *x >= *l && *x <= *h)
}

/// 最初の warm call 時に TR を初期化する。
/// `warmup` は config.init_warmup から渡す (デフォルト 0 = 無効)。
// [CHANGED]: 修正① — 早期 TR 収縮防止のため warmup_remaining を init_warmup から設定する。
// デフォルト 0 で既存テストへの影響なし。本番での推奨値は 2–3。
pub fn init_tr(center: &[f32], best_value: f32, l_init: f32, warmup: usize) -> TrustRegionState {
    TrustRegionState {
        center: center.to_vec(),
        side_length: l_init,
        success_count: 0,
        failure_count: 0,
        best_value,
        active: true,
        warmup_remaining: warmup,
    }
}

/// propose() 呼び出しごとに TR を更新する。
/// global_best_val / global_best_params: 全履歴中の最良点 (raw 最大化方向値)
pub fn update_tr(
    state: &TrustRegionState,
    global_best_val: f32,
    global_best_params: &[f32],
    config: &ProposeConfig,
) -> TrustRegionState {
    let tau_succ = config.tau_succ;
    let tau_fail = if config.tau_fail == 0 {
        config.n_dims.max(4)
    } else {
        config.tau_fail
    };

    let mut s = state.clone();

    let in_warmup = s.warmup_remaining > 0;
    if in_warmup {
        s.warmup_remaining -= 1;
    }

    if global_best_val > s.best_value {
        let rel_improvement = if s.best_value.abs() > 1e-8 {
            (global_best_val - s.best_value) / s.best_value.abs()
        } else {
            1.0
        };
        s.best_value = global_best_val;
        if rel_improvement >= 0.01 {
            s.center = global_best_params.to_vec();
            s.success_count += 1;
            s.failure_count = 0;
        }
        // minor improvement: neutral
    } else if !in_warmup {
        s.failure_count += 1;
        s.success_count = 0;
    }

    if s.success_count >= tau_succ {
        s.side_length = (s.side_length * 2.0).min(config.l_max);
        s.success_count = 0;
        s.failure_count = 0;
    }
    if s.failure_count >= tau_fail {
        s.side_length /= 2.0;
        s.success_count = 0;
        s.failure_count = 0;
    }

    if s.side_length < config.l_min {
        s.active = false;
    }

    s
}

/// L < L_min になった TR を global best center から再起動する。
/// TuRBO 論文に準拠: 単一 TR は exhausted 後に既知最良点から l_init で再探索。
pub fn restart_tr(
    global_best_params: &[f32],
    global_best_val: f32,
    config: &ProposeConfig,
) -> TrustRegionState {
    TrustRegionState {
        center: global_best_params.to_vec(),
        side_length: config.l_init,
        success_count: 0,
        failure_count: 0,
        best_value: global_best_val,
        active: true,
        warmup_remaining: 0,
    }
}

// ── TuRBO-M 追加関数 ──────────────────────────────────────────────────────────

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// M 個の TR を初期化する (TuRBO-M 用)。
///
/// - TR_0: グローバル最良点 (搾取 TR)
/// - TR_1+: Greedy Farthest-Point で選ばれた多様な点 (探索 TR)
///
/// `params`/`raw_values` は cold start 終了時点の全 feasible データ。
pub fn init_multi_tr(
    params: &[Vec<f32>],
    raw_values: &[f32],
    n_trs: usize,
    l_init: f32,
) -> Vec<TrustRegionState> {
    let n = params.len();
    let n_trs = n_trs.min(n).max(1);

    // TR_0: グローバル最良点のインデックス
    let best_idx = raw_values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut chosen = vec![best_idx];

    // TR_1+: 既選択点群から最も遠い点を貪欲に追加 (Greedy Farthest-Point)
    while chosen.len() < n_trs {
        let next = (0..n)
            .filter(|&i| !chosen.contains(&i))
            .max_by(|&a, &b| {
                let min_dist_a = chosen
                    .iter()
                    .map(|&c| l2_dist(&params[a], &params[c]))
                    .fold(f32::INFINITY, f32::min);
                let min_dist_b = chosen
                    .iter()
                    .map(|&c| l2_dist(&params[b], &params[c]))
                    .fold(f32::INFINITY, f32::min);
                min_dist_a.partial_cmp(&min_dist_b).unwrap_or(Ordering::Equal)
            });
        match next {
            Some(idx) => chosen.push(idx),
            None => break,
        }
    }

    chosen
        .iter()
        .enumerate()
        .map(|(k, &idx)| TrustRegionState {
            center: params[idx].clone(),
            side_length: l_init,
            success_count: 0,
            failure_count: 0,
            best_value: raw_values[idx],
            active: true,
            // 探索 TR (k>0) には初期 warmup を与えて即時収縮を防ぐ
            warmup_remaining: if k == 0 { 0 } else { 2 },
        })
        .collect()
}

/// 空間的帰属による TR 更新 (TuRBO-M 用)。
///
/// 自身の境界内にある全 feasible データから local_best を求め、
/// `state.best_value` と比較することで各 TR が独立して成否を判定する。
/// 境界内にデータがない場合は failure として扱う。
pub fn update_tr_spatial(
    state: &TrustRegionState,
    params: &[Vec<f32>],
    raw_values: &[f32],
    config: &ProposeConfig,
) -> TrustRegionState {
    let (lo, hi) = tr_bounds(&state.center, state.side_length);

    // 境界内で最良の (params, value) ペアを探す
    let best_in_bounds = params
        .iter()
        .zip(raw_values.iter())
        .filter(|(p, _)| in_bounds(p, &lo, &hi))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let (local_best_val, local_best_params) = match best_in_bounds {
        Some((p, &v)) => (v, p.clone()),
        // 境界内にデータなし → failure として扱う (center は変更しない)
        None => (state.best_value, state.center.clone()),
    };

    // 既存の update_tr ロジックに委譲
    update_tr(state, local_best_val, &local_best_params, config)
}

/// 探索 TR 用のリスタート (TuRBO-M 用)。
///
/// - TR_0 (tr_idx == 0): グローバル最良点に戻る (搾取継続)
/// - TR_k (tr_idx  > 0): 既観測点から最も遠い Halton 候補に移動 (密度考慮探索)
///
/// `observed_params` に全 feasible データを渡すことで、
/// 既に密に観測された領域を避けた多様な再配置を実現する。
///
/// `best_value = global_best_val` にセットすることで
/// グローバルベストを超えた場合のみ success と見なす (積極的探索)。
pub fn restart_tr_explore(
    global_best_params: &[f32],
    global_best_val: f32,
    tr_idx: usize,
    config: &ProposeConfig,
    seed: u64,
    observed_params: &[Vec<f32>],
) -> TrustRegionState {
    let center = if tr_idx == 0 {
        global_best_params.to_vec()
    } else {
        // 100 点の Halton 候補を生成し、全観測点から最も遠い点を選ぶ
        let halton_seed = seed
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(tr_idx as u64 * 0xdeadbeef);
        let candidates = candidate::halton(100, config.n_dims, halton_seed);

        if observed_params.is_empty() {
            candidates.into_iter().next().unwrap_or_else(|| global_best_params.to_vec())
        } else {
            // 各候補について「最も近い観測点との距離」を計算し、最大の候補を選ぶ
            candidates
                .into_iter()
                .max_by(|a, b| {
                    let min_a = observed_params
                        .iter()
                        .map(|p| l2_dist(a, p))
                        .fold(f32::INFINITY, f32::min);
                    let min_b = observed_params
                        .iter()
                        .map(|p| l2_dist(b, p))
                        .fold(f32::INFINITY, f32::min);
                    min_a.partial_cmp(&min_b).unwrap_or(Ordering::Equal)
                })
                .unwrap_or_else(|| global_best_params.to_vec())
        }
    };
    TrustRegionState {
        center,
        side_length: config.l_init,
        success_count: 0,
        failure_count: 0,
        best_value: global_best_val,
        active: true,
        warmup_remaining: if tr_idx == 0 { 0 } else { 2 },
    }
}
