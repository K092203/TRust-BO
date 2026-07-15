/// 貪欲法でバッチを選択する (initial_excluded から開始)。
///
/// 最高スコア点を選んだ後、距離 eps 未満の点を候補から除外して繰り返す。
/// `initial_excluded` で事前に除外しておく点を指定できる (Multi-TR スロット保証に使用)。
/// 利用可能な候補が count に満たない場合は取得できた分だけ返す。
pub fn greedy_select_partial(
    pool: &[Vec<f32>],
    scores: &[f32],
    count: usize,
    eps: f32,
    initial_excluded: &[bool],
) -> Vec<usize> {
    let n = pool.len();
    let mut excluded = initial_excluded.to_vec();
    let mut selected = Vec::with_capacity(count);

    while selected.len() < count {
        let best = (0..n)
            .filter(|&i| !excluded[i])
            .max_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let Some(idx) = best else { break };

        selected.push(idx);
        excluded[idx] = true;

        for j in 0..n {
            if !excluded[j] {
                let dist: f32 = pool[idx]
                    .iter()
                    .zip(&pool[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist < eps {
                    excluded[j] = true;
                }
            }
        }
    }

    selected
}

/// 貪欲法でバッチを選択する (全候補から開始)。
/// `greedy_select_partial` の convenience wrapper。
pub fn greedy_select(
    pool: &[Vec<f32>],
    scores: &[f32],
    batch_size: usize,
    eps: f32,
) -> Vec<usize> {
    greedy_select_partial(pool, scores, batch_size, eps, &vec![false; pool.len()])
}

/// アンサンブルメンバーを決定的シナリオ集合とした joint marginal-improvement greedy。
/// 選択済み候補の予測値でメンバーごとの running best を更新し、候補は再選択しない。
pub fn joint_greedy_select(
    pool: &[Vec<f32>],
    member_preds: &[Vec<f32>],
    running_best_init: &[f32],
    feas_weights: Option<&[f32]>,
    acq_scores: &[f32],
    batch_size: usize,
) -> Vec<usize> {
    const EXCLUSION_RADIUS: f32 = 0.1;
    assert_eq!(member_preds.len(), running_best_init.len());
    assert!(member_preds.iter().all(|preds| preds.len() == pool.len()));
    if let Some(weights) = feas_weights {
        assert_eq!(weights.len(), pool.len());
    }
    assert_eq!(acq_scores.len(), pool.len());
    // Feasibility probabilityを決定的シナリオのrunning bestへ正しく取り込む
    // joint近似ではないため、制約ありでは既存の安全なgreedyをそのまま使う。
    if feas_weights.is_some() {
        return greedy_select(pool, acq_scores, batch_size, EXCLUSION_RADIUS);
    }
    if member_preds.is_empty() {
        return Vec::new();
    }

    let mut running_best = running_best_init.to_vec();
    let mut remaining = vec![true; pool.len()];
    let mut selected = Vec::with_capacity(batch_size.min(pool.len()));

    while selected.len() < batch_size {
        let mut best: Option<(usize, f32)> = None;
        for i in 0..pool.len() {
            if !remaining[i] {
                continue;
            }
            let score = member_preds
                .iter()
                .zip(running_best.iter())
                .map(|(preds, &incumbent)| (preds[i] - incumbent).max(0.0))
                .sum::<f32>()
                / member_preds.len() as f32;
            // 同点では先に走査した小さい pool index を維持する。
            if best.map_or(true, |(_, best_score)| score > best_score) {
                best = Some((i, score));
            }
        }

        let Some((idx, _)) = best else { break };
        selected.push(idx);
        remaining[idx] = false;
        for j in 0..pool.len() {
            if remaining[j] {
                let dist = pool[idx]
                    .iter()
                    .zip(&pool[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist < EXCLUSION_RADIUS {
                    remaining[j] = false;
                }
            }
        }
        for (m, incumbent) in running_best.iter_mut().enumerate() {
            *incumbent = incumbent.max(member_preds[m][idx]);
        }
    }

    if selected.len() < batch_size {
        let mut excluded = vec![false; pool.len()];
        for &idx in &selected {
            for j in 0..pool.len() {
                let dist = pool[idx]
                    .iter()
                    .zip(&pool[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist < EXCLUSION_RADIUS {
                    excluded[j] = true;
                }
            }
        }
        selected.extend(greedy_select_partial(
            pool,
            acq_scores,
            batch_size - selected.len(),
            EXCLUSION_RADIUS,
            &excluded,
        ));
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::{greedy_select, joint_greedy_select};

    #[test]
    fn joint_greedy_updates_member_running_best() {
        let pool = vec![vec![0.0], vec![0.5], vec![1.0]];
        let member_preds = vec![vec![3.0, 2.4, 0.0], vec![0.0, 2.4, 3.0]];
        let selected = joint_greedy_select(
            &pool, &member_preds, &[0.0, 0.0], None, &[3.0, 2.4, 3.0], 2,
        );
        // 点1が平均改善最大。更新後は点0/2が同点なので小さいindexの点0。
        assert_eq!(selected, vec![1, 0]);
    }

    #[test]
    fn constrained_joint_uses_existing_greedy_for_threshold_counterexample() {
        let pool = vec![vec![0.0], vec![0.5], vec![1.0]]; // A, C, B
        let member_preds = vec![vec![1000.0, 80.0, 90.0]];
        let acq_scores = vec![510.0, 80.0, 90.0];
        let selected = joint_greedy_select(
            &pool,
            &member_preds,
            &[0.0],
            Some(&[0.51, 1.0, 1.0]),
            &acq_scores,
            2,
        );
        let greedy = greedy_select(&pool, &acq_scores, 2, 0.1);
        assert_eq!(greedy, vec![0, 2]);
        assert_eq!(selected, greedy);
    }

    #[test]
    fn joint_greedy_excludes_duplicate_coordinates() {
        let pool = vec![vec![0.0], vec![0.0], vec![1.0]];
        let member_preds = vec![vec![1.0, 1.0, 0.0]];
        let selected = joint_greedy_select(
            &pool, &member_preds, &[0.0], None, &[1.0, 1.0, 0.0], 2,
        );
        assert_eq!(selected, vec![0, 2]);
        assert_ne!(pool[selected[0]], pool[selected[1]]);
    }
}
