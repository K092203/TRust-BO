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
