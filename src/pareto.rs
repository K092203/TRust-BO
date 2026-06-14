//! 多目的: 非支配ソートと 2D Pareto フロント構築 (すべて最小化前提)。
//!
//! Python 側 (`multiobjective.py::_pareto_mask`) と同じ判定ロジックを Rust に移植。
//! 最大化目的は呼び出し側で符号反転して最小化空間に変換しておくこと。

use std::cmp::Ordering;

/// `a` が `b` を支配するか (最小化: 全成分 a<=b かつ 少なくとも一つ a<b)。
pub fn dominates(a: &[f32], b: &[f32]) -> bool {
    let mut strictly = false;
    for (x, y) in a.iter().zip(b) {
        if x > y {
            return false;
        }
        if x < y {
            strictly = true;
        }
    }
    strictly
}

/// 最小化における非支配解のインデックスを返す。O(n^2 · m)。
///
/// 同一点 (重複) は互いに支配しないため両方残る (Python 実装と一致)。
pub fn nondominated_indices(costs: &[Vec<f32>]) -> Vec<usize> {
    let n = costs.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !keep[j] {
                continue;
            }
            // j が i を支配するなら i を落とす。支配の推移性により、
            // 既に落とした j をスキップしても i は別の支配者に捕捉される。
            if dominates(&costs[j], &costs[i]) {
                keep[i] = false;
                break;
            }
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// 2D 最小化フロント: 非支配点を第 1 目的の昇順 (→ 第 2 目的降順) でソートして返す。
///
/// EHVI / hypervolume のセル分解はこの「昇順 x・降順 y」の階段構造を前提とする。
pub fn sorted_front_2d(costs: &[Vec<f32>]) -> Vec<[f64; 2]> {
    debug_assert!(costs.iter().all(|c| c.len() >= 2));
    let idx = nondominated_indices(costs);
    let mut front: Vec<[f64; 2]> = idx
        .iter()
        .map(|&i| [costs[i][0] as f64, costs[i][1] as f64])
        .collect();
    front.sort_by(|a, b| {
        a[0]
            .partial_cmp(&b[0])
            .unwrap_or(Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(Ordering::Equal))
    });
    front
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominates_basic() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 2.0], &[1.0, 3.0])); // 一成分等しく一成分小さい
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0])); // トレードオフ → 非支配
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0])); // 同一点は支配しない
    }

    #[test]
    fn nondominated_simple() {
        let costs = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.0],
            vec![4.0, 1.0],
            vec![3.0, 3.0], // (2,2) に支配される
        ];
        let mut nd = nondominated_indices(&costs);
        nd.sort();
        assert_eq!(nd, vec![0, 1, 2]);
    }

    #[test]
    fn nondominated_keeps_duplicates() {
        let costs = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let nd = nondominated_indices(&costs);
        assert_eq!(nd.len(), 2);
    }

    #[test]
    fn sorted_front_orders_ascending_x() {
        let costs = vec![vec![4.0, 1.0], vec![1.0, 4.0], vec![2.0, 2.0]];
        let front = sorted_front_2d(&costs);
        assert_eq!(front.len(), 3);
        assert!(front[0][0] < front[1][0] && front[1][0] < front[2][0]);
        // 第 2 目的は降順
        assert!(front[0][1] > front[1][1] && front[1][1] > front[2][1]);
    }
}
