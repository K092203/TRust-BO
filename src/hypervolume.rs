//! 2D 超体積指標 (最小化)。スイープライン O(n log n)。
//!
//! 参照点 `reference` は全フロント点を支配される側 (各成分が大きい) であること。

use std::cmp::Ordering;

/// 任意の 2D 点群に対する超体積 (最小化)。
///
/// 入力は非支配でなくてもよい (内部で参照点フィルタ + 階段抽出を行う)。
/// 参照点に支配されない点は寄与 0 として無視される。
pub fn hypervolume_2d(points: &[[f64; 2]], reference: [f64; 2]) -> f64 {
    // 参照点を真に支配する点のみ採用
    let mut pts: Vec<[f64; 2]> = points
        .iter()
        .filter(|p| p[0] < reference[0] && p[1] < reference[1])
        .copied()
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    // 第 1 目的昇順、同値は第 2 目的昇順
    pts.sort_by(|a, b| {
        a[0]
            .partial_cmp(&b[0])
            .unwrap_or(Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(Ordering::Equal))
    });

    // 単調降順の階段 (非支配集合) を抽出: x 昇順で y が厳密に小さくなる点のみ残す
    let mut staircase: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    let mut min_y = f64::INFINITY;
    for p in pts {
        if p[1] < min_y {
            staircase.push(p);
            min_y = p[1];
        }
    }

    // 各点が [x_i, x_{i+1}) × [y_i, ref_y) の縦ストリップを占める
    let mut hv = 0.0;
    for i in 0..staircase.len() {
        let x = staircase[i][0];
        let y = staircase[i][1];
        let x_right = if i + 1 < staircase.len() {
            staircase[i + 1][0]
        } else {
            reference[0]
        };
        hv += (x_right - x) * (reference[1] - y);
    }
    hv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point() {
        // 点 (1,1)、参照 (3,3) → 面積 (3-1)*(3-1) = 4
        let hv = hypervolume_2d(&[[1.0, 1.0]], [3.0, 3.0]);
        assert!((hv - 4.0).abs() < 1e-9, "hv={hv}");
    }

    #[test]
    fn two_point_staircase() {
        // (1,2),(2,1), ref (3,3)
        // strip1: x in [1,2], height 3-2=1 → 1
        // strip2: x in [2,3], height 3-1=2 → 2
        // 合計 3
        let hv = hypervolume_2d(&[[1.0, 2.0], [2.0, 1.0]], [3.0, 3.0]);
        assert!((hv - 3.0).abs() < 1e-9, "hv={hv}");
    }

    #[test]
    fn dominated_point_ignored() {
        // (1,1) が (2,2) を支配 → (2,2) は寄与しない、面積 = 4
        let hv = hypervolume_2d(&[[1.0, 1.0], [2.0, 2.0]], [3.0, 3.0]);
        assert!((hv - 4.0).abs() < 1e-9, "hv={hv}");
    }

    #[test]
    fn point_outside_reference_ignored() {
        let hv = hypervolume_2d(&[[5.0, 5.0]], [3.0, 3.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn empty_is_zero() {
        let hv = hypervolume_2d(&[], [3.0, 3.0]);
        assert_eq!(hv, 0.0);
    }
}
