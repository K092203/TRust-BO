/// Abramowitz & Stegun (26.2.17) による正規 CDF 近似。誤差 < 7.5e-8。
fn normal_cdf(z: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t * (0.319_381_53
        + t * (-0.356_563_782
            + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = (-z * z / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
    let cdf = 1.0 - phi * poly;
    if z >= 0.0 { cdf } else { 1.0 - cdf }
}

fn normal_pdf(z: f32) -> f32 {
    (-z * z / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt()
}

pub fn ucb(means: &[f32], stds: &[f32], beta: f32) -> Vec<f32> {
    means.iter().zip(stds).map(|(m, s)| m + beta * s).collect()
}

/// Expected Improvement (EI)。内部は最大化。f_best は正規化済み現在最良値。
pub fn ei(means: &[f32], stds: &[f32], f_best: f32) -> Vec<f32> {
    means
        .iter()
        .zip(stds)
        .map(|(m, s)| {
            if *s < 1e-6 {
                return 0.0; // std が極小 → 予測確実 → 改善なし
            }
            let z = (m - f_best) / s;
            ((m - f_best) * normal_cdf(z) + s * normal_pdf(z)).max(0.0)
        })
        .collect()
}

pub fn score(means: &[f32], stds: &[f32], f_best: f32, beta: f32, acq_type: &str) -> Vec<f32> {
    match acq_type {
        "ei" => ei(means, stds, f_best),
        _ => ucb(means, stds, beta), // "ucb" + フォールバック(lib.rs で検証済み)
    }
}

// ── 多目的: 2D Expected Hypervolume Improvement (EHVI、最小化) ──────────────────
//
// 設計 (Phase K-2):
//   2 目的の予測を独立な正規分布 Y_k ~ N(μ_k, σ_k²) と仮定する。
//   EHVI(μ,σ) = E[HV(P ∪ {Y}) − HV(P)]
//             = ∫∫_{I}  P(Y ⪯ w) dw                                   … (★)
//   I は「参照点 r を支配し (w ⪯ r)、既存フロント P には支配されない」改善領域。
//   最小化では P(Y ⪯ w) = Φ((w₁−μ₁)/σ₁)·Φ((w₂−μ₂)/σ₂) と分離できる。
//   I は第 1 目的の昇順フロントを境界とする互いに素な縦ストリップ (箱) に分解でき、
//   各箱上の (★) は 1 次元積分の積になる。Φ の不定積分は
//       G(t;μ,σ) = (t−μ)Φ((t−μ)/σ) + σ·φ((t−μ)/σ),   G(−∞)=0
//   で閉じる。導出は docs を参照。MC 積分との一致は単体テストで検証済み。

fn normal_cdf_f64(z: f64) -> f64 {
    0.5 * erfc_f64(-z / std::f64::consts::SQRT_2)
}

fn normal_pdf_f64(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// 補誤差関数 erfc の有理近似 (Numerical Recipes、相対誤差 < 1.2e-7)。
fn erfc_f64(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Φ の不定積分 G(t;μ,σ) = (t−μ)Φ((t−μ)/σ) + σφ((t−μ)/σ)。G(−∞)=0。
fn g_antideriv(t: f64, mu: f64, sigma: f64) -> f64 {
    if t == f64::NEG_INFINITY {
        return 0.0;
    }
    let s = sigma.max(1e-12);
    let z = (t - mu) / s;
    (t - mu) * normal_cdf_f64(z) + s * normal_pdf_f64(z)
}

/// 1 候補の 2D EHVI (最小化)。
///
/// `front_sorted`: 非支配・第 1 目的昇順ソート済みの 2D フロント (空でも可)。
/// `reference`: 参照点 (各成分が全フロント点より大きいこと)。
/// `mu`,`sigma`: 候補の 2 目的の予測平均・標準偏差。
pub fn ehvi_2d(
    mu: [f64; 2],
    sigma: [f64; 2],
    front_sorted: &[[f64; 2]],
    reference: [f64; 2],
) -> f64 {
    // 第 1 目的のストリップ境界 x_0=-∞, x_1..x_n=front.x, x_{n+1}=r₁
    // 各ストリップ i の第 2 目的の天井 ceil_i: i=1 は r₂、i>=2 は front[i-2].y
    // (front は昇順 x ⇒ 降順 y なので、左側にある点のうち最小 y が天井)
    let n = front_sorted.len();
    let mut ehvi = 0.0;
    let (mu1, mu2) = (mu[0], mu[1]);
    let (s1, s2) = (sigma[0], sigma[1]);

    for i in 0..=n {
        // ストリップ i (0-indexed): x ∈ [x_lo, x_hi]
        let x_lo = if i == 0 {
            f64::NEG_INFINITY
        } else {
            front_sorted[i - 1][0]
        };
        let x_hi = if i == n {
            reference[0]
        } else {
            front_sorted[i][0]
        };
        // 天井 (第 2 目的): i=0 は参照点、i>=1 は左隣フロント点の y
        let y_ceil = if i == 0 {
            reference[1]
        } else {
            front_sorted[i - 1][1]
        };
        if x_hi <= x_lo {
            continue; // 退化ストリップ (重複 x など)
        }
        let int_x = g_antideriv(x_hi, mu1, s1) - g_antideriv(x_lo, mu1, s1);
        let int_y = g_antideriv(y_ceil, mu2, s2); // 下端 -∞ で G(-∞)=0
        ehvi += int_x * int_y;
    }
    ehvi.max(0.0)
}

/// バッチ版: 候補ごとの EHVI を返す。
pub fn ehvi_2d_batch(
    means: &[[f64; 2]],
    stds: &[[f64; 2]],
    front_sorted: &[[f64; 2]],
    reference: [f64; 2],
) -> Vec<f32> {
    means
        .iter()
        .zip(stds)
        .map(|(&m, &s)| ehvi_2d(m, s, front_sorted, reference) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypervolume::hypervolume_2d;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn box_muller(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-12);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// モンテカルロで EHVI を推定する (参照実装)。
    fn ehvi_mc(
        mu: [f64; 2],
        sigma: [f64; 2],
        front_sorted: &[[f64; 2]],
        reference: [f64; 2],
        n_samples: usize,
        seed: u64,
    ) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let hv_base = hypervolume_2d(front_sorted, reference);
        let mut acc = 0.0;
        for _ in 0..n_samples {
            let y = [
                mu[0] + sigma[0] * box_muller(&mut rng),
                mu[1] + sigma[1] * box_muller(&mut rng),
            ];
            let mut pts = front_sorted.to_vec();
            pts.push(y);
            let hv_new = hypervolume_2d(&pts, reference);
            acc += (hv_new - hv_base).max(0.0);
        }
        acc / n_samples as f64
    }

    fn assert_close_mc(
        mu: [f64; 2],
        sigma: [f64; 2],
        front: &[[f64; 2]],
        reference: [f64; 2],
    ) {
        let exact = ehvi_2d(mu, sigma, front, reference);
        let mc = ehvi_mc(mu, sigma, front, reference, 400_000, 12345);
        let tol = 0.03 * exact.max(0.05) + 0.01;
        assert!(
            (exact - mc).abs() < tol,
            "EHVI mismatch: exact={exact:.5}, mc={mc:.5}, tol={tol:.5}"
        );
    }

    #[test]
    fn ehvi_empty_front_equals_box_probability() {
        // フロント空: EHVI = E[HV(単一点)] = G(r₁)·G(r₂)
        assert_close_mc([0.0, 0.0], [1.0, 1.0], &[], [3.0, 3.0]);
    }

    #[test]
    fn ehvi_single_front_point() {
        let front = vec![[1.0, 1.0]];
        assert_close_mc([0.5, 0.5], [0.7, 0.7], &front, [3.0, 3.0]);
    }

    #[test]
    fn ehvi_staircase_front() {
        let front = vec![[0.5, 2.5], [1.5, 1.5], [2.5, 0.5]];
        assert_close_mc([1.0, 1.0], [0.8, 0.8], &front, [3.0, 3.0]);
    }

    #[test]
    fn ehvi_anisotropic_sigma() {
        let front = vec![[0.5, 2.5], [2.0, 1.0]];
        assert_close_mc([1.2, 1.2], [1.5, 0.4], &front, [3.0, 3.0]);
    }

    #[test]
    fn ehvi_candidate_far_from_ref_is_small() {
        // 平均が参照点の外 (悪い側) → EHVI は小さい
        let front = vec![[1.0, 1.0]];
        let e = ehvi_2d([5.0, 5.0], [0.3, 0.3], &front, [3.0, 3.0]);
        assert!(e < 1e-3, "expected ~0, got {e}");
    }

    #[test]
    fn ehvi_nonnegative_and_monotone_in_promise() {
        // 良い (小さい) 平均の候補は悪い平均より EHVI が大きい
        let front = vec![[1.0, 1.0]];
        let good = ehvi_2d([0.2, 0.2], [0.5, 0.5], &front, [3.0, 3.0]);
        let bad = ehvi_2d([0.9, 0.9], [0.5, 0.5], &front, [3.0, 3.0]);
        assert!(good > bad && bad >= 0.0, "good={good}, bad={bad}");
    }
}
