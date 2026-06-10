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
