pub fn zscore(vals: &[f32]) -> (Vec<f32>, f32, f32) {
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    // Bessel 補正 (不偏分散): n-1 で割る。n=1 時は 1 にクランプ。
    let denom = (n - 1.0).max(1.0);
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / denom)
        .sqrt()
        .max(1e-8);
    (vals.iter().map(|v| (v - mean) / std).collect(), mean, std)
}
