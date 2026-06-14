// Micro-GP (Matern 5/2) — Tandem Residual-GP の Phase 2 サロゲート。
// 自己完結モジュール: std + rand のみに依存し crate 内 import を持たない
// (branin_demo が #[path] で取り込んで同一コードを検証するため)。
// 内部計算は全て f64。外部 BLAS/LAPACK は使わない。
use rand::rngs::StdRng;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpError {
    /// 重複除去後の点数が n_dims + 2 未満
    TooFewPoints,
    /// 残差の標準偏差がほぼゼロ (GP で学習する情報がない)
    DegenerateResiduals,
    /// 全ハイパーパラメータ候補で Cholesky 失敗または尤度が非有限
    CholeskyFailed,
    /// 入力に NaN / inf が含まれる
    NonFinite,
}

#[derive(Debug)]
pub struct MicroGp {
    /// ARD: n_dims 本 / isotropic: 1 本
    lengthscales: Vec<f64>,
    sf2: f64,
    chol_l: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    x_data: Vec<Vec<f64>>,
    // 残差は標準化して fit し、predict で復元する
    r_mean: f64,
    r_scale: f64,
}

/// 残差 r に Micro-GP をフィットする。panic しない。
/// 失敗時は Err を返し、呼び出し側は global CEM へフォールバックする契約。
pub fn fit_micro_gp(xs: &[Vec<f64>], r: &[f64], rng: &mut StdRng) -> Result<MicroGp, GpError> {
    let n_dims = match xs.first() {
        Some(x) if !x.is_empty() => x.len(),
        _ => return Err(GpError::TooFewPoints),
    };
    if xs.len() != r.len() {
        return Err(GpError::TooFewPoints);
    }
    if r.iter().any(|v| !v.is_finite()) || xs.iter().flatten().any(|v| !v.is_finite()) {
        return Err(GpError::NonFinite);
    }

    // 近接重複点 (L∞ < 1e-9) を除去 — Cholesky の特異性を防ぐ
    let mut keep: Vec<usize> = Vec::with_capacity(xs.len());
    'outer: for i in 0..xs.len() {
        for &j in &keep {
            if xs[i].iter().zip(&xs[j]).all(|(a, b)| (a - b).abs() < 1e-9) {
                continue 'outer;
            }
        }
        keep.push(i);
    }
    if keep.len() < n_dims + 2 {
        return Err(GpError::TooFewPoints);
    }
    let xs_k: Vec<Vec<f64>> = keep.iter().map(|&i| xs[i].clone()).collect();
    let r_k: Vec<f64> = keep.iter().map(|&i| r[i]).collect();
    let n = xs_k.len();

    // 残差標準化
    let r_mean = r_k.iter().sum::<f64>() / n as f64;
    let std_r =
        (r_k.iter().map(|v| (v - r_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if std_r < 1e-10 {
        return Err(GpError::DegenerateResiduals);
    }
    let r_scale = std_r.max(1e-8);
    let rn: Vec<f64> = r_k.iter().map(|v| (v - r_mean) / r_scale).collect();

    // ハイパーパラメータ多起点ランダム探索 (周辺対数尤度最大化)。
    // 高次元 ARD は候補数に対して不利なため n_dims > 10 は isotropic に切替。
    // d > 10 は isotropic（探索空間 3 次元固定）なので線形スケールは不要。
    // 200D で n_hypers=800 になり 544s/run になっていた問題を修正。
    let n_ls = if n_dims <= 10 { n_dims } else { 1 };
    let n_hypers = if n_dims <= 10 { 40.max(4 * n_dims) } else { 60 };
    let (ln_ls_lo, ln_ls_hi) = ((0.05f64).ln(), (2.0f64).ln());
    let (ln_nz_lo, ln_nz_hi) = ((1e-4f64).ln(), (1e-1f64).ln());

    let mut best: Option<(f64, Vec<f64>, f64, Vec<Vec<f64>>, Vec<f64>)> = None;
    for _ in 0..n_hypers {
        let ls: Vec<f64> = (0..n_ls)
            .map(|_| rng.gen_range(ln_ls_lo..ln_ls_hi).exp())
            .collect();
        let sf2: f64 = rng.gen_range(0.2..3.0);
        let noise = rng.gen_range(ln_nz_lo..ln_nz_hi).exp().max(1e-6);

        let k = build_k(&xs_k, &ls, sf2, noise);
        let Some(l) = cholesky_jitter(k) else { continue };
        let y1 = forward_subst(&l, &rn);
        let alpha = backward_subst_lt(&l, &y1);
        let quad: f64 = 0.5 * rn.iter().zip(&alpha).map(|(a, b)| a * b).sum::<f64>();
        let logdet: f64 = (0..n).map(|i| l[i][i].ln()).sum();
        let logml = -quad - logdet - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
        if !logml.is_finite() {
            continue;
        }
        if best.as_ref().map_or(true, |b| logml > b.0) {
            best = Some((logml, ls, sf2, l, alpha));
        }
    }

    let Some((_, lengthscales, sf2, chol_l, alpha)) = best else {
        return Err(GpError::CholeskyFailed);
    };
    Ok(MicroGp { lengthscales, sf2, chol_l, alpha, x_data: xs_k, r_mean, r_scale })
}

/// 事後予測 (mu, var)。var >= 1e-12 を保証。標準化を復元して返す。
pub fn gp_predict(gp: &MicroGp, x: &[f64]) -> (f64, f64) {
    let n = gp.x_data.len();
    let kstar: Vec<f64> = (0..n)
        .map(|i| matern52(&gp.x_data[i], x, &gp.lengthscales, gp.sf2))
        .collect();
    let mu_n: f64 = kstar.iter().zip(&gp.alpha).map(|(a, b)| a * b).sum();
    let v = forward_subst(&gp.chol_l, &kstar);
    let var_n = (gp.sf2 - v.iter().map(|x| x * x).sum::<f64>()).max(1e-12);
    (
        mu_n * gp.r_scale + gp.r_mean,
        (var_n * gp.r_scale * gp.r_scale).max(1e-12),
    )
}

// ── カーネル / 線形代数 ──────────────────────────────────────────────────────

/// Matern 5/2。ls が 1 本なら isotropic、n_dims 本なら ARD。
fn matern52(x1: &[f64], x2: &[f64], ls: &[f64], sf2: f64) -> f64 {
    let mut r2 = 0.0;
    for d in 0..x1.len() {
        let l = if ls.len() == 1 { ls[0] } else { ls[d] };
        let diff = (x1[d] - x2[d]) / l;
        r2 += diff * diff;
    }
    let r = r2.sqrt();
    let s5 = 5.0_f64.sqrt();
    sf2 * (1.0 + s5 * r + 5.0 * r2 / 3.0) * (-s5 * r).exp()
}

fn build_k(xs: &[Vec<f64>], ls: &[f64], sf2: f64, noise: f64) -> Vec<Vec<f64>> {
    let n = xs.len();
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            k[i][j] = matern52(&xs[i], &xs[j], ls, sf2);
        }
        k[i][i] += noise;
    }
    k
}

/// K = L Lᵀ。非正定値なら None (panic しない)。
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// jitter を 1e-6 から 10 倍ずつ最大 8 回加えて Cholesky を試す。全失敗で None。
fn cholesky_jitter(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    if let Some(l) = cholesky(&a) {
        return Some(l);
    }
    let n = a.len();
    let mut add = 1e-6;
    for _ in 0..8 {
        for i in 0..n {
            a[i][i] += add;
        }
        if let Some(l) = cholesky(&a) {
            return Some(l);
        }
        add *= 10.0;
    }
    None
}

fn forward_subst(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    y
}

fn backward_subst_lt(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(0)
    }

    /// 滑らかな関数の残差を GP が補間できる
    #[test]
    fn fit_and_predict_recovers_smooth_function() {
        let mut r = rng();
        let xs: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 / 19.0, (i as f64 / 19.0).powi(2)])
            .collect();
        let ys: Vec<f64> = xs.iter().map(|x| (3.0 * x[0]).sin() + x[1]).collect();
        let gp = fit_micro_gp(&xs, &ys, &mut r).expect("fit failed");
        let (mu, var) = gp_predict(&gp, &xs[10]);
        assert!((mu - ys[10]).abs() < 0.3, "mu={mu} vs y={}", ys[10]);
        assert!(var >= 1e-12);
    }

    #[test]
    fn too_few_points_errors() {
        let xs = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let ys = vec![0.0, 1.0];
        assert_eq!(fit_micro_gp(&xs, &ys, &mut rng()).unwrap_err(), GpError::TooFewPoints);
    }

    #[test]
    fn degenerate_residuals_errors() {
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.5]).collect();
        let ys = vec![2.5; 10];
        assert_eq!(
            fit_micro_gp(&xs, &ys, &mut rng()).unwrap_err(),
            GpError::DegenerateResiduals
        );
    }

    #[test]
    fn nan_input_errors() {
        let mut xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 1.0]).collect();
        xs[3][0] = f64::NAN;
        let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(fit_micro_gp(&xs, &ys, &mut rng()).unwrap_err(), GpError::NonFinite);
    }

    /// 重複点は除去され、残りが足りなければ TooFewPoints
    #[test]
    fn duplicates_are_removed() {
        let xs = vec![vec![1.0, 2.0]; 30];
        let ys: Vec<f64> = (0..30).map(|i| i as f64).collect();
        assert_eq!(fit_micro_gp(&xs, &ys, &mut rng()).unwrap_err(), GpError::TooFewPoints);
    }

    /// isotropic 切替 (n_dims > 10) でも fit が通る
    #[test]
    fn high_dim_isotropic_fits() {
        let mut r = rng();
        use rand::Rng;
        let d = 20;
        let xs: Vec<Vec<f64>> = (0..40)
            .map(|_| (0..d).map(|_| r.gen::<f64>()).collect())
            .collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.iter().sum::<f64>()).collect();
        let gp = fit_micro_gp(&xs, &ys, &mut r).expect("high-dim fit failed");
        let (_, var) = gp_predict(&gp, &xs[0]);
        assert!(var.is_finite());
    }
}
