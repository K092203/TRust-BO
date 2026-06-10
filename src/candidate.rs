use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Latin Hypercube Sampling: n_samples 点を [0,1]^n_dims に配置。
#[allow(dead_code)]
pub fn lhs(n_samples: usize, n_dims: usize, seed: u64) -> Vec<Vec<f32>> {
    if n_samples == 0 || n_dims == 0 {
        return vec![];
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = vec![vec![0f32; n_dims]; n_samples];
    for j in 0..n_dims {
        let mut perm: Vec<usize> = (0..n_samples).collect();
        for i in (1..n_samples).rev() {
            let k = rng.gen_range(0..=i);
            perm.swap(i, k);
        }
        for i in 0..n_samples {
            let lo = perm[i] as f32 / n_samples as f32;
            let hi = (perm[i] + 1) as f32 / n_samples as f32;
            samples[i][j] = lo + rng.gen::<f32>() * (hi - lo);
        }
    }
    samples
}

// --- Scrambled Halton (準乱数列) ---
// 各次元が独立した素数基底の Van der Corput 列。
// seed でランダムデジットスクランブルを適用して統計的独立性を改善する。

// [CHANGED]: 100D ベンチマーク対応のため素数リストを 64 → 128 次元に拡張。
const PRIMES: [usize; 128] = [
    2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,
   59,  61,  67,  71,  73,  79,  83,  89,  97, 101, 103, 107, 109, 113, 127, 131,
  137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
  227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
  313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
  419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
  509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613,
  617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719,
];

fn van_der_corput(mut n: usize, base: usize) -> f64 {
    let mut result = 0.0f64;
    let mut f = 1.0 / base as f64;
    while n > 0 {
        result += f * (n % base) as f64;
        n /= base;
        f /= base as f64;
    }
    result
}

/// Scrambled Halton 準乱数列。
/// seed が同じなら完全再現。LHS より 50D 以上での空間充填性が均一。
pub fn halton(n_samples: usize, n_dims: usize, seed: u64) -> Vec<Vec<f32>> {
    if n_samples == 0 || n_dims == 0 {
        return vec![];
    }
    assert!(n_dims <= PRIMES.len(), "n_dims {} exceeds Halton max {}", n_dims, PRIMES.len());

    // 次元ごとに固定オフセット (additive scramble) を生成
    let mut rng = StdRng::seed_from_u64(seed);
    let offsets: Vec<f64> = (0..n_dims).map(|_| rng.gen::<f64>()).collect();

    (0..n_samples)
        .map(|i| {
            (0..n_dims)
                .map(|j| {
                    let raw = van_der_corput(i + 1, PRIMES[j]);
                    ((raw + offsets[j]) % 1.0) as f32
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lhs_reproducible() {
        assert_eq!(lhs(5, 3, 42), lhs(5, 3, 42));
    }
    #[test]
    fn lhs_in_unit_cube() {
        for v in lhs(10, 4, 0).iter().flatten() {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }
    #[test]
    fn halton_reproducible() {
        assert_eq!(halton(10, 5, 42), halton(10, 5, 42));
    }
    #[test]
    fn halton_different_seeds() {
        assert_ne!(halton(10, 5, 42), halton(10, 5, 99));
    }
    #[test]
    fn halton_in_unit_cube() {
        for v in halton(20, 10, 0).iter().flatten() {
            assert!(*v >= 0.0 && *v <= 1.0, "value {v} out of [0,1]");
        }
    }
    #[test]
    fn halton_50d_works() {
        let pts = halton(50, 50, 7);
        assert_eq!(pts.len(), 50);
        assert_eq!(pts[0].len(), 50);
    }
}
