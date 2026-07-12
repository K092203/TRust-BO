use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer as _},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{activation::relu, backend::Backend, Tensor, TensorData},
};

// AB を共通型として使う。valid() の型推論問題を回避するため、
// train 後も AB モデルのまま保持する(勾配グラフは inference 時に生成されるが backward を呼ばないので無害)。
pub type B = NdArray<f32>;
pub type AB = Autodiff<B>;

#[derive(Module, Debug, Clone)]
pub struct Mlp<Bk: Backend> {
    l1: Linear<Bk>,
    l2: Linear<Bk>,
    l3: Linear<Bk>,
    out: Linear<Bk>,
}

impl<Bk: Backend> Mlp<Bk> {
    fn new(n_dims: usize, device: &Bk::Device) -> Self {
        Self {
            l1: LinearConfig::new(n_dims, 64).with_bias(true).init(device),
            l2: LinearConfig::new(64, 64).with_bias(true).init(device),
            l3: LinearConfig::new(64, 32).with_bias(true).init(device),
            out: LinearConfig::new(32, 1).with_bias(true).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let x = relu(self.l1.forward(x));
        let x = relu(self.l2.forward(x));
        let x = relu(self.l3.forward(x));
        self.out.forward(x)
    }
}

fn bytes_to_hex(b: &[u8]) -> String {
    b.iter().fold(String::with_capacity(b.len() * 2), |mut s, &byte| {
        s.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((byte & 0xf) as u32, 16).unwrap());
        s
    })
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

pub struct Ensemble {
    pub members: Vec<Mlp<AB>>,
    /// σ (メンバー間分散) に掛ける温度係数。校正無効時は 1.0 (predict の挙動は不変)。
    pub sigma_scale: f32,
}

impl Ensemble {
    /// Bootstrap アンサンブル: メンバーごとに復元抽出でデータをサンプリングし、
    /// 独立した予測分布の多様性を確保する。
    /// warm_states: 前回 propose() の model_states (hex 文字列)。Some(&[]) の場合も cold start と同等。
    /// calibrate_sigma: true の場合、末尾 min(n/5, 20) 件 (n<20 ならスキップ) を
    /// ホールドアウトとして温存し、NLL 最小化の閉形式解 (温度スケーリング) で
    /// predict() の σ に掛ける係数を求める。ホールドアウト選択は決定的
    /// (データ末尾を固定的に切り出すのみ) なので RNG 消費列は変化しない。
    /// 戻り値: (Ensemble, mean_final_loss, model_states_hex)
    pub fn train(
        params: &[Vec<f32>],
        norm_values: &[f32],
        n_dims: usize,
        n_members: usize,
        seed: u64,
        epochs: usize,
        lr: f64,
        warm_states: Option<&[String]>,
        calibrate_sigma: bool,
    ) -> (Self, f32, Vec<String>) {
        let device: <AB as Backend>::Device = Default::default();
        let n_total = params.len();
        let n_holdout = if calibrate_sigma {
            let k = (n_total / 5).min(20);
            if n_total.saturating_sub(k) >= n_dims + 2 {
                k
            } else {
                0
            }
        } else {
            0
        };
        let n_train = n_total - n_holdout;
        let params_train = &params[..n_train];
        let norm_values_train = &norm_values[..n_train];
        let n = params_train.len();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        let results: Vec<(Mlp<AB>, f32, String)> = (0..n_members)
            .map(|i| {
                B::seed(seed.wrapping_add(i as u64 * 0x9e3779b97f4a7c15));

                let mut boot_rng = StdRng::seed_from_u64(
                    seed.wrapping_add(i as u64 * 0x1234_5678_9abc_def0),
                );
                let boot_idx: Vec<usize> = (0..n).map(|_| boot_rng.gen_range(0..n)).collect();
                let xs_boot: Vec<f32> = boot_idx
                    .iter()
                    .flat_map(|&j| params_train[j].iter().copied())
                    .collect();
                let ys_boot: Vec<f32> = boot_idx.iter().map(|&j| norm_values_train[j]).collect();

                // ウォームスタート: 前回の weights から開始する (なければ cold start)。
                // n_holdout>0 (校正実行時) は強制的に cold start する — 前回までの
                // 全履歴で学習済みの重みを引き継ぐと、今回ホールドアウトした点も
                // 既にモデルへ「見られて」おり out-of-sample にならない
                // (sol監査で指摘された漏洩)。校正時は速度より正しさを優先する。
                let warm_hex = if n_holdout > 0 {
                    None
                } else {
                    warm_states.and_then(|ws| ws.get(i))
                };
                let mut model = if let Some(hex) = warm_hex.filter(|h| !h.is_empty()) {
                    let bytes = hex_to_bytes(hex);
                    let load_result: Result<<Mlp<AB> as Module<AB>>::Record, _> =
                        <BinBytesRecorder<FullPrecisionSettings> as Recorder<AB>>::load(
                            &recorder,
                            bytes,
                            &device,
                        );
                    match load_result {
                        Ok(record) => <Mlp<AB> as Module<AB>>::load_record(
                            Mlp::<AB>::new(n_dims, &device),
                            record,
                        ),
                        Err(_) => Mlp::<AB>::new(n_dims, &device),
                    }
                } else {
                    Mlp::<AB>::new(n_dims, &device)
                };

                let mut optim = AdamConfig::new()
                    .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
                    .init::<AB, Mlp<AB>>();

                let mut final_loss = 1.0f32;
                let mut prev_loss = f32::INFINITY;
                // ウォームスタート時は収束が速いため、より頻繁に収束チェックを行う
                let check_every = if warm_hex.filter(|h| !h.is_empty()).is_some() { 10 } else { 50 };
                let x = Tensor::<AB, 2>::from_data(
                    TensorData::new(xs_boot.clone(), [n, n_dims]),
                    &device,
                );
                let y_true = Tensor::<AB, 2>::from_data(
                    TensorData::new(ys_boot.clone(), [n, 1]),
                    &device,
                );
                for ep in 0..epochs {
                    let diff = model.forward(x.clone()) - y_true.clone();
                    let loss = (diff.clone() * diff).mean();
                    final_loss = loss.clone().into_scalar();
                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads::<AB, Mlp<AB>>(grads, &model);
                    model = optim.step(lr, model, grads);
                    // 50 epoch ごとに収束チェック: 相対改善 < 1% なら打ち切り
                    if ep % check_every == check_every - 1 {
                        let rel_impr = (prev_loss - final_loss) / (prev_loss + 1e-8);
                        if rel_impr < 0.01 {
                            break;
                        }
                        prev_loss = final_loss;
                    }
                }

                // 次の propose() へ渡すウォームスタート用 weights を保存
                let record = <Mlp<AB> as Module<AB>>::into_record(model.clone());
                let bytes = <BinBytesRecorder<FullPrecisionSettings> as Recorder<AB>>::record(
                    &recorder,
                    record,
                    (),
                )
                .unwrap_or_default();
                let hex = bytes_to_hex(&bytes);

                (model, final_loss, hex)
            })
            .collect();

        let mean_loss = results.iter().map(|(_, l, _)| l).sum::<f32>() / n_members as f32;
        let states: Vec<String> = results.iter().map(|(_, _, h)| h.clone()).collect();
        let members: Vec<Mlp<AB>> = results.into_iter().map(|(m, _, _)| m).collect();

        let mut ensemble = Ensemble { members, sigma_scale: 1.0 };
        if n_holdout > 0 {
            // ホールドアウトはデータ末尾 (時系列で最新の feasible 点 = 現在の TR 近傍)
            // から取る。z-score (norm_values) は全データ(ホールドアウト込み)から
            // 計算済みのため軽微な情報漏洩が残るが、影響は無視できる規模と判断
            // (n が小さくない限り平均・分散はホールドアウト有無でほぼ不変)。
            let holdout_params = &params[n_train..];
            let holdout_values = &norm_values[n_train..];
            let (ho_means, ho_stds) = ensemble.predict(holdout_params, n_dims);
            // NLL 最小化の閉形式解 (正規分布、平均固定): c^2 = mean((y-mu)^2 / sigma^2)。
            // 非有限項は除外し、有効項が無ければ校正をスキップ (sigma_scale=1.0 のまま)。
            let sq_ratios: Vec<f32> = holdout_values
                .iter()
                .zip(ho_means.iter())
                .zip(ho_stds.iter())
                .map(|((&y, &mu), &s)| ((y - mu) / s).powi(2))
                .filter(|v| v.is_finite())
                .collect();
            if !sq_ratios.is_empty() {
                let mse_ratio = sq_ratios.iter().sum::<f32>() / sq_ratios.len() as f32;
                let c = mse_ratio.sqrt();
                // 縮小方向は許可しない (下限 1.0): このプロジェクトではσを縮める変更が
                // 過去に探索を大きく阻害した実績があるため (コヒーレント単一メンバーTS、
                // 比1.375の悪化)。校正は「過小評価の補正」のみに用途を絞る。
                if c.is_finite() {
                    ensemble.sigma_scale = c.clamp(1.0, 5.0);
                }
            }
        }
        (ensemble, mean_loss, states)
    }

    pub fn predict(&self, candidates: &[Vec<f32>], n_dims: usize) -> (Vec<f32>, Vec<f32>) {
        let device: <AB as Backend>::Device = Default::default();
        let n = candidates.len();
        let flat: Vec<f32> = candidates.iter().flat_map(|r| r.iter().copied()).collect();

        let all_preds: Vec<Vec<f32>> = self
            .members
            .iter()
            .map(|m| {
                let t = Tensor::<AB, 2>::from_data(
                    TensorData::new(flat.clone(), [n, n_dims]),
                    &device,
                );
                m.forward(t).into_data().to_vec().unwrap()
            })
            .collect();

        let k = self.members.len() as f32;
        let means: Vec<f32> = (0..n)
            .map(|i| all_preds.iter().map(|p| p[i]).sum::<f32>() / k)
            .collect();
        let stds: Vec<f32> = (0..n)
            .map(|i| {
                let m = means[i];
                ((all_preds.iter().map(|p| (p[i] - m).powi(2)).sum::<f32>() / k).sqrt()
                    * self.sigma_scale)
                    .max(1e-8)
            })
            .collect();

        (means, stds)
    }
}
