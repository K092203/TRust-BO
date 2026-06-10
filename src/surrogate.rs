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
}

impl Ensemble {
    /// Bootstrap アンサンブル: メンバーごとに復元抽出でデータをサンプリングし、
    /// 独立した予測分布の多様性を確保する。
    /// warm_states: 前回 propose() の model_states (hex 文字列)。Some(&[]) の場合も cold start と同等。
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
    ) -> (Self, f32, Vec<String>) {
        let device: <AB as Backend>::Device = Default::default();
        let n = params.len();
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
                    .flat_map(|&j| params[j].iter().copied())
                    .collect();
                let ys_boot: Vec<f32> = boot_idx.iter().map(|&j| norm_values[j]).collect();

                // ウォームスタート: 前回の weights から開始する (なければ cold start)
                let warm_hex = warm_states.and_then(|ws| ws.get(i));
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
                for ep in 0..epochs {
                    let x = Tensor::<AB, 2>::from_data(
                        TensorData::new(xs_boot.clone(), [n, n_dims]),
                        &device,
                    );
                    let y_true = Tensor::<AB, 2>::from_data(
                        TensorData::new(ys_boot.clone(), [n, 1]),
                        &device,
                    );
                    let diff = model.forward(x) - y_true;
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
        let members = results.into_iter().map(|(m, _, _)| m).collect();
        (Ensemble { members }, mean_loss, states)
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
                (all_preds.iter().map(|p| (p[i] - m).powi(2)).sum::<f32>() / k)
                    .sqrt()
                    .max(1e-8)
            })
            .collect();

        (means, stds)
    }
}
