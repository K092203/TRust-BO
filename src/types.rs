use serde::{Deserialize, Serialize};

/// Trust Region の状態。Python 側で永続化し、propose() 呼び出しごとに渡す。
/// best_value は生の値(最大化方向統一済み、z-score 前)。
/// z-score 正規化済みで保存すると、新データ追加後にスケールがずれて成否判定が壊れる。
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrustRegionState {
    pub center: Vec<f32>,
    pub side_length: f32,
    pub success_count: usize,
    pub failure_count: usize,
    pub best_value: f32,
    pub active: bool,
    /// 初期 warmup 残り propose() 呼び出し数。この間は TR 収縮しない。
    /// 旧バージョンとの互換性のため #[serde(default)] で 0 にフォールバック。
    #[serde(default)]
    pub warmup_remaining: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProposeConfig {
    pub n_dims: usize,
    pub batch_size: usize,
    pub n_init: usize,
    pub ensemble_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub n_cem_samples: usize,
    pub n_cem_iters: usize,
    pub elite_fraction: f32,
    pub beta: f32,
    pub acquisition: String,
    // Trust Region (Phase 3)
    pub tau_succ: usize,   // 成功バッチ連続数 → 辺長拡大
    pub tau_fail: usize,   // 失敗バッチ連続数 → 辺長縮小 (0 = n_dims ベース自動設定)
    pub l_init: f32,       // 初期辺長
    pub l_max: f32,        // 最大辺長
    pub l_min: f32,        // 最小辺長 (以下でリスタート)
    /// サロゲートウォームスタート: 前回の weights (hex 文字列、メンバーごと)
    #[serde(default)]
    pub model_states: Vec<String>,
    /// Feasibility surrogate のウォームスタート weights
    #[serde(default)]
    pub feas_model_states: Vec<String>,
    /// 並列 Trust Region 数 (TuRBO-M)。1 = 従来の単一 TR (デフォルト、後方互換)。
    #[serde(default = "default_n_trs")]
    pub n_trs: usize,
    /// TR 初期化時の warmup ラウンド数。この間は failure_count を増やさない。
    /// デフォルト 0 (無効)。早期収縮が懸念される場合は 2–3 を推奨。
    /// 注意: テスト seed=0 で l_init=0.5 のとき warmup>0 だと expand→shrink で
    /// side_length が l_init に戻る可能性があるため、デフォルトは 0 に保つ。
    // [CHANGED]: 修正① — 早期 TR 収縮防止の warmup 機能を configurable として実装。
    // デフォルト 0 で既存テストに影響なし。本番では init_warmup=2 推奨。
    #[serde(default)]
    pub init_warmup: usize,
    /// Tandem Residual-GP Phase 2 を有効化 (単一 TR のみ)。デフォルト false で既存挙動。
    #[serde(default)]
    pub enable_phase2: bool,
    /// 現在の phase ("global" | "local")。Python 側で往復させる sticky 状態。
    #[serde(default = "default_phase")]
    pub phase: String,
    /// EI 停滞カウンタ (max EI < 1e-5 の連続回数)。Python 側で往復。
    #[serde(default)]
    pub stagnation_count: usize,
    /// Phase 2 遷移に必要な最小 feasible 評価数。0 = 自動 (3 × n_init)。
    #[serde(default)]
    pub phase2_min_evals: usize,
    /// local GP の学習点数。0 = 自動 (max(50, n_dims + 2))。
    #[serde(default)]
    pub phase2_local_points: usize,
    /// Phase 2 GP の長さスケールに次元スケール LogNormal 事前 (Hvarfner 2024) を課す。
    /// デフォルト false で既存挙動 (MLE) と完全一致。
    #[serde(default)]
    pub phase2_ls_prior: bool,
    /// Phase 2 早期発火: TR 辺長が l_init × この値以下になったら TR 枯渇を待たず
    /// local 遷移を許可する。0.0 (デフォルト) で無効 = 既存挙動 (tr_exhausted / EI停滞のみ)。
    #[serde(default)]
    pub phase2_early_frac: f32,
    /// RAASP 型次元マスク (Xu et al. ICML 2025): グローバル CEM のサンプル生成で
    /// 各次元を確率 min(1, 20/d) でのみ摂動し、残りは CEM 平均に固定する。
    /// 高次元での全次元同時摂動による局所性崩壊への対処。デフォルト false で既存挙動。
    #[serde(default)]
    pub cem_dim_mask: bool,
    /// bilog 出力変換 (SCBO/HEBO): サロゲート学習前の目的値に sgn(v)·ln(1+|v|) を適用し
    /// 外れ値を減衰する。TR の best_value (生値) には影響しない。デフォルト false。
    #[serde(default)]
    pub bilog_transform: bool,
}

fn default_n_trs() -> usize {
    1
}

fn default_phase() -> String {
    "global".to_string()
}

// ── 多目的 (Phase K-2: EHVI) ───────────────────────────────────────────────────

/// 多目的 propose_mo 用の設定。すべて最小化空間を前提とする
/// (最大化目的は Python 側で符号反転して渡す)。
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProposeMoConfig {
    pub n_dims: usize,
    pub n_obj: usize,
    pub batch_size: usize,
    pub n_init: usize,
    pub ensemble_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub n_cem_samples: usize,
    pub n_cem_iters: usize,
    pub elite_fraction: f32,
    /// 目的ごとのサロゲートのウォームスタート weights (外側=目的, 内側=メンバー)。
    #[serde(default)]
    pub model_states: Vec<Vec<String>>,
    /// 参照点マージン: ref_k = nadir_k + ref_margin·(nadir_k − ideal_k)。default 0.1。
    #[serde(default = "default_ref_margin")]
    pub ref_margin: f32,
    /// CEM 初期標準偏差。default 0.2。
    #[serde(default = "default_sigma_init")]
    pub sigma_init: f32,
    /// CEM スタート点数 (Pareto 点 + ランダムを巡回)。default 5。
    #[serde(default = "default_n_cem_starts")]
    pub n_cem_starts: usize,
}

fn default_ref_margin() -> f32 {
    0.1
}
fn default_sigma_init() -> f32 {
    0.2
}
fn default_n_cem_starts() -> usize {
    5
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeMoOutput {
    pub candidates: Vec<Vec<f32>>,
    pub ehvi_scores: Vec<f32>,
    /// 目的ごとの次回ウォームスタート weights。
    pub model_states: Vec<Vec<String>>,
    pub pareto_size: usize,
    /// 現在の Pareto フロントの超体積 (動的参照点・最小化空間)。診断用。
    pub hypervolume: f32,
    pub mode: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeOutput {
    pub candidates: Vec<Vec<f32>>,
    pub tr_states: Vec<TrustRegionState>,
    pub acq_scores: Vec<f32>,
    pub pred_means: Vec<f32>,
    pub pred_stds: Vec<f32>,
    pub surrogate_loss: f32,
    pub mode: String,
    /// 次回 propose() に渡すウォームスタート weights (hex 文字列、メンバーごと)
    pub model_states: Vec<String>,
    /// Feasibility surrogate の次回ウォームスタート weights
    pub feas_model_states: Vec<String>,
    /// 現在の phase ("global" | "local")。Python 側が次回 config で返す。
    #[serde(default = "default_phase")]
    pub phase: String,
    /// EI 停滞カウンタ。Python 側が次回 config で返す。
    #[serde(default)]
    pub stagnation_count: usize,
}
