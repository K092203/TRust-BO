# TRust-BO — Phase 3 アーキテクチャ解説

> **歴史的記録**: これは Phase 3 時点(2026-06 以前)のアーキテクチャスナップショット。
> multi-TR(TuRBO-M)、rayon 並列化、Phase 2(Tandem Residual-GP)、多目的 EHVI、
> 実 CFD パイプライン(NeuralFoil/SU2)は本書執筆後に実装済み。現在のアーキテクチャは
> `CLAUDE.md` と `docs/DEVELOPMENT.md` を参照。

## 1. 全体構成

```
trust-bo/
├── python/trust_bo/          # Python 層 (オーケストレータ)
│   ├── engine.py               # TrustBoEngine: ask/tell 公開 API
│   ├── space.py                # パラメータ定義・encode/decode
│   ├── history.py              # JSONL 履歴管理
│   └── integrations/
│       └── optuna.py           # Optuna サンプラー互換ラッパー
│
└── src/                        # Rust 層 (計算コア)
    ├── lib.rs                  # PyO3 エントリポイント: propose()
    ├── types.rs                # ProposeConfig / TrustRegionState / ProposeOutput
    ├── surrogate.rs            # Bootstrap MLP アンサンブル (学習・推論)
    ├── acquisition.rs          # UCB / EI スコアリング
    ├── cem.rs                  # Cross-Entropy Method (候補生成)
    ├── tr.rs                   # Trust Region 管理
    ├── candidate.rs            # LHS サンプリング
    ├── batch.rs                # 多様性確保バッチ選択
    └── normalize.rs            # z-score 正規化
```

---

## 2. Python 層と Rust 層の責務

### Python 層が持つもの (状態・IO)

| 責務 | ファイル |
|---|---|
| パラメータ空間の定義 (Float/Int/Categorical) | `space.py` |
| エンコード/デコード (`[0,1]` ↔ 実値) | `space.py` |
| 全評価履歴の保存 (JSONL) | `history.py` |
| TR 状態の永続化 (JSON) | `engine.py` |
| seed の派生 (SHA256) | `engine.py` |
| save/load (`.trm` = zip) | `engine.py` |
| 最小化 → 最大化の符号反転 | `engine.py` |
| Optuna サンプラー互換 | `integrations/optuna.py` |

### Rust 層が持つもの (純粋計算・ステートレス)

Python が `propose()` を呼ぶたびに **全履歴** と **TR 状態** を渡す。
Rust は状態を保持せず、毎回ゼロから計算する。

| 責務 | ファイル |
|---|---|
| LHS サンプリング (cold start) | `candidate.rs` |
| z-score 正規化 | `normalize.rs` |
| Bootstrap MLP アンサンブル学習・推論 | `surrogate.rs` |
| UCB / EI 獲得関数スコアリング | `acquisition.rs` |
| CEM 候補プール生成 (TR 境界クランプ付き) | `cem.rs` |
| TR 初期化・更新・restart | `tr.rs` |
| 多様性確保グリーディバッチ選択 | `batch.rs` |

> **ステートレス設計の理由**: Python が全履歴を毎回渡すことで、
> 再現性が seed のみで完全保証できる。Rust にキャッシュがない。

---

## 3. ask/tell の流れ

```
ユーザー
  │
  ▼ engine.ask(batch_size=8)
Python: TrustBoEngine.ask()
  ├─ history.complete_trials() → 全完了トライアル取得
  ├─ space.encode(params)       → [0,1] 正規化
  ├─ _to_maximize(values)       → 最小化なら符号反転 (-v)
  ├─ _derive_seed(master, ask_count) → SHA256 で seed 派生
  └─ rust.propose(params, values, feasibility, [], tr_states_json, config_json, seed)
        │
        ▼ Rust: Engine::propose()
        ├─ n_complete < n_init? → LHS cold start → return
        ├─ feasible 点のみフィルタ
        ├─ zscore(feasible_values)
        ├─ Ensemble::train (bootstrap MLP × 5)
        ├─ TR 状態 init or update
        ├─ TR 境界内 top-3 → multi-start CEM (TR-CEM)
        ├─ pool を EI でスコアリング
        ├─ greedy_select → batch 候補
        └─ return ProposeOutput (candidates, tr_states, ...)
  ├─ tr_states を Python が保存 (次回渡す)
  └─ space.decode(candidates) → 実値パラメータ dict
  │
  ▼ 候補 list[dict] をユーザーに返す

ユーザー: CFD/FEM 評価 → results
  │
  ▼ engine.tell(candidates, results)
Python: TrustBoEngine.tell()
  └─ history.add(Trial) → JSONL に追記
```

---

## 4. Engine.propose() の入力と出力

### 入力 (Python → Rust)

```python
rust.propose(
    params,             # list[list[float]]  [0,1] 正規化済み, 全評価済み点
    values,             # list[float]        最大化方向生値 (minimize なら -v)
    feasibility,        # list[bool]         True=実行可能 (Phase 3 まで全 True)
    [],                 # constraint_values  Phase 4 まで空
    tr_states_json,     # str                JSON: list[TrustRegionState]
    config_json,        # str                JSON: ProposeConfig
    seed,               # int                u64: 決定論的 seed
)
```

### 出力 (Rust → Python)

```python
{
    "candidates": [[0.3, 0.7, ...], ...],   # [batch_size, n_dims] [0,1] 正規化
    "tr_states":  [{"center": [...], ...}], # TR 状態 (Python が次回まで保持)
    "acq_scores": [1.23, ...],              # 候補の EI/UCB スコア (診断用)
    "pred_means": [0.8, ...],               # アンサンブル平均予測 (診断用)
    "pred_stds":  [0.2, ...],               # アンサンブル標準偏差 (不確実性)
    "surrogate_loss": 0.0,                  # Phase 3 現在: 未計算 (TODO)
    "mode": "tr_cem"                        # "cold_start" | "tr_cem"
}
```

> `tr_states` は Python 側で `self._tr_states` に保存され、次の `ask()` で渡される。
> この往復によって TR 状態が継続する。

---

## 5. TrustRegionState の意味

```rust
pub struct TrustRegionState {
    pub center: Vec<f32>,       // TR 中心 ([0,1] 正規化空間)
    pub side_length: f32,       // TR の辺長 L (正方形の一辺)
    pub success_count: usize,   // 連続改善バッチ数
    pub failure_count: usize,   // 連続未改善バッチ数
    pub best_value: f32,        // 参照値 (raw 最大化方向値, z-score 前)
    pub active: bool,           // false = L < L_min → restart 待ち
}
```

**best_value を z-score で保存しない理由**:
新しい観測が追加されると z-score の平均・標準偏差が変化する。
生値で保存しておき、propose() 内で毎回同じスケールで比較する。

**center の意味**:
全履歴中の最良観測点の encoded 座標。改善が見つかるたびに移動する。

---

## 6. TR がいつ広がる / 縮む / restart するか

```
propose() 呼び出し k 時点の判定:
  global_best = max(全履歴の feasible values)

  if global_best > TR.best_value:     ← 前回 propose() より改善あり
      success_count += 1
      failure_count = 0
      TR.center  ← global_best の encoded 座標に移動
      TR.best_value ← global_best
  else:                               ← 改善なし
      failure_count += 1
      success_count = 0

  if success_count >= tau_succ (=3):
      side_length = min(side_length × 2, L_max=1.0)  ← 拡大
      カウンタリセット

  if failure_count >= tau_fail (=10):
      side_length /= 2                               ← 縮小
      カウンタリセット

  if side_length < L_min (=0.5^7 ≈ 0.0078):
      active = false  → restart_tr() で新しいランダム中心
```

**図でイメージ**:
```
L=1.0 (初期, フル空間)
  ↓ 10バッチ改善なし
L=0.5
  ↓ 3バッチ連続改善
L=1.0 (拡大)
  ↓ 10バッチ改善なし × 7回
L=0.0078 → restart (新ランダム中心)
```

> Phase 3 は TR が1つ。Phase 4+ で multi-TR (並列TR) を検討。

---

## 7. CEM と TR-CEM の候補生成の流れ

### Phase 3 の TR-CEM (実装: `src/cem.rs`)

```
入力: init_mu (スタート点), bounds_lo/hi (TR 境界), sigma_init = L/6

for iter in 0..n_cem_iters (=10):
  1. N(mu, sigma) から n_cem_samples (=512) 点をサンプリング
     → Box-Muller 変換 → bounds にクランプ
  2. アンサンブルで (mean, std) を予測
  3. EI スコアを計算 (best_norm を閾値として改善確率を評価)
  4. 上位 elite_k (= 512 × 0.1 = 51) 点を選択
  5. mu, sigma を elites から更新 (MLE)
  6. max(sigma) < halt_eps → 早期停止

return 最終 elite pool (51点)
```

### マルチスタートの仕組み (実装: `src/lib.rs`)

```
TR 境界内の観測点を値の高い順にソート → 上位 3 点をスタートに使う
(TR 内に 3 点未満の場合は TR center のみ)

pool = []
for k in 0..3:
    elites_k = cem_pool(start_mu=top_k[k], ...)
    pool.extend(elites_k)

pool サイズ: 最大 3 × 51 = 153 点
```

**マルチスタートの意図**: 1点スタートだと surrogate の局所最大に CEM が収束し、
本当に良い領域を見逃すリスクがある。3点から並列スタートすることで
surrogate が一部の領域を誤推定していても他のスタートが救済する。

### sigma_init = L/6 の根拠

```
N(mu, sigma) の 99.7% が mu ± 3*sigma に収まる。
sigma = L/6 → 3*sigma = L/2 → TR 辺長の半径内に 99.7% のサンプルが収まる。
```

---

## 8. Bootstrap Ensemble が何を解決しているか

### 問題: 全メンバーが同じデータで学習すると多様性がない

MLP アンサンブルの不確実性推定は「メンバー間の予測の分散」に依存する。
しかし同じデータ × 異なる乱数初期化だけでは、
過パラメータなネットワーク (n_params ≈ 7000 > n_data ≈ 22) が
全メンバー同じ局所解に収束 → std ≈ 0 → EI が機能しない。

### 解決: Bootstrap サンプリング

```rust
// surrogate.rs: Ensemble::train() 内
for i in 0..n_members {
    // 1. burn の乱数シードをメンバーごとに設定 (重み初期化に影響)
    B::seed(seed + i * 0x9e3779b97f4a7c15);

    // 2. n 個を復元抽出 (平均 ~15 ユニーク点 / n=24 の場合)
    let boot_idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
    // boot_idx 例: [3, 3, 7, 0, 15, 3, ...] (重複あり、一部欠落)

    // 3. メンバー i は boot_idx に対応する点のみで学習
    let xs_boot = boot_idx.iter().flat_map(|&j| params[j].iter()).collect();
    let ys_boot = boot_idx.iter().map(|&j| norm_values[j]).collect();
}
```

### 結果

メンバーそれぞれが「少し違うデータ」で学習 → 未観測領域での予測が異なる
→ std が正しく "知らない場所 = 不確かさが高い" を表現できる。

```
メンバー 1 (欠落点あり): 領域 A を高評価
メンバー 2 (別の欠落点): 領域 B を高評価
→ 平均 = 中程度、std = 高い → EI でちょうどよい exploration
```

### UCB から EI に切り替えた理由

UCB (β=2): score = mean + 2 × std
  → std が高い未知領域が常に勝つ → CEM がランダムな不確実領域を追う
  → warm path 全評価が bad region に集中 → LHS より悪化

EI: score = E[max(f(x) - f*, 0)]
  → best_norm より改善できる確率 × 改善量 → 探索と活用が自然にバランス
  → CEM が「今より良くなりそうな場所」に収束

---

## 9. 現在の Phase 1/2/3 テストが何を保証しているか

### Phase 1 (`tests/test_phase1.py`, 4 tests)

| テスト | 保証内容 |
|---|---|
| `test_sphere_5d` | cold start LHS のみで sphere 5D が val < 15 (上限チェック) |
| `test_ackley_5d` | cold start LHS のみで ackley 5D が val < 10 |
| `test_reproducibility_across_sessions` | 同一 seed → 完全一致、異なる seed → 異なる結果 |
| `test_optuna_adapter` | Optuna 経由で 3D sphere が val < 5 (統合テスト) |

**保証しないこと**: surrogate がよいこと、warm path が機能すること

### Phase 2 (`tests/test_phase2.py`, 5 tests)

設定: BUDGET=120, BATCH=8, epochs=300, ensemble_size=5, acquisition="ei"

| テスト | 保証内容 |
|---|---|
| `test_sphere_10d_beats_random` | 3 seed 平均で TRM < Random (10D sphere) |
| `test_ackley_10d_beats_random` | 3 seed 平均で TRM < Random (10D ackley) |
| `test_reproducibility_phase2` | warm path 込みで 5D sphere が完全再現 |
| `test_tpe_comparison` | TRM が Optuna TPE の 2 倍以内 (同等水準) |
| `test_generate_benchmark_report` | CSV レポート生成 + 再現性 (tolerance 5e-4) |

**保証しないこと**: 50D での動作、TR の収縮/拡大挙動

### Phase 3 (`tests/test_phase3.py`, 4 tests)

設定: BUDGET=200, BATCH=10, epochs=200, 50D

| テスト | 保証内容 |
|---|---|
| `test_ackley_50d_beats_random` | 3 seed 平均で TRM < Random (50D ackley) |
| `test_rosenbrock_50d_beats_random` | 3 seed 平均で TRM < Random (50D rosenbrock) |
| `test_tr_shrinks_on_no_improvement` | warm path が動作し ackley < 25 を達成できる |
| `test_reproducibility_phase3` | TR warm path 込みで 10D ackley が完全再現 |

**保証しないこと**: TR の success/failure カウント内訳、TURBO/CMA-ES との比較

---

## 10. 50D 結果の読み方

### 現在の数値 (BUDGET=200, BATCH=10, 3 seeds)

```
50D Ackley:
  seed=42: TRM=8.729, Random=8.940  (差: +0.211)
  seed= 7: TRM=7.777, Random=8.875  (差: +1.097)
  seed=13: TRM=7.953, Random=9.167  (差: +1.214)
  平均: TRM=8.15, Random=9.0  (約 9% 改善)

50D Rosenbrock:
  seed=42: TRM=187268, Random=279719  (差: +92451, 33% 改善)
  seed= 7: TRM=140550, Random=300684  (差: +160134, 53% 改善)
  seed=13: TRM=198777, Random=396065  (差: +197288, 50% 改善)
  平均: TRM=175.5k, Random=325.5k  (約 46% 改善)
```

### 解釈

- **Ackley 50D**: 改善率 9% は小さく見えるが、Ackley は多峰性が強く 50D では
  LHS 200 点が既にかなり良い基準値。TR が局所探索で安定的に改善している。
  参考: Ackley のグローバル最適値 = 0。Random で 8-9 は典型的な結果。

- **Rosenbrock 50D**: 改善率 33-53% は顕著。Rosenbrock は「谷に沿って進む」
  必要がある問題で、TR の局所集中が効く。
  参考: Rosenbrock のグローバル最適値 = 0 (x_i=1 で達成)。

- **注意事項**:
  - 3 seeds のみで分散は不明
  - TURBO/CMA-ES との定量比較はまだない
  - 収束曲線 (best-so-far curve) を見ていない

---

## 11. Phase 4 に進む前に理解・確認すべきポイント

### アーキテクチャ上の未確認事項

1. **surrogate_loss が常に 0.0**
   `lib.rs` の `ProposeOutput` で `surrogate_loss: 0.0` をハードコード中。
   実際の学習損失を計測・記録していないため、サロゲートが収束しているか
   判断できない。`Ensemble::train()` が最終 loss を返すよう修正が必要。

2. **TR success/failure の計測がない**
   TR が実際に縮小/拡大しているかをテストで確認できていない。
   `tr_states` を Python 側で観察するユーティリティが必要。

3. **cold start が LHS のみ (Sobol 未実装)**
   `TRM_Engine_Dev.md` では「初期設計は Sobol 列」と記載されているが、
   現在は LHS。Sobol は低食い違い列として LHS より空間被覆が均一。
   50D では差が出やすい。

4. **rayon 並列学習が未実装**
   `surrogate.rs` はメンバーを逐次学習している。
   `Cargo.toml` に rayon は依存関係として宣言済みだが使用していない。
   50D で epochs=500 にすると学習が律速になる。

5. **infeasible 点の扱いが未テスト**
   `feasibility=False` の点は propose() でフィルタ済みだが、
   「全点が infeasible だった場合」のフォールバックパスが正しいか
   確認が必要。

### 数値的安定性

6. **zscore が n=1 のとき std=1e-8 に固定**
   `normalize.rs` で `std.max(1e-8)` としている。
   warm path 開始直後に全点が同一値の場合に問題ないか確認が必要。

7. **EI の std=0 時の安全処理**
   `acquisition.rs` の EI で `s=0` の場合 `z` が NaN になりうる。
   現在 `stds` は `max(1e-8)` でクランプしているが、
   `(m - f_best) / s` の計算が依然として問題になりうる。

### テスト品質

8. **seed=3 点では分散が大きい**
   Phase 2/3 のテストはいずれも 3 seeds で平均を取っている。
   1 つの外れ値 seed で平均が逆転するリスクがある。
   Phase 3.5 では 10 seeds で median/IQR を使うべき。

9. **best-so-far curve の欠如**
   現在のテストは「最終値が random より小さいか」だけを確認している。
   「何 eval 目から TRM が random を上回り始めるか」が見えない。
   収束速度の評価に必要。

10. **TURBO/CMA-ES との比較なし**
    Phase 3 exit criteria に「TURBO・CMA-ES に competitive」とあるが、
    現在のテストはそれを保証していない。Phase 3.5 で導入を検討。
