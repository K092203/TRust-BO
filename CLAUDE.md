# CLAUDE.md — エージェント向けオンボーディング(このファイルを読めば全体像が掴める)

TRust-BO = **Rust製トラスト領域ベイズ最適化エンジン**(PyO3でPythonへ公開、CPUのみ、GPU/BLAS不要)。
CFDなど高価な評価関数の設計最適化で、高次元(50–100D+)・ノイズあり・制約ありに強い。
GPではなく **MLPブートストラップ・アンサンブルが主サロゲート** である点が最大の特徴。

## アーキテクチャ(3行)

- **Rust側は完全ステートレス**: `Engine.propose()` が全履歴+状態JSONを毎回受け取り、候補+更新状態JSONを返す。
- **Python側が状態を往復保持**: `TRustBOEngine`(python/trust_bo/engine.py)が履歴・TR状態・モデル重み(hex)・phaseを保存し、ask/tell APIを提供。
- ビルド: `maturin develop --release`(**releaseほぼ必須**、venvは`.venv`、rustupは`~/.cargo`)。

## ask() 1回の処理フロー(src/lib.rs `propose()`, 全体で~800行)

1. feasible数 < `n_init` → Halton準乱数を返すだけ(cold start、candidate.rs)
2. feasible値をz-score正規化(normalize.rs)→ **MLPアンサンブル学習**(surrogate.rs):
   5メンバー × 4層MLP(64-64-32-1, ReLU) × 最大500エポックAdam、前回重みからウォームスタート、
   burn(NdArrayバックエンド)。infeasibleがあれば同型の feasibility サロゲートも学習
3. **TR更新**(tr.rs): TuRBO準拠。成功`tau_succ=3`連続で辺長×2、失敗`tau_fail=5`連続で÷2、
   `l_min=0.5^7`未満でリスタート。`n_trs>1`でTuRBO-M(空間帰属+Farthest-Point初期化)
4. **CEM候補生成**(cem.rs): TRごとにTR内best点3スタート、512サンプル×25イテレーション。
   ジョブは **rayonで並列実行**(順序・シード完全保存、lib.rs `cem_jobs`)
5. **獲得関数**(acquisition.rs): デフォルト **"ts"** = 候補ごとに `μ+z·σ, z~N(0,1)` を乱択サンプル。
   "ei"/"ucb"も選択可。制約時は非負シフト後に `× P(feasible)`
6. **バッチ選択**(batch.rs): スコア降順greedy、除外半径0.1。multi-TRは各TRに最低1スロット保証
7. (オプション `enable_phase2`) TR枯渇後、**Matern5/2残差マイクロGP**(gp.rs, f64, 自前Cholesky)
   でインカンベント近傍を精密化("tandem")

## ファイルマップ(src/ 約2900行)

| ファイル | 行数 | 内容 |
|---|---|---|
| lib.rs | ~800 | propose() 制御フロー全体 + propose_mo(2目的EHVI) + PyO3 |
| gp.rs | 292 | Phase2用マイクロGP(ハイパラはランダム探索MLE、>10Dでisotropic) |
| cem.rs | 290 | CEM本体3種(通常/GP合成/EHVI) |
| tr.rs | 265 | TR更新・TuRBO-M・リスタート |
| acquisition.rs | ~260 | ts/ei/ucb + 閉形式2D-EHVI |
| surrogate.rs | ~210 | MLPアンサンブル学習/予測(重みはhex文字列で往復) |
| types.rs | ~150 | ProposeConfig等(新機能は`#[serde(default)]`フィールドで後方互換に追加する慣習) |
| candidate/batch/normalize/pareto/hypervolume | 各<130 | 補助 |

Python側(python/trust_bo/, 計~1550行): engine.py(本体+デフォルト設定辞書)、
space.py(Float/Int/Categorical→[0,1]エンコード)、history.py(Trial保存、save/load)、
multiobjective.py(Chebyshevスカラー化、Rust不要)、rolling_engine.py(SLURM等の非同期並列評価)、
tandem.py(scipy版Phase2、旧)、integrations/optuna.py(sampler、acquisition="ei"固定)。

## 検証コマンド

```bash
source ~/.cargo/env && cargo test --release   # Rust 35テスト
.venv/bin/maturin develop --release            # ビルド+インストール(~30s)
.venv/bin/python -m pytest tests/ -q           # Python 69テスト(~6分、要 scipy/sklearn)
```
ベンチ: benchmarks/midbudget_benchmark.py が合成関数のA/B標準(SMOKE=1で1分スモーク)。
エンジンA/Bのコツ: 新機能はconfigフラグで入れ、**まずデフォルト挙動のビット一致を確認**してから比較する
(config dictは `TRustBOEngine(config={...})` → JSON → serde でそのままRustに届く)。

## 2026-07 性能改善の結果(コミット d6a02f5)— 再検討の重複を避けるため必読

採用(検証: Ackley/Rastrigin/Rosenbrock/Levy × 50/100D × ノイズ{0,5%} × 8シード, 予算250):
- **rayon並列CEM+学習テンソル巻き上げ**: ask()約2倍高速、同シードでビット一致
- **獲得関数デフォルト "ts"**: EI比リグレット幾何平均10–14%改善(27勝5敗/32)

実装したが**実測で棄却済み**(文献で有望でも本構成では効かない — 再実装前にこの数値を見よ):
- ノイズ耐性TR成功閾値: 比1.004(効果なし) / ランク重み+平滑化CEM: 1.021(悪化)
- TR×√D多様性半径: 0.999(選択結果ほぼ不変) / n_init 50→20: 1.045(悪化)
- コヒーレント単一メンバーTS: **1.375(大幅悪化)** — "ts"の乱択ばらつき自体が探索に効いている

未着手の有望案: Phase2マイクロGPへの次元スケールLogNormal長さスケール事前分布(Hvarfner 2024)、
実CFD(NeuralFoil/SU2)でのts再検証、SAASBO比較。

## 落とし穴・不変条件

- **決定性が仕様**: 同シード同入力→同出力。並列化・リファクタ時はビット一致で検証すること。
  burnの`B::seed()`は**グローバル**なのでモデル初期化を並列化してはいけない(学習ループも
  autodiffランタイムがグローバルなため並列不可 — 2026-07に試して失敗済み)
- surrogate.rsの重みhex往復(`model_states`)を壊すとwarm pathが静かに機能停止する
  (症状: warm開始後bestが一切改善しなくなる)
- TR `best_value` は**生値**(z-score前)で保持。正規化値を入れるとスケールずれで成否判定が壊れる
- EI停滞検出(Phase2遷移シグナル)は acquisition="ei" 限定。デフォルト"ts"では tr_exhausted のみが引き金
- 値は最大化方向に統一してRustへ渡る(minimizeはPython側で符号反転)。入力は[0,1]エンコード済み前提
- debugビルドはu64オーバーフローで壊れる既知問題 → 常に --release

## ドキュメント(深掘り用)

docs/ALGORITHM.md(アルゴリズム全編・設定表) / DEVELOPMENT.md(開発手順・設定表・フェーズ史) /
PERFORMANCE_ASSESSMENT.md(**正直な性能評価**: 低次元・小予算・滑らかな問題ではBoTorch/HEBOが上、
50D+/ノイズ/制約/実CFDで5–10倍速×同等以上の品質が本領) / BENCHMARK.md / ROADMAP.md
