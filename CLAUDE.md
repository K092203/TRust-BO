# CLAUDE.md — エージェント向けオンボーディング(このファイルを読めば全体像が掴める)

TRust-BO = **Rust製トラスト領域ベイズ最適化エンジン**(PyO3でPythonへ公開、CPUのみ、GPU/BLAS不要)。
CFDなど高価な評価関数の設計最適化で、高次元(50–100D+)・ノイズあり・制約ありに強い。
GPではなく **MLPブートストラップ・アンサンブルが主サロゲート** である点が最大の特徴。

現在バージョン: **v0.2.0**(2026-07-11、PyPI公開済み。`pip install trust-bo==0.2.0`)。
v0.1.0→v0.2.0差分は CHANGELOG.md `[0.2.0]` 節が正。

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

## ファイルマップ(src/ 約2700行)

| ファイル | 行数 | 内容 |
|---|---|---|
| lib.rs | ~800 | propose() 制御フロー全体 + propose_mo(2目的EHVI) + PyO3 |
| gp.rs | 374 | Phase2用マイクロGP(ハイパラ探索はMLE、`phase2_ls_prior`でMAP切替可・デフォルトoff。>10Dでisotropic) |
| cem.rs | 290 | CEM本体3種(通常/GP合成/EHVI) |
| tr.rs | 265 | TR更新・TuRBO-M・リスタート |
| acquisition.rs | ~260 | ts/ei/ucb + 閉形式2D-EHVI |
| surrogate.rs | ~210 | MLPアンサンブル学習/予測(重みはhex文字列で往復) |
| types.rs | ~150 | ProposeConfig等(新機能は`#[serde(default)]`フィールドで後方互換に追加する慣習) |
| candidate/batch/normalize/pareto/hypervolume | 各<130 | 補助 |

Python側(python/trust_bo/, 計~1700行): engine.py(本体+デフォルト設定辞書)、
space.py(Float/Int/Categorical→[0,1]エンコード)、history.py(Trial保存、save/load)、
multiobjective.py(Chebyshevスカラー化、Rust不要)、rolling_engine.py(SLURM等の非同期並列評価)、
multifidelity.py(CascadeMFEngine: LF→HF 2段カスケード、Float空間限定、Rustコア不変)、
tandem.py(scipy版Phase2、旧)、integrations/optuna.py(sampler、acquisition="ei"固定)。

## 検証コマンド

```bash
source ~/.cargo/env && cargo test --release   # Rust 39テスト
.venv/bin/maturin develop --release            # ビルド+インストール(~30s)
.venv/bin/python -m pytest tests/ -q           # Python 71テスト(~6分、要 scipy/sklearn)
```
ベンチ: benchmarks/midbudget_benchmark.py が合成関数のA/B標準(SMOKE=1で1分スモーク)。
エンジンA/Bのコツ: 新機能はconfigフラグで入れ、**まずデフォルト挙動のビット一致を確認**してから比較する
(config dictは `TRustBOEngine(config={...})` → JSON → serde でそのままRustに届く)。

## 2026-07 性能改善の結果(コミット d6a02f5)— 再検討の重複を避けるため必読

採用(検証: Ackley/Rastrigin/Rosenbrock/Levy × 50/100D × ノイズ{0,5%} × 8シード, 予算250。
詳細データは BENCHMARK.md §14):
- **rayon並列CEM+学習テンソル巻き上げ**: ask()約2倍高速、同シードでビット一致
- **獲得関数デフォルト "ts"**: EI比リグレット幾何平均10–14%改善(27勝5敗/32)。
  ただし合成多峰関数限定の結果 — 下記「実CFDでのts再検証」参照

実装したが**実測で棄却済み**(文献で有望でも本構成では効かない — 再実装前にこの数値を見よ):
- ノイズ耐性TR成功閾値: 比1.004(効果なし) / ランク重み+平滑化CEM: 1.021(悪化)
- TR×√D多様性半径: 0.999(選択結果ほぼ不変) / n_init 50→20: 1.045(悪化)
- コヒーレント単一メンバーTS: **1.375(大幅悪化)** — "ts"の乱択ばらつき自体が探索に効いている
- Phase2 GPのLogNormal長さスケール事前(Hvarfner 2024, `phase2_ls_prior`フラグとして実装済み・
  デフォルトoff): 比0.996(効果なし、8勝/128、Phase2発火45/128) — 2026-07-11に棄却。
  局所残差GPには論文前提(グローバル高次元GP)が当てはまらない。詳細 BENCHMARK.md §15.1
- dual TR(`n_trs=2`, 中予算250): tr1/tr2比**0.841(16%悪化)**、39勝/128 — 2026-07-12に棄却。
  履歴分割の害が支配的。多TRは大予算専用。詳細 §17
- RAASP型次元マスク(`cem_dim_mask`フラグ、Papenmeier ICML2025のCEM適応): 同予算ei比
  0.971(58勝/128、利得なし) — 2026-07-12に棄却。TRが既に局所性を担保する構成では
  次元マスクの出番がない。初版のσ人工収縮欠陥はsol監査で検出→修正済みの上での判定。詳細 §18.3

**実CFDでのts再検証(2026-07-11)**: NeuralFoil CST 16D(Cl/Cd)では**EIがTSに明確勝利**
(ts/ei幾何平均 0.916/ノイズ5%で0.862、ts 2勝6敗)。SU2実RANS(H-2同条件、3シード)でも
**ts/ei比0.353・ts 0勝3敗**で確定。合成多峰関数と逆転 — TSの探索ボーナスは多峰性問題限定。
デフォルトは"ts"のままだが、**実CFD系では acquisition="ei" 推奨**。詳細 BENCHMARK.md §15–16

**獲得関数ミックス+Phase2早期発火(2026-07-11, BENCHMARK.md §16)**:
- `acquisition="ts_ei"`(バッチ前半EI+残りTS、単一TR限定): 万能デフォルトの資格なしだが
  **ノイズあり(5%)CFD様問題では全アーム最良**(NeuralFoilでei比+9%)。ニッチ用途フラグとして残置
- `phase2_early_frac`(TR辺長≤l_init×fracでPhase2遷移許可): **enable_phase2構成の明確な改善**
  (0.25でts比GM 1.372全体/1.129 rosenbrock除外、84勝/128)。デフォルト"ts"はEI停滞シグナルを
  持たずPhase2発火が45/128に留まるのが弱点で、これを補う。デフォルト0.0のまま、
  **enable_phase2使用時は0.25推奨**。教訓: Phase2有効構成では「localに入れること」が
  獲得関数の差より支配的

**マルチフィデリティ・カスケード(2026-07-12, BENCHMARK.md §18)**: `CascadeMFEngine`
(python/trust_bo/multifidelity.py, Rustコア不変)を実装・採用。NeuralFoil CST 16D
(LF=xsmall/HF=xxxlarge)で **HF30評価がHF直接100評価をGM 1.671・8/8勝で上回る =
評価回数70%削減を品質+67%で達成**(CFD系ユースケース)。ただしLF/HF相関が高い理想ペアでの
結果 — SU2実ペア検証は未実施。単一忠実度の合成問題では予算75はbase250のGM 0.46止まりで
**70%削減は不可能と実証**(忠実度軸が唯一の経路)。

未着手の有望案: SU2実ペアでのカスケード検証、入力拡張MF-MLP(LF予測をサロゲート特徴へ)、
bilog出力変換、アンサンブルσ校正、phase2_early_fracの実CFD検証+v0.3デフォルト化判断、
SAASBO比較(WSL環境ではメモリ不足で不可)。

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
- **`v*` タグをpushすると`.github/workflows/release.yml`が自動発火**し、PyPI公開+GitHub Release
  作成まで人手を介さず進む(`pypi`環境にreviewer承認ゲートなし)。バージョンを上げる作業自体
  (pyproject.toml/Cargo.toml/CHANGELOG.md)はローカルで完結する安全な変更だが、
  タグの作成・pushは必ずユーザーの明示的な確認を経てから行うこと

## ドキュメント(深掘り用)

docs/ALGORITHM.md(アルゴリズム全編・設定表) / DEVELOPMENT.md(開発手順・設定表・フェーズ史) /
PERFORMANCE_ASSESSMENT.md(**正直な性能評価**: 低次元・小予算・滑らかな問題ではBoTorch/HEBOが上、
50D+/ノイズ/制約/実CFDで5–10倍速×同等以上の品質が本領) / BENCHMARK.md / ROADMAP.md /
AI_OPERATIONS.md(マルチエージェント運用の実データ・役割設計の理由・損益の経験則。
下記規約の根拠データはここ)

## エージェント運用規約(別プロジェクトのAGENTS.mdから抽出し、2026-07の開発で実運用した方針)

- **応答は日本語**(コード・コマンド・API名のみ英語可)
- **役割分担**: Claude Code = 主実装者・統合判断役。Codex CLI = セカンドオピニオン/検証台
  (レビュー・正誤確認・性能分析)。定型実装の委譲は可だが、成果物は必ず
  「差分レビュー + ベンチ実測」でClaude側が検証する
- **Codexの使い分け**: 定型作業 = `codex exec -s workspace-write -m gpt-5.6-terra
  -c model_reasoning_effort="high"`。数学・数値の精密レビュー等の難所 = `-m gpt-5.6-sol`。
  最終リリース前監査・最難関 = `-m gpt-5.6-sol -c model_reasoning_effort="xhigh"`(sol ultra)。
  レビュー用途は read-only(`-s` なし)で呼ぶ
- **codex exec は必ず `timeout <秒>` でラップする**: terraが1.5時間無出力でハングした実績あり
  (2026-07-11)。25分(1500s)程度のハードキャップ+出力ファイルの早期確認。ハング時はkillして
  自前実装に切替(スクリプト系なら大抵その方が速い)
- **ハルシネーション対策は多重チェックで担保**: 「実装 ⇔ 独立レビュー ⇔ 実測A/B」の三重化。
  実績: solレビューが実バグ2件(制約付きTSの符号誤り、Box-Muller偏り)を検出、
  ビット一致検証がterraの実装事故(warm path破壊)を検出、sol ultra監査がRAASP初版の
  σ人工収縮(√p倍/反復)を検出(2026-07-12 — A/B実行中に発覚し測定やり直しで済んだ)
- **欠陥版で走ったA/B結果は破棄して再測定**: 監査・レビューで実装欠陥が見つかったら、
  その版で収集済みのA/B行をCSVから削除し(bench_resumeが再実行してくれる)、修正版で
  取り直してから採否判定する。欠陥版の数値で棄却/採用を決めない
- **文献調査サブエージェントには棄却済みリストを必ず渡す**: 渡し忘れたエージェントが
  棄却済みのHvarfner事前を再提案した実例あり(2026-07-12)。本ファイルの「棄却済み」節を
  プロンプトに貼ること。逆に、調査エージェントの役割は明確に分割する
  (MF-BO/高次元/サロゲートの3分割は機能した)
- **ドキュメントは一元管理**: 同じ内容を複数mdに重複記載しない(陳腐化・矛盾の温床)。
  エージェント向け要約は本ファイルのみ、詳細は docs/ 各編へリンクで委ねる
- **結果は数値で正直に報告**: 改善しなかった案・失敗した実装も数値とともに記録する
  (上の「棄却済み」節がその実践)
