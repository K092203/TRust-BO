# AGENTS.md — エージェント向けオンボーディング(このファイルを読めば全体像が掴める)

TRust-BO = **Rust製トラスト領域ベイズ最適化エンジン**(PyO3でPythonへ公開、CPUのみ、GPU/BLAS不要)。
CFDなど高価な評価関数の設計最適化で、高次元(50–100D+)・ノイズあり・制約ありに強い。
GPではなく **MLPブートストラップ・アンサンブルが主サロゲート** である点が最大の特徴。

現在バージョン: **v0.3.0**(2026-07-13、PyPI公開済み。`pip install trust-bo==0.3.0`)。
v0.2.0→v0.3.0差分は CHANGELOG.md `[0.3.0]` 節が正(TandemEngine削除・phase2_early_frac
既定0.25化・ts_ei/min_cd追加等)。

## アーキテクチャ(3行)

- **Rust側は完全ステートレス**: `Engine.propose()` が全履歴+状態JSONを毎回受け取り、候補+更新状態JSONを返す。
- **Python側が状態を往復保持**: `TRustBOEngine`(python/trust_bo/engine.py)が履歴・TR状態・モデル重み(hex)・phaseを保存し、ask/tell APIを提供。
- ビルド: `maturin develop --release`(**releaseほぼ必須**、venvは`.venv`、rustupは`~/.cargo`)。

**このステートレス設計は守る価値が実証済み**: エンジンを複数インスタンス自由に合成できるため、
非同期並列(rolling_engine)もMFカスケード(multifidelity.py — HF評価70%削減を達成した機能)も
**Rustコア変更ゼロのPython合成**で実装できた(2026-07-12)。「コアに状態を持たせる」方向の
変更提案はこの配当を失う — 原則として拒否し、状態はPython層で往復させること。

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
integrations/optuna.py(sampler、acquisition="ei"固定)。
旧scipy版Phase2(tandem.py)は2026-07-12に削除済み(v0.3.0で公開済み、CHANGELOG参照)。

## 検証コマンド

```bash
source ~/.cargo/env && cargo test --release   # Rust 44テスト
.venv/bin/maturin develop --release            # ビルド+インストール(~30s)
.venv/bin/python -m pytest tests/ -q           # Python 84テスト(82+2skip、~5分、要 scipy/sklearn)
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
- bilog出力変換(`bilog_transform`フラグ、SCBO/HEBO): base比GM **0.881(12%悪化)**、
  本命のrosenbrockで**0.634(37%悪化)** — 2026-07-12に棄却。詳細 §20.2
- アンサンブルσのpost-hoc校正(`sigma_calibration`フラグ、NLL閉形式解、縮小禁止[1.0,5.0]
  クランプ): base比GM **1.008(誤差範囲)** — 2026-07-13に棄却。詳細 §22
- **決定的多様化マルチスタート(`cem_diverse_starts`フラグ、候補6)**: CEM多スタートを
  TR内top-3からFarthest-Point選択へ変更。SU2実RANS 8 seed実測でbaseline比GM
  **0.806〜0.784(19-22%悪化)**、評価効率でbaselineに勝った例0/8 — 2026-07-15に棄却。
  実CFDで探索多様性を強める変更が裏目に出るパターン(ts vs ei と同型)。詳細 §23
- **Opportunistic MADS poll(`enable_mads_poll`フラグ、候補1)**: Phase2局所停滞時の
  coordinate poll保険。単体ではSU2実RANSでbaseline比GM **0.943(ほぼ横ばい、6/8 seedで
  完全一致)** — 明確な改善なしのため2026-07-15に棄却。詳細 §23
- 決定的joint batch選択(`joint_batch_select`フラグ、候補9): アンサンブル5メンバーを
  決定的シナリオとしたjoint marginal-improvement greedy。監査で数式バグ2件検出・修正の末、
  **feasibility surrogate存在時は既存greedyへ完全fallback**という安全設計に確定
  (制約付き問題では実質無寄与)。2026-07-15に棄却。詳細 §23
- BAPI型早期終了(候補11、SU2発散予測によるwall-clock短縮): オフラインROCゲート
  (SU2実測200点)で**FPR≤2%かつTPR≥30%を同時に満たす閾値なし**(最大TPR 20.3%) —
  2026-07-15にゲート不通過でSU2 runner統合見送り。詳細 §23

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
評価回数70%削減を品質+67%で達成**(CFD系ユースケース)。単一忠実度の合成問題では
予算75はbase250のGM 0.46止まりで**70%削減は不可能と実証**(忠実度軸が唯一の経路)。
**ただしSU2実ペアは相関ゲート不通過(2026-07-12, §19)**: NeuralFoil xlarge↔SU2は
R²=0.284・ρ=0.689で文献成立条件(R²>0.75)を大きく下回り、**NeuralFoil→SU2カスケードは
現構成では非推奨**。カスケードA/Bは中止し約17時間のSU2計算を回避。

**SU2実ペアは2種のLF候補とも相関ゲート不通過、カスケード再開見送り確定
(2026-07-12/13, §19・§21)**: LF=NeuralFoil xlarge(R²=0.284・ρ=0.689)、
LF=SU2粗メッシュ(SMOKE 8点ではR²=0.859と出たが本走48点で**R²=0.447・ρ=0.423に逆転**)—
いずれも文献成立条件(R²>0.75)未達。教訓: **相関ゲートはSMOKE規模で確定させず本走
(n≥30程度)まで判定を待つ**。

**実CFDニッチ尖鋭化ジョブ(2026-07-14/15, BENCHMARK.md §23)— 全面棄却**: 「得意分野の
性能を70-300%向上」という/goalで、Codex sol調査からコア候補3件(決定的多様化
マルチスタート/MADS poll/joint batch選択)+実行速度軸(BAPI型早期終了)を選定・実装・
sol xhigh監査2ラウンド(候補9のfeasibility加重running_best数式誤りを検出→閾値式修正でも
根治せず→制約時は既存経路へ完全fallbackで確定)を経てSU2実測A/B(8 seed×4アーム×
budget100)を実施したが、**24ペア比較全てでv0.3.0 baselineに勝てなかった**
(評価効率で勝った例0件、最終品質も全アームGM<1)。早期終了もROCゲート不通過。
教訓: このエンジンは実CFDニッチで既に強い局所最適に近く、追加のアルゴリズム的介入
(特に探索性を強める方向)は総じて逆効果になりやすい。**このジョブの候補6/1/9/11も
再検証しない**。

未着手の有望案: 忠実度軸を1本に絞った代替LF定義(MF-2'': 反復数のみ削減 or
境界層解像度保持の粗メッシュ)、入力拡張MF-MLP(低相関でも安全だが優先度低下)、
SAASBO比較(WSL環境ではメモリ不足で不可)。

## 3Dフロントウィング最適化パイプライン(2026-07-16、実用検証・初の3D CFD成功例)

「本来の目的(個人のFormula Studentカー空力最適化)を3Dで検証したい」という要求に対し、
`benchmarks/su2/wing3d_mesh.py`(3Dメッシュ生成、gmsh不使用)、
`benchmarks/su2/wing3d_runner.py`(SU2 3D RANS実行ラッパー)、
`benchmarks/wing3d_benchmark.py`(65D TRustBOEngine統合・resumable本走ハーネス)を新規構築。
250評価本走でコールドスタート比+16%改善(12.62→14.63、ダウンフォース/ドラッグ比)、
AoA-CL相関-0.75・AoA-CD相関+0.988という教科書通りの物理挙動を確認。詳細は
`/home/kotaro/.claude/plans/rippling-puzzling-fiddle.md`(このジョブの一次記録)。

**次に類似コンポーネント(ディフューザー等)を作る際に再利用できる知見**:

- **FSAE実走行条件は2Dベンチマークと1桁違う**: 2D CST翼型ベンチ(Re=3e6)はFSAEフロントウィング
  の実条件(V≈40km/h, 翼弦150-250mm→Re≈1.5-3×10⁵)とは無関係。同じ車体上の別コンポーネントでも
  この速度域(Re~2×10⁵)を流用してよいが、**既存2Dベンチの設定を安易に引き継がないこと**
- **この速度域(M≈0.03)では`SOLVER=INC_RANS`(非圧縮)が正しい選択**。圧縮性`RANS`の
  低Mach極限での数値散逸・収束悪化を避けられる。地面はmovingウォール
  (`SURFACE_MOVEMENT=MOVING_WALL`+`MARKER_MOVING`、固定格子上で壁面速度のみ付与)、
  半裁モデルは`MARKER_SYM`で構築可能(SU2公式ドキュメントで裏付け済み)
  (config全文は`wing3d_runner.py`参照、そのまま流用可)
- **接地効果流はCauchy収束せず振動する**(周期約2800反復、SU2自身の1e-5基準を満たさない)。
  厳密Cauchy収束のみに頼らず、windowed-average(直近N反復平均、変動係数・前後半差で
  安定性判定)を併用すること。ただし**SU2自身が真にCauchy収束した場合はそちらを優先**し、
  windowed-averageの緩い基準で上書きしないこと(両者の使い分けロジックは
  `wing3d_runner.py::_is_converged`参照)
- **並列度は非単調**: この実行環境(20コア・RAM7.6GB)では2並列×10スレッドが
  1並列×20スレッドにも4並列×5スレッド(タイムアウトするほど非効率)にも勝った。
  「コアを細分割するほど速い」という直感は通用しない。**新しい計算環境では必ず
  実測でWORKERS数を決めること**(2/4並列だけでなく中間値も試す価値がある)
- **gmshは3Dでも依然使えない**(libGLU/sudo制約、2D O-メッシュと同じ理由で再確認)。
  「2D O-メッシュをスパン方向にB-splineで補間しながら押し出す」方式で外部依存ゼロの
  3D構造格子を作れる(camber/thickness分解でthickness係数を正値制約するだけで
  自己交差を構造的に防止できる、という設計は特に有用)
- **resumableのprotocol fingerprintは実装ファイルのハッシュまで含めないと穴が残る**:
  設定値(ITER, WORKERS, 収束閾値等)だけでなく、レンダリング済みSU2 config全文と
  関連する全実装ファイル(mesh生成・runner・harness本体)のSHA-256を含めること。
  1回作って終わりでなく、**監査で「まだ穴がある」と指摘され3回作り直した**(後述)

**運用上の教訓(最重要)**: sol xhigh監査が**3ラウンド**必要だった。1回目でClaudeが
自ら実装した収束判定に「SIGTERM+exit code 0で23反復の過渡状態を収束扱いにする」という
重大バグを検出、その修正にも2回目監査で別の欠陥(Cauchy時の平均範囲不一致等)、
3回目でようやくGO。**「自分で直したから大丈夫」という油断が最も危険**——新規実装への
修正それ自体が新しいバグの温床になりうることを、今回具体例つきで再確認した。

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
- **SU2の「収束しました」系メッセージを文字列マッチだけで信用しない**: SIGTERM等で
  中断されても`SU2_CFD`は`returncode=0`かつログに`"All convergence criteria satisfied"`と
  出すことがある(実際の収束表はCauchy[CL]/[CD]ともNo)。収束判定は必ずログ末尾の
  Convergence Field表を個別パースすること(2026-07-16、wing3d_runner.py参照)
- **SU2の並列度(WORKERS×スレッド)は実測必須、直感で決めない**: 環境によってはコアを
  細分割するほど遅くなる(20コア機で2並列×10スレッドが1×20にも4×5にも勝った実績あり)

## ドキュメント(深掘り用)

docs/ALGORITHM.md(アルゴリズム全編・設定表) / DEVELOPMENT.md(開発手順・設定表・フェーズ史) /
PERFORMANCE_ASSESSMENT.md(**正直な性能評価**: 低次元・小予算・滑らかな問題ではBoTorch/HEBOが上、
50D+/ノイズ/制約/実CFDで5–10倍速×同等以上の品質が本領) / BENCHMARK.md / ROADMAP.md /
**AGENT_WORKFLOW.md(セッションの始め方・エージェント運用規約・Codex-only運用規約・
実データに基づく損益判断 — 「どう開発を進めるか」の一次情報源。次節はそこへの入口のみ)**

## セッション開始・エージェント運用は docs/AGENT_WORKFLOW.md を参照

新しいセッションを始めるとき、およびサブエージェント/Codex CLIへの委譲を行うときは、必ず
**[docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md)** のセッション開始チェックリスト・
エージェント運用規約・Codex-only運用の優先規約に従うこと。要点だけ書くと:

1. 状態確認(`git log`/`git status`)→ 本ファイルの「棄却済み」節・「落とし穴・不変条件」節を再読
2. 標準パイプライン: 調査(サブエージェント並列)→ 吟味(sol)→ 実装(コア=自分/定型=terra)→
   監査(最終実装はsol xhigh、A/B投入前)→ 実測A/B(SMOKE→本走)→ 記録(BENCHMARK.md+本ファイル)
3. `codex exec`は必ず`timeout`ラップ+stdinパイプ+出力全量リダイレクト。委譲後の検収
   (差分レビュー→SMOKE→本走)を省略しない。コミット/タグはユーザー指示後のみ
4. Codex内蔵マルチエージェントのみで作業する場合は、AGENT_WORKFLOW.mdの
   「Codex-only運用の優先規約」(heartbeat・監査合格前のA/B禁止など)を優先適用する

詳細な手順・実データ・委譲の型・失敗事例は docs/AGENT_WORKFLOW.md 本体を参照。
