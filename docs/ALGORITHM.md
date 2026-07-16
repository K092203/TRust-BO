# TRust-BO アルゴリズム詳細解説

TRust-BO の内部アルゴリズムを、コードの構造に沿って詳細に解説する。
対象コード: `src/*.rs`（Rust コア、約 2,700 行）+ `python/trust_bo/*.py`（Python ラッパー）。

---

## 目次

1. [全体アーキテクチャ](#1-全体アーキテクチャ)
2. [Python 層 — ステートの所有者](#2-python-層--ステートの所有者)
3. [探索空間の正規化（space.py）](#3-探索空間の正規化spacepy)
4. [Rust コア `propose()` のパイプライン全体像](#4-rust-コア-propose-のパイプライン全体像)
5. [Cold Start — Halton 準乱数（candidate.rs）](#5-cold-start--halton-準乱数candidaters)
6. [値の正規化（normalize.rs）](#6-値の正規化normalizers)
7. [サロゲートモデル — MLP Bootstrap Ensemble（surrogate.rs）](#7-サロゲートモデル--mlp-bootstrap-ensemblesurrogaters)
8. [Trust Region 動力学（tr.rs）](#8-trust-region-動力学trrs)
9. [CEM — 候補生成（cem.rs）](#9-cem--候補生成cemrs)
10. [獲得関数（acquisition.rs）](#10-獲得関数acquisitionrs)
11. [バッチ選択（batch.rs）](#11-バッチ選択batchrs)
12. [Phase 2 — Tandem Residual Micro-GP（gp.rs + lib.rs）](#12-phase-2--tandem-residual-micro-gpgprs--librs)
13. [制約処理 — Feasibility Surrogate](#13-制約処理--feasibility-surrogate)
14. [TuRBO-M（Multi Trust Region）](#14-turbo-mmulti-trust-region)
15. [設計思想の総括](#15-設計思想の総括)
16. [既知の弱点と限界](#16-既知の弱点と限界)

---

## 1. 全体アーキテクチャ

### 1.1 二層構造

```
┌─────────────────────────────────────────────┐
│ Python 層 (trust_bo パッケージ)               │
│  - TRustBOEngine: ask/tell API、履歴管理       │
│  - SearchSpaceManager: [0,1]^d ⇔ 実空間変換   │
│  - HistoryStore: 全試行の永続記録              │
│  - save/load (zip)                           │
└──────────────┬──────────────────────────────┘
               │ PyO3 (JSON 文字列で往復)
┌──────────────▼──────────────────────────────┐
│ Rust コア (_lib.so)                          │
│  Engine.propose() — 1 関数のみ公開            │
│  - サロゲート訓練 (burn/ndarray)              │
│  - TR 管理 / CEM / 獲得関数 / バッチ選択       │
│  - Phase 2 Micro-GP                          │
└─────────────────────────────────────────────┘
```

### 1.2 最重要の設計判断: Rust コアは完全ステートレス

`Engine` 構造体はフィールドを持たない（`pub struct Engine {}`）。
全状態（評価履歴・TR 状態・サロゲート重み・phase）は **Python 側が所有**し、
`propose()` 呼び出しごとに JSON で渡して JSON で受け取る。

**なぜこの設計か:**

1. **save/load が自明になる** — 状態は Python の dict 群なので、zip に JSON を
   書くだけで完全なレジューム（`engine.save("study.zip")` → `load()`）が成立する。
   Rust 側に隠れた状態があると、シリアライズ漏れが必ずバグになる。
2. **再現性** — 同じ入力（履歴 + seed）に対して `propose()` は純粋関数として
   同じ出力を返す。デバッグ時に状態の食い違いを疑う必要がない。
3. **プロセス再起動耐性** — CFD ワークフローでは 1 評価が数時間かかるため、
   途中でマシンが落ちても zip から再開できることが実用上の生命線。

**コスト:** サロゲート重み（後述の hex 文字列、約 1MB/round）を毎回往復させる
オーバーヘッド。これは既知の非効率として README にも明記されている。

### 1.3 ask/tell プロトコル

```python
candidates = engine.ask(batch_size=4)   # 次に評価すべき 4 点を提案
results = [{"value": f(c), "feasible": True} for c in candidates]
engine.tell(candidates, results)        # 結果を履歴に追加
```

`ask` が Rust の `propose()` を呼び、`tell` は Python 側で履歴に追記するだけ
（Rust は呼ばない）。オプティマイザの全計算は `ask` 時に行われる。
これは「評価器（CFD ソルバー）が主、オプティマイザが従」という CFD ワークフローの
制御反転に合わせた設計で、ソルバー側のスケジューラ（ジョブキュー等）に主導権を渡せる。

---

## 2. Python 層 — ステートの所有者

`python/trust_bo/engine.py`

### 2.1 方向の統一（minimize → maximize）

Rust コアは**内部的に常に最大化**で動く。Python 層が入口で符号反転する:

```python
def _to_maximize(self, objective_values):
    v = objective_values[0]
    return -v if self._direction == "minimize" else v
```

これにより Rust 側は「大きいほど良い」という単一の前提で書け、
TR の成否判定（`global_best_val > s.best_value`）等に分岐が不要になる。

### 2.2 シード導出 — SHA-256 によるストリーム分割

```python
def _derive_seed(master, count):
    data = struct.pack("<QQ", master, count)
    return struct.unpack("<Q", hashlib.sha256(data).digest()[:8])[0]
```

`ask()` ごとに `(master_seed, ask_count)` をハッシュして 64bit シードを作る。

**なぜ単純な `seed + count` ではないのか:** 連番シードは StdRng 等でストリーム間
相関を生むことがある。暗号ハッシュを使えば呼び出し回ごとに統計的に独立な
シードが得られ、しかも決定論的（再現可能）。再現性とランダム品質の両立。

### 2.3 infeasible 試行の値埋め

infeasible（制約違反）試行は feasibility surrogate の訓練には使うが、
目的値が無意味なことがある。その際 `0.0` 固定で埋めると feasible 値のスケールに
依存したバイアスが生じるため、**feasible 試行の最悪値**で埋める:

```python
_worst_feasible = min(_feasible_vals) if _feasible_vals else 0.0
values = [... if t.status == "complete" else _worst_feasible ...]
```

（Rust 側は主サロゲート訓練で infeasible をフィルタするので実害はないが、
「意味的に正しい値を渡す」防御的設計。）

### 2.4 適応的デフォルト

```python
self._config.setdefault("n_init", max(10, min(2 * (n + 1), 50)))
adaptive_l = min(0.8, max(0.3, (5.0 / n_init_val) ** (1.0 / max(n, 1))))
self._config.setdefault("l_init", adaptive_l)
```

- **n_init = max(10, min(2(d+1), 50))**: 次元に比例して初期サンプルを増やすが
  50 で頭打ち。50D 以上では n_init=50。
- **adaptive l_init**: 「初期 TR の中に訓練点が約 5 点入る」ように辺長を逆算する。
  一様分布なら TR 内の点数期待値は `n_init × l^d` なので `l = (5/n_init)^(1/d)`。
  d が大きいほど l→1 に近づく（高次元では体積が急減するため広い TR が必要）。
  [0.3, 0.8] にクランプ。

**設計意図:** TuRBO 論文の固定 `l_init=0.8` は低次元で広すぎる。TR 内に
訓練点がないと CEM のスタート点が center のみになり初動が弱いため、
「TR 内 5 点」を不変条件としてスケールさせる。

### 2.5 デフォルト設定一覧

| パラメータ | 値 | 意味 |
|---|---|---|
| `ensemble_size` | 5 | MLP アンサンブルのメンバー数 |
| `epochs` | 500 | 各メンバーの最大学習エポック |
| `learning_rate` | 5e-4 | Adam 学習率 |
| `n_cem_samples` | 512 | CEM の 1 イテレーションあたりサンプル数 |
| `n_cem_iters` | 25 | CEM 最大イテレーション数 |
| `elite_fraction` | 0.1 | CEM エリート比率（512×0.1≈51 点） |
| `acquisition` | "ts" | 獲得関数（TS 風乱択 / EI / UCB）。デフォルト "ts" は候補ごとに N(μ,σ²) からスコアを乱択サンプリングする |
| `beta` | 2.0 | UCB の探索係数（EI 時は未使用） |
| `tau_succ` | 3 | 連続成功 → TR 拡大の閾値 |
| `tau_fail` | 5 | 連続失敗 → TR 縮小の閾値 |
| `l_max` | 1.0 | TR 最大辺長 |
| `l_min` | 0.0078125 (=0.5⁷) | TR 最小辺長（下回ると restart） |
| `n_trs` | 1 | 並列 TR 数（TuRBO-M） |

---

## 3. 探索空間の正規化（space.py）

全パラメータを **[0,1]^d の単位超立方体**に写像してから Rust に渡す。
Rust 側は実空間の存在を知らない。

| 型 | encode | decode |
|---|---|---|
| `Float` | 線形 or 対数で [0,1] へ | 逆変換 + step 丸め + クランプ |
| `Int` | 線形 or 対数で [0,1] へ | round + step 整列 + クランプ |
| `Categorical` | `index/(n-1)` | `round(nv×(n-1))` で choice 復元 |

**なぜ単位立方体か:**
1. TR の辺長 `side_length` が全次元で同じ意味を持つ（異方性は ARD でなく
   サロゲートに任せる）
2. CEM の σ 初期値 `L/6` が次元によらず妥当になる
3. MLP の入力スケールが安定し学習率を固定できる

**Categorical の扱いの限界:** カテゴリを 1 次元連続値に埋め込むため、
choices 間に擬似的な順序が生まれる。one-hot ではないのは次元爆発を避けるため
だが、カテゴリ数が多い場合は順序バイアスが性能を落とす可能性がある（既知の簡略化）。

---

## 4. Rust コア `propose()` のパイプライン全体像

`src/lib.rs` の `propose()` は基本フローとして以下の段階を順に通る(2026-06時点の
初版は約500行だったが、2026-07にearly_frac/MADS poll/diverse starts/joint batch選択等の
デフォルトoff実験フラグが追加され`src/lib.rs`全体は1100行超に増加。以下はデフォルト構成の
骨格のみを示す):

```
入力: params [n×d], values [n], feasibility [n], TR状態, config, seed
  │
  ├─ (A) feasible 件数 < n_init ？ ──YES→ Halton cold start で即 return
  │
  ├─ (B) z-score 正規化、global best 特定
  ├─ (C) MLP Bootstrap Ensemble 訓練（ウォームスタート）
  ├─ (D) infeasible があれば feasibility surrogate も訓練
  ├─ (E) TR 状態更新（成功/失敗カウント → 拡大/縮小/restart）
  │
  ├─ (F) Phase 2 判定:
  │       enable_phase2 && 単一TR && 評価数 ≥ 3×n_init
  │       && (sticky_local || TR枯渇 || EI停滞≥5 || TR辺長≤l_init×phase2_early_frac)
  │       ──YES→ 残差 Micro-GP fit → local CEM → return (mode="tandem_gp")
  │              (GP fit 失敗時はこの分岐を抜けて global へフォールバック)
  │
  ├─ (G) 各 TR で CEM 実行 → 候補プール生成
  ├─ (H) 獲得関数（TS/EI/UCB、デフォルト "ts"）でプール採点、feasibility で減点
  ├─ (I) 多様性付き貪欲選択で batch_size 件選抜
  └─ (J) EI 停滞カウンタ更新 → JSON で return (mode="tr_cem")
```

デフォルト off の実験フラグ（`enable_mads_poll`/`cem_diverse_starts`/`joint_batch_select`）は
上記の基本フローに差し込まれる形で存在するが、いずれも実 CFD A/B で棄却済み
（CLAUDE.md「棄却済み」節、docs/BENCHMARK.md §23 参照）。デフォルト挙動には影響しない。

**なぜ単一関数か:** パイプラインの各段が前段の中間結果（正規化値・ensemble・
TR 状態）を共有するため、分割するとデータの受け渡しが煩雑になる。代わりに
各段の**部品**（tr.rs, cem.rs, …）をモジュール分割し、lib.rs はオーケストレータ
に徹する構成を取っている。

---

## 5. Cold Start — Halton 準乱数（candidate.rs）

feasible な評価数が `n_init` 未満の間は、サロゲートを訓練せず
**Scrambled Halton 列**で空間充填サンプルを返す。

### 5.1 Halton 列の構成

各次元 j に互いに異なる素数 `PRIMES[j]` を基底とする Van der Corput 列を割り当てる:

```rust
fn van_der_corput(mut n: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    while n > 0 {
        result += f * (n % base) as f64;
        n /= base;
        f /= base as f64;
    }
    result
}
```

これは整数 n を base 進展開し、桁を小数点の鏡像に反転させる操作。
n = 1,2,3,... と進むにつれ [0,1] を「最も隙間の大きい場所」から埋めていく。

### 5.2 スクランブル（seed 対応）

素の Halton 列は seed 概念がなく毎回同一になる。そこで次元ごとに乱数オフセットを
加えて mod 1 する **additive scramble** を施す:

```rust
let offsets: Vec<f64> = (0..n_dims).map(|_| rng.gen::<f64>()).collect();
((raw + offsets[j]) % 1.0) as f32
```

これで (a) seed が同じなら完全再現、(b) seed が違えば異なる列、(c) 低不一致性
（low-discrepancy）は保たれる、の 3 つが同時に成立する。

### 5.3 高次元フォールバック（LHS）

素数テーブルは 128 個まで。高素数基底の Halton は次元間相関が強くなるという
既知の欠陥があるため、**129 次元以上は Latin Hypercube Sampling に切り替える**:

```rust
if n_dims > PRIMES.len() {
    return lhs(n_samples, n_dims, seed);
}
```

LHS は各次元を n_samples 等分し、各区間から 1 点ずつ独立な置換で取る方式。
Halton ほどの均一性はないが任意次元で破綻しない。
（このフォールバックは 200D ベンチマークで assert panic が発覚して追加された。
失敗モードを「panic」から「品質の僅かな低下」に変える防御的修正。）

### 5.4 なぜ純乱数でなく準乱数か

n_init=50 程度の小サンプルでは、純乱数はクラスタと空白を作りやすい。
初期サンプルの空間充填性はサロゲートの初期品質と TR の初期中心の質を直接決める
ため、低不一致列を使う価値が大きい。LHS でなく Halton をデフォルトにしたのは
**逐次拡張性**（任意の n で打ち切っても均一）のため。LHS は n を先に固定する必要がある。

---

## 6. 値の正規化（normalize.rs）

```rust
pub fn zscore(vals: &[f32]) -> (Vec<f32>, f32, f32) {
    // Bessel 補正 (n-1 で割る)。std は 1e-8 でクランプ
}
```

目的値を平均 0・分散 1 に標準化してからサロゲートに学習させる。

**理由:**
1. MLP の学習は出力スケールに敏感（学習率 5e-4 固定のため、値が 10⁶ オーダー
   だと発散し、10⁻⁶ だと学習しない）
2. EI の計算 `(m - f_best)/s` がスケール不変になる

**重要な罠と対策（types.rs のコメントに明記）:** TR の `best_value` は
**生の値（z-score 前）**で保存する。z-score は毎回データセット全体から再計算
されるため、正規化済み値を保存すると新データ追加でスケールがずれ、
過去の best と今回の best の比較（成否判定）が壊れる。
「ラウンドをまたいで比較する値は raw、1 ラウンド内で完結する値は normalized」
という規律が貫かれている。

---

## 7. サロゲートモデル — MLP Bootstrap Ensemble（surrogate.rs）

### 7.1 ネットワーク構造

```
入力 d → Linear(d,64) → ReLU → Linear(64,64) → ReLU
       → Linear(64,32) → ReLU → Linear(32,1)
```

これを **5 本独立に**訓練してアンサンブルにする。フレームワークは
`burn`（`ndarray` バックエンド = 純 CPU、BLAS 不要）。

### 7.2 Bootstrap による不確実性推定

GP と違い MLP は予測分散を持たない。そこで:

1. 各メンバー i は訓練データから**復元抽出**（bootstrap）したデータセットで学習
2. 予測時は 5 本の出力の**平均を μ、標準偏差を σ** とする

```rust
let boot_idx: Vec<usize> = (0..n).map(|_| boot_rng.gen_range(0..n)).collect();
```

復元抽出により各メンバーは約 63% のユニークデータしか見ない。データが密な領域
では 5 本の予測が一致し σ→小、疎な領域では学習データの違いが予測のばらつきと
して現れ σ→大。これがベイズ最適化に必要な「不確実性」の代用品になる。

**なぜ GP でなく MLP アンサンブルか（プロジェクトの核心的トレードオフ）:**

| | GP (BoTorch/HEBO) | MLP Ensemble (TRust-BO) |
|---|---|---|
| 訓練コスト | O(n³)（Cholesky） | O(n × epochs)、実質線形 |
| 予測コスト | O(n) /点 | O(1) /点 |
| 不確実性の質 | 理論的に校正済み | 経験的・無校正 |
| 高次元 | カーネル設計が難しい | そのまま動く |

budget=500・50D で BoTorch が 297s/run かかるのに対し TRust-BO が 28.6s/run で済む
（ベンチ実測、10.4× 差）のはこの差。100D では 504s vs 73s（6.9× 差）。
**校正された不確実性を捨てて速度とスケーラビリティを買う**のが本質的な賭けであり、
小 budget（評価 50 回以下）では GP 系に負け、budget≥100 で逆転するという
実測結果はこのトレードオフの素直な帰結である（詳細は `docs/BENCHMARK.md`）。

### 7.3 ウォームスタート — 41% の訓練時間削減

毎ラウンド、訓練済み重みを burn の `BinBytesRecorder` でバイト列化し
**hex 文字列**にして Python に返す。次ラウンドはその重みから fine-tune する:

```rust
let mut model = if let Some(hex) = warm_hex {
    // hex → bytes → Record → load_record で前回の重みを復元
} else {
    Mlp::new(n_dims, &device)   // cold start
};
```

データは毎ラウンド数点しか増えないため、前回の重みは良い初期値になる。

**収束チェックの非対称性:**

```rust
let check_every = if warm { 10 } else { 50 };
// check_every ごとに相対改善 < 1% なら early stop
```

ウォームスタート時は数十エポックで収束するため、チェック間隔を 50→10 に詰めて
無駄なエポックを刈る。これが「~41% 訓練時間削減」の実装実体。

**なぜ hex 文字列か:** 状態は JSON で往復させる設計（§1.2）のため、バイナリを
JSON-safe にする必要がある。base64 でなく hex なのは実装の単純さ優先
（依存ゼロ、デバッグ容易）。サイズは膨れる（~1MB/round）が、既知の非効率として
受容し「バイナリ転送は将来課題」と README に明記している。

### 7.4 シードの流儀

メンバー i のシードは `seed + i × 黄金比ハッシュ定数` で分散させる:

```rust
B::seed(seed.wrapping_add(i as u64 * 0x9e3779b97f4a7c15));
```

`0x9e3779b97f4a7c15` は 2⁶⁴/φ（黄金比）。連番シード間の相関を避ける
標準的なテクニック（Fibonacci hashing と同じ定数）。

---

## 8. Trust Region 動力学（tr.rs）

TuRBO（Eriksson et al. 2019）の Trust Region 管理を実装する。
「グローバルなサロゲートは高次元で当てにならない。**当てになる局所**に探索を
限定し、成功すれば広げ、失敗すれば狭める」が TuRBO の思想。

### 8.1 TR の構造

```rust
pub struct TrustRegionState {
    center: Vec<f32>,      // 中心（通常 = global best の位置）
    side_length: f32,      // 辺長 L（[0,1] 空間）
    success_count: usize,
    failure_count: usize,
    best_value: f32,       // raw 値（§6 の理由で z-score 前）
    active: bool,
    warmup_remaining: usize,
}
```

TR は中心 ± L/2 の超立方体（[0,1] にクランプ）。

### 8.2 更新則（update_tr）

毎 `propose()` 呼び出しで 1 回更新される:

```
改善があった場合:
  rel_improvement = (new_best - old_best) / |old_best|
  ├─ rel ≥ 1%  → center を新 best に移動、success_count += 1
  └─ rel < 1%  → 「中立」: center 移動せず、カウンタも動かさない
改善がない場合:
  failure_count += 1 (warmup 中は据え置き)

success_count ≥ τ_succ(3) → L ← min(2L, l_max)、カウンタリセット
failure_count ≥ τ_fail(5) → L ← L/2、カウンタリセット
L < l_min(0.5⁷) → active = false（枯渇）
```

**「中立帯」の設計（オリジナル TuRBO からの逸脱）:**
素の TuRBO は任意の改善を success と数えるが、本実装は **1% 未満の改善を
中立**として center 移動もカウントもしない。浮動小数点ノイズ程度の改善で
center が小刻みに動くと CEM のスタート点が不安定になり、また success が
溜まって TR が不当に拡大する。「意味のある改善だけが TR を動かす」という
安定化フィルタ。README の "Neutral center stability" がこれ。

**τ_fail の次元適応:** `tau_fail == 0` を指定すると `max(n_dims, 4)` になる。
高次元ほど 1 バッチで改善する確率が下がるため、失敗許容を次元に比例させる
（ベンチマークハーネスの BoTorch 実装も同じ `max(dim, 5)` を使っており、
TuRBO 論文の流儀に揃えている）。

### 8.3 Warmup（早期収縮防止）

`init_warmup > 0` を設定すると、TR 初期化直後の数ラウンドは failure_count を
増やさない。初期 TR はまだサロゲートが粗く、最初の 1〜2 バッチで改善が出ない
のは正常なので、それで即縮小に向かうのを防ぐ。デフォルト 0（無効）なのは
既存テストの seed 依存挙動を変えないため（types.rs のコメントに経緯が残る）。

### 8.4 Restart（枯渇からの再出発）

L < l_min まで縮んだ TR は「その局所を掘り尽くした」と見なす。
単一 TR の場合は **global best を中心に L=l_init で再起動**（TuRBO 論文準拠）。
ただし Phase 2 が有効な場合、この枯渇イベントは restart でなく
**Phase 2 への遷移シグナル**として扱われる（§12.2）。

---

## 9. CEM — 候補生成（cem.rs）

TR 内で獲得関数を最大化する点を見つける内部最適化に
**Cross-Entropy Method** を使う。

### 9.1 アルゴリズム

```
mu ← スタート点, sigma ← L/6 (全次元)
repeat n_cem_iters(25) 回:
  1. N(mu, diag(sigma²)) から 512 点サンプル（Box-Muller）、TR 境界にクランプ
  2. アンサンブルで予測 → 獲得関数スコア
  3. スコア上位 10%（≈51 点）をエリートとして選抜
  4. mu ← エリートの平均、sigma ← エリートの標準偏差
  5. max(sigma) < L×10⁻³ なら早期終了（収束）
return 最終エリート集合（51 点）
```

**なぜ勾配法でなく CEM か:**
1. アンサンブルの獲得関数面は ReLU の折れ目だらけで勾配が不安定
2. CEM は予測を**バッチで**評価できる（512 点まとめて forward）。
   MLP の forward は行列演算なのでバッチ評価が圧倒的に効率的
3. 多峰的な獲得関数面でも分布ごと探すため局所トラップに比較的強い
4. 実装が単純で panic 要素がない

**σ_init = L/6 の根拠:** 正規分布の ±3σ ≈ TR 全幅。つまり初期分布が
ちょうど TR を覆う。クランプで境界外は壁に張り付くが、イテレーションが進むと
分布は内側に収縮していく。

### 9.2 マルチスタート

各 TR につき **TR 内の既知トップ 3 点**からそれぞれ CEM を起動する
（TR 内に点がなければ center の 1 本のみ）:

```rust
let start_points = in_tr_sorted[..3.min(len)]  // 値の良い順トップ3
```

3 本 × 51 エリート ≈ 最大 153 点が候補プールに入る。獲得関数面が多峰でも
複数の盆地を拾える。スタート点を「既知の良い点」にするのは、改善は良い点の
近傍で起きやすいという exploit 寄りの判断（探索は σ の広さと EI が担う）。

### 9.3 序盤のイテレーション制限

```rust
let effective_n_cem_iters = if n_complete < 3 * config.n_init {
    config.n_cem_iters.min(16)
} else { config.n_cem_iters };
```

データが少ない序盤はサロゲートが粗く、その獲得関数面を 25 イテレーションかけて
精密に最適化するのは**過剰最適化**（モデルの間違いに過剰適合する）。
評価数が 3×n_init に達するまでは 16 イテレーションに制限する。

---

## 10. 獲得関数（acquisition.rs）

### 10.0 Thompson サンプリング風乱択（"ts"、デフォルト）

```
TS(x) = μ(x) + z·σ(x),   z ~ N(0, 1)  (候補ごとに独立サンプル)
```

アンサンブル予測の周辺分布 N(μ, σ²) から候補ごとにスコアを 1 回サンプルする
乱択獲得関数（厳密な Thompson sampling ではなく、その周辺分布近似）。
CEM の各イテレーションで再サンプルされるため探索に自然な多様性が生まれ、
50–100D の合成多峰関数ベンチマークで EI 比リグレットが改善したため 2026-07
にデフォルト化した(数値は `BENCHMARK.md` §14、要約は `CLAUDE.md` 参照)。
非有限予測の候補は f32::MIN に落として実質除外する。制約付きでは
P(feasible) 乗算の前にスコアを非負へシフトする(符号付きのままだと
infeasible 候補が有利になるため)。なお EI 停滞検出(Phase 2 遷移シグナル)
は acquisition="ei" 限定のまま。

**注意**: 上記の改善は合成多峰関数限定。実 CFD 様の滑らかな応答面
(NeuralFoil 翼型ベンチ)では逆に "ei" が優位という結果が出ている
(`BENCHMARK.md` §15.2)。CFD 系の実務では `acquisition="ei"` を推奨。

### 10.1 Expected Improvement（"ei"）

```
EI(x) = (μ(x) − f_best) Φ(z) + σ(x) φ(z),   z = (μ − f_best)/σ
```

（内部最大化なので符号は教科書の最小化版と逆。）
σ < 10⁻⁶ の点は EI=0（予測が確実 → 改善余地なし）として除外する。

正規 CDF は **Abramowitz & Stegun 26.2.17 の多項式近似**（誤差 < 7.5×10⁻⁸）
で計算する。`erf` に依存しないのは、`std` のみで完結させ外部数学ライブラリへの
依存を避けるという方針の徹底。

### 10.2 UCB（代替）

`UCB(x) = μ(x) + β σ(x)`（β=2.0）。config で選択可能。**現行デフォルトは "ts"**
（2026-07-11、合成多峰関数で EI 比リグレット 10–14% 改善のため変更。実 CFD では
逆に EI が ts に勝つため `acquisition="ei"` 推奨、詳細は CLAUDE.md 参照）。
EI は Phase 2 の遷移シグナル（EI 停滞検出、§12.2）の前提となる値域を提供するため、
`enable_phase2` 使用時は "ei" または "ts_ei" との組み合わせが有効。

---

## 11. バッチ選択（batch.rs）

CEM が作った候補プール（~153 点）から batch_size 件を選ぶ。
単純に上位 k 点を取ると**ほぼ同一の点が k 個**選ばれる（CEM は収束した分布から
サンプルするため）。そこで多様性制約付き貪欲法を使う:

```
repeat batch_size 回:
  1. 未除外のうち最高スコアの点を選択
  2. その点から L2 距離 0.1 未満の点を全て除外
```

除外半径 0.1 は [0,1] 空間での値。バッチ内の点同士が最低 0.1 離れることを保証し、
並列評価の情報利得を高める（同じ場所を 4 回測っても 1 回分の情報しかない）。

**不足時の挙動の使い分け（後方互換の配慮):**
- デフォルト: プールが痩せていて batch_size 件取れなければ**不足を許容**
  （既存挙動の維持）
- `enable_phase2=true` 時: TR 内の一様乱数で **backfill して必ず batch_size
  件返す**。Phase 2 系のベンチで「バッチが 3 件しか返らず評価ループが狂う」
  事故を防ぐため

---

## 12. Phase 2 — Tandem Residual Micro-GP（gp.rs + lib.rs）

本プロジェクト独自の機構。**MLP アンサンブルの「終盤の鈍さ」を GP で補正する**。

### 12.1 動機

MLP アンサンブルは大域的な形状把握は得意だが、最適点近傍の微細な曲率を
表現するには鈍い（ReLU 区分線形 + 5 本平均のスムージング）。終盤戦では
「best 近傍のわずかな谷」を見つける精度が律速になる。
一方 GP は点数が少なければ O(n³) でも安く、局所の補間精度は抜群。

そこで:

```
f(x) ≈ MLP_ensemble(x) + GP_residual(x)
```

**MLP の予測残差**（実測 − MLP 予測）だけを、best 近傍の点群で小さな GP に
学習させる。大域構造は MLP が、局所補正は GP が担う「タンデム」構成。

### 12.2 遷移条件（3 つのシグナル）

Phase 2（local モード）に入るのは以下が**すべて**成立したとき:

```rust
let enter_local = phase2_armed                        // enable_phase2 && 単一TR
    && feasible_params.len() >= max(phase2_min_evals, 3 * n_init)  // 最小データ量
    && (sticky_local            // 既に local（一度入ったら維持）
        || tr_exhausted         // TR が l_min まで縮んで枯渇した
        || stagnation_count >= 5);  // EI 停滞が 5 ラウンド連続
```

- **TR 枯渇**: 「局所を掘り尽くした」の最も明確なシグナル。Phase 2 有効時は
  restart せず枯渇状態を保留し、local 遷移の引き金にする。
- **EI 停滞**: 全候補の max EI < 10⁻⁵ が 5 回連続 = サロゲートが
  「もうどこにも改善はない」と言っている状態。モデルの解像度不足が疑われる
  ため、GP 補正に切り替える。
- **sticky**: 一度 local に入ると以後ずっと local（GP fit が失敗しない限り）。
  phase は Python 側を往復する状態変数で、ラウンドをまたいで持続する。

**最小データ量ガード（3×n_init）**: GP は残差を学習するため、MLP がある程度
学習できている必要がある。データが少ない段階で入ると残差がただのノイズになる。

### 12.3 local モードの処理

```
1. global best から近い順に n_local = max(50, d+2) 点を選ぶ
2. その点群での残差 r_i = norm_value_i − MLP(x_i) を計算
3. Micro-GP を残差にフィット（失敗したら global へフォールバック）
4. TR を凍結（更新も restart もしない）。L_frozen = max(L, 0.02)
5. best ± L_frozen/2 の箱の中で CEM（予測 = MLP + GP残差、σ = GP分散）
6. EI で採点 → 半径 min(L×0.1, 0.1) の貪欲選択 → 不足は乱数 backfill
7. mode="tandem_gp", phase="local" で返す
```

**TR 凍結の理由:** local モードでは探索箱は GP の信頼領域（= best 近傍の
データ密集域）であるべきで、TuRBO の成功/失敗動力学で動かす意味がない。
むしろ失敗カウントで縮み続けると GP の学習点が箱の外に出てしまう。

**σ に GP 分散のみを使う理由（MLP の σ を捨てる）:** local モードの目的は
「GP がまだ自信のない方向を探る」こと。MLP のアンサンブル分散は大域的な
不確実性であり、best 近傍ではほぼ一様に小さく情報がない。

### 12.4 Micro-GP の実装（gp.rs）

外部 BLAS/LAPACK に依存しない、約 300 行の自己完結 GP:

- **カーネル**: Matern 5/2。`d ≤ 10` なら ARD（次元ごとの lengthscale）、
  `d > 10` なら isotropic（1 本）に自動切替。高次元 ARD はハイパーパラメータ
  空間が広すぎてランダム探索では当たらないため。
- **ハイパーパラメータ最適化**: 勾配法ではなく**ランダム多起点探索**。
  `n_hypers = max(40, 4d)` 個の候補 (lengthscale, sf², noise) を対数一様に
  サンプルし、周辺対数尤度が最大のものを採用:

  ```
  log ML = −½ rᵀK⁻¹r − ½ log|K| − (n/2) log 2π
  ```

  勾配法（L-BFGS 等）を避けたのは実装の単純さと「panic しない」保証のため。
  局所最適に嵌まる勾配法より、粗いが頑健なランダム探索を選んだ。
- **数値安定化の多層防御**:
  1. 近接重複点（L∞ < 10⁻⁹）を事前除去 — K の特異化を防ぐ
  2. 残差を標準化してからフィット（predict で復元）
  3. Cholesky 失敗時は jitter を 10⁻⁶ から 10 倍ずつ最大 8 回追加して再試行
  4. それでも失敗なら `Err(CholeskyFailed)` を返す — **panic しない**
- **エラー契約**: `TooFewPoints` / `DegenerateResiduals` / `CholeskyFailed` /
  `NonFinite` のいずれでも、呼び出し側（lib.rs）は global CEM パスへ
  フォールバックする。**Phase 2 は「失敗しても通常動作に戻るだけ」の
  純粋な上乗せ**として設計されており、これが「1 フラグで有効化でき、リスクが
  低い」と言える根拠。
- **f64 で計算**: MLP/CEM は f32 だが GP 内部は全て f64。Cholesky は条件数に
  敏感で、f32 では jitter 再試行が頻発するため。f32↔f64 変換は cem.rs の
  `combined_predict` に集約されている。

**`n_hypers` の次元適応（修正済み）:** 旧実装では `40.max(4 * n_dims)` とし、
200D で 800 候補 × O(n³) 尤度評価が走り 544s/run の主因となっていた。
d > 10 で isotropic（探索空間 3 次元固定）に切り替わるため線形スケールは矛盾。
現在の実装:
```rust
let n_hypers = if n_dims <= 10 { 40.max(4 * n_dims) } else { 60 };
```
修正後: 200D で **3.8s/run**（144× 高速化）。50D/100D の品質は変化なし。

### 12.5 gp.rs が自己完結である理由

ファイル冒頭のコメントにある通り、`branin_demo`（検証用バイナリ）が
`#[path]` でこのファイルを直接取り込んで同一コードを検証するため、
crate 内 import を持たない（std + rand のみ）。テスト容易性のための制約。

---

## 13. 制約処理 — Feasibility Surrogate

`tell()` で `feasible: false` が報告されると、**実行可能性を予測する
第二のアンサンブル**を訓練する:

- 訓練データ: 全試行（feasible/infeasible 両方）、ラベルは 1.0/0.0
- 主サロゲートと同じ MLP 構造・ウォームスタート機構を流用
- 獲得関数を `EI × clamp(P(feasible), 0, 1)` で減点

つまり「改善が見込めても infeasible 確率が高い領域は選ばない」ソフト制約。
拒絶でなく乗算ペナルティなのは、制約境界ギリギリの点（しばしば最適点がある）
を完全に殺さないため。infeasible 試行が 1 件もなければこの機構は起動せず、
コストはゼロ。

---

## 14. TuRBO-M（Multi Trust Region）

`n_trs > 1` で並列 TR モードになる（実装済みだが experimental、
CFD スケールでは単一 TR が安定とされ deprioritized）。

- **初期化**: TR₀ は global best、TR₁₊ は **Greedy Farthest-Point**
  （既選択点群から最も遠い点を貪欲に追加）で多様に配置
- **更新**: 各 TR は**空間的帰属** — 自分の境界内のデータだけで成否判定
  （`update_tr_spatial`）。境界内にデータがなければ failure 扱い
- **restart**: TR₀ は global best へ（搾取継続）、TR₁₊ は「100 点の Halton
  候補のうち全観測点から最も遠い点」へ移動（密度考慮の探索）。
  かつ `best_value = global_best` にセットするため、**グローバルベストを
  超えない限り success にならない**（探索 TR に厳しい基準を課す）
- **バッチ配分**: 各 TR から最低 1 件を保証（スロット飢餓防止）した後、
  残り枠は全プールから貪欲選択

**サロゲートは全 TR で共有**（単一モデル）。TuRBO 論文は TR ごとに独立 GP を
持つが、本実装はアンサンブル 1 つを使い回す。訓練コスト n_trs 倍を回避する
妥協で、TR 間の独立性は CEM の探索範囲の違いだけで担保される。

---

## 15. 設計思想の総括

コード全体を貫く原則を抽出すると:

### 15.1 「panic しない」の徹底
- Halton 範囲外 → LHS フォールバック
- GP fit 失敗（4 種のエラー全て）→ global パスへフォールバック
- Cholesky 失敗 → jitter 8 段再試行 → それでもだめなら Err
- NaN/inf スコア → `f32::MIN` に置換して実質除外
- バッチ不足 → 乱数 backfill

最適化エンジンは長時間ジョブの中で呼ばれる。1 回の数値的不運でプロセスが
死ぬことは許されず、**全ての失敗モードに「品質は落ちるが動き続ける」経路**
が用意されている。

### 15.2 「依存を増やさない」
- GP は自前実装（BLAS なし）、正規 CDF は多項式近似（erf なし）、
  NN は burn + ndarray（GPU ランタイムなし）
- 「ラップトップで pip install して動く」という存在意義に直結する制約

### 15.3 「後方互換をコードで守る」
- `#[serde(default)]` で旧バージョンの保存ファイルを読める
- 新機能（warmup, phase2, multi-TR, backfill）はすべてデフォルト無効 or
  デフォルト値で既存挙動を変えない
- シード加算規則まで「n_trs=1 との後方互換を保つ」コメント付きで設計
  （`seed+1,2,3` が単一 TR 時代と一致するようオフセット計算している）

### 15.4 「比較する値のスケール規律」
- ラウンドをまたぐ比較は raw 値（TR best_value）
- ラウンド内で完結する計算は z-score 値（サロゲート・EI）
- GP 内部はさらに残差を再標準化
- この三層のスケール管理が崩れると成否判定が静かに壊れるため、
  types.rs に理由がコメントで固定されている

### 15.5 計算量の配分
1 ラウンドの支配項は:

| 処理 | コスト | 備考 |
|---|---|---|
| MLP 訓練 ×5(+5) | O(n·epochs)・early stop あり | warm start で大幅短縮 |
| CEM | 25 iter × 512 点 × 3 start のバッチ forward | MLP forward は安い |
| Micro-GP fit | O(n_hypers × n_local³) | n_local≈50 なら軽い、d=200 で爆発 |
| TR/バッチ選択 | O(pool²) | pool~150 で無視できる |

GP 系オプティマイザの O(n³)（n = 全評価数）と違い、**全評価数 n に対して
ほぼ線形**なのが大 budget で勝てる構造的理由。

---

## 16. 既知の弱点と限界

実測・コードレビューで判明しているもの:

1. **小 budget（≤50 評価）では GP 系に負ける** — アンサンブルの不確実性は
   データが少ないと未校正で、初動の獲得関数の質で BoTorch/HEBO に劣る。
   ベンチ実測: budget=50 の全問題・全次元で BoTorch が優位。逆転点は
   Ackley/Levy で budget=100、Rastrigin で budget=200（`docs/BENCHMARK.md` 参照）。
2. ~~**Micro-GP の n_hypers = max(40, 4d) スケール**~~ — **修正済み**。
   d > 10 で `n_hypers = 60` に固定。200D: 544s → **3.8s**（`docs/ROADMAP.md` Phase A）。
3. **重み転送が hex 文字列で ~1MB/round** — JSON 設計の代償。バイナリ化は将来課題。
4. **Categorical の 1 次元埋め込み** — カテゴリ数が多いと擬似順序バイアス。
5. **多目的の EHVI は 2 目的限定** — 閉形式 EHVI（`acquisition.rs::ehvi_2d`,
   `pareto.rs`, `hypervolume.rs`, Python `MultiObjectiveEngine`）は 2 目的のみ。
   3 目的以上は Chebyshev スカラー化（`method="chebyshev"`）を使う。
6. **Multi-TR は experimental** — サロゲート共有の妥協があり、CFD スケールでは
   単一 TR が安定。
7. **EI 停滞検出は acquisition="ei" 限定** — UCB 選択時は Phase 2 の
   遷移シグナルが TR 枯渇のみになる。
