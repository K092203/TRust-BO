# TRust-BO 実装ロードマップ

用途: CFD/エアロ最適化・物理シミュレーション最適化・高次元 BB 最適化  
実装順: **A → G → H → K**

---

## フェーズ概要

| フェーズ | 内容 | 規模 | ステータス |
|---------|------|------|-----------|
| **A** | Micro-GP `n_hypers` キャップ | 極小（1行修正 + ベンチ） | ✅ 完了 |
| **G** | 並列非同期評価ハーネス | 中（新モジュール） | ✅ 完了 |
| **H** | 実 CFD ベンチマーク | 大（環境構築含む） | ✅ H-1（NeuralFoil）・H-2（SU2 RANS）完了 |
| **K** | 多目的最適化 | 大（コアアルゴリズム変更） | ✅ K-1・K-2・K-2-8（実 CFD Cl/Cd）完了 |

ステータス凡例: 🔲 未着手 / 🔄 進行中 / ✅ 完了 / ⛔ ブロック中

---

## Phase A: Micro-GP `n_hypers` キャップ

### 目的
200D での実行時間 **544s → 目標 <80s** へ。300D+ を現実的にする。

### 背景
`src/gp.rs` で `n_hypers = max(40, 4 * n_dims)` としているが、
`n_dims > 10` で isotropic に切り替わると探索空間は (ls, sf², noise) の **3 次元固定**。
それにも関わらず候補数が線形スケールするのは矛盾であり 200D で 800 候補になる。

### 変更対象
`src/gp.rs` 行 76〜77

```rust
// Before
let n_hypers = 40.max(4 * n_dims);

// After
let n_hypers = if n_dims <= 10 {
    40.max(4 * n_dims)   // ARD: (n_dims + 2) パラメータ → 線形スケール維持
} else {
    60                    // isotropic: 3 パラメータ固定 → 60 候補で十分
};
```

### サブタスク

| # | 作業 | ステータス |
|---|------|-----------|
| A-1 | `src/gp.rs` 修正 | ✅ |
| A-2 | `cargo test` 全パス確認 | ✅ 19/19 passed |
| A-3 | 200D タイミングベンチ（目標 <80s） | ✅ **3.8s**（旧 544s、144× 高速化） |
| A-4 | 50D/100D の品質が変化しないことを確認 | ✅ 50D: 6.22 / 100D: 7.63（変化なし） |
| A-5 | `docs/ALGORITHM.md` §12.4 更新 | ✅ |

### 合格基準
- [ ] `cargo test` 全パス
- [ ] Ackley 200D / budget=100 の実行時間 < 80s（現状 544s）
- [ ] 50D/100D の median best_value 変化 ±0.5% 以内

---

## Phase G: 並列非同期評価ハーネス

### 目的
CFD の HPC ジョブキュー（SLURM/PBS）に対応した  
**「K ジョブ並列 → 完了次第 tell → 即 ask(1)」** のローリングウィンドウ実行を可能にする。

### 設計方針

```
現在:  [ask4]→[run×4全完了待ち]→[tell4]→[ask4]→...
                  ↑ 1ジョブが遅れると全スロットが止まる

目標:  常に max_concurrent スロットを埋め続ける
       完了1件 → tell(1) → ask(1) → 即投入
```

**pending 点の扱い**: Phase G-1 では「無視」（diversity 制約で代替）。
hallucination は Phase H の実結果を見てから要否を判断。

### インターフェース設計

**新規ファイル: `python/trust_bo/rolling_engine.py`**

```python
class JobEvaluator(ABC):
    @abstractmethod
    def submit(self, candidate: dict) -> str:
        """1点をジョブとして投入。ジョブIDを返す"""

    @abstractmethod
    def poll(self) -> list[tuple[str, float, bool]]:
        """完了済み (job_id, value, feasible) を返す。ブロックしない"""

class RollingTRustBOEngine:
    def __init__(self, base_engine, evaluator, max_concurrent=4, poll_interval=5.0):
        ...
    def run(self, budget: int) -> dict:
        """budgetに達するまでローリング実行。best result を返す"""
```

**使用例（SLURM）**:

```python
class SlurmCFDEvaluator(JobEvaluator):
    def submit(self, candidate):
        jid = subprocess.check_output(["sbatch", ...]).decode().strip()
        return jid
    def poll(self):
        # squeue確認 → 完了分を返す
        ...

engine = RollingTRustBOEngine(
    base_engine=TRustBOEngine(space, direction="maximize", seed=42),
    evaluator=SlurmCFDEvaluator(),
    max_concurrent=8,
    poll_interval=30.0,
)
engine.run(budget=200)
```

### サブタスク

| # | 作業 | ステータス |
|---|------|-----------|
| G-1 | インターフェース API 確定（JobEvaluator / RollingEngine） | ✅ |
| G-2 | `python/trust_bo/rolling_engine.py` 実装 | ✅ |
| G-3 | `MockEvaluator`（sleep ベース）作成 | ✅ |
| G-4 | 単体テスト: Ackley 50D / budget=50 / concurrent=4 | ✅ 完走・差 4.9% |
| G-5 | 逐次実行との best_value 比較（同 seed） | ✅ ±10% 以内 |
| G-6 | ジョブ失敗・タイムアウトのハンドリング | ✅ failure_rate=0.2 で完走確認 |
| G-7 | `SlurmEvaluator` テンプレート + ドキュメント | ✅ rolling_engine.py に実装済み |
| G-8 | `python/trust_bo/__init__.py` エクスポート追加 | ✅ |

### 合格基準
- [ ] MockEvaluator（sleep 0〜5s）で budget=50 / concurrent=4 が完走
- [ ] 逐次実行との best_value 差 ±10% 以内（完了順が変わるため完全一致は不要）
- [ ] ジョブ失敗時にプロセスが死なずスキップして継続

---

## Phase H: 実 CFD ベンチマーク

### 目的
合成関数での優位が実流体問題に転移することを実証する。  
TRust-BO vs BoTorch vs CMA-ES の 3 者比較。

### 2 段階構成

| ステージ | ソルバー | 評価時間 | 次元 | 目的 |
|---------|---------|---------|------|------|
| **H-1** | XFOIL | ~3〜5s/eval | 16D（CST） | パイプライン検証 |
| **H-2** | SU2 RANS | ~10〜30min/eval | 20〜30D（CST） | 学術的検証 |

### H-1: XFOIL ベンチマーク

**問題設定**:
- 翼型: CST パラメータ化（上面 8 係数 + 下面 8 係数 = **16D**）
- 目的: Cl/Cd 最大化（α=4°固定）
- 手法: TRust-BO+P2 / BoTorch / CMA-ES / Random
- Budget: 200 / Seeds: 10(TRust-BO,Random) / 3(BoTorch)

**サブタスク**:

| # | 作業 | ステータス |
|---|------|-----------|
| H-1-1 | NeuralFoil インストール・動作確認（xfoil 代替） | ✅ NACA2412 Cl/Cd=111 確認 |
| H-1-2 | `benchmarks/cfd_neuralfoil_benchmark.py` 作成 | ✅ 16D CST、4手法対応 |
| H-1-3 | TRust-BO / BoTorch / CMA-ES / Random ベンチ実行 | ✅ 完了（28 runs） |
| H-1-4 | 結果分析・`docs/BENCHMARK.md` 更新 | ✅ §10 追記済み |
| H-1-5 | Phase G rolling engine との統合テスト | ✅ ALL PASSED（budget=50 / concurrent=4 / failure_rate=0.2） |

### H-2: SU2 RANS ベンチマーク

**問題設定**:
- 翼型: CST 16D（上面 8 + 下面 8、H-1 と同一パラメータ化）
- 設計空間: upper [0.05, 0.35], lower [-0.35, 0.05]（実行可能率を seed 一様に確保しつつ一部 infeasible を残す）
- 流体条件: Ma=0.3, Re=3×10⁶, α=2°, SA 乱流モデル
- 目的: Cl/Cd 最大化
- 手法: TRust-BO+P2 / BoTorch / CMA-ES / Random
- Budget: 100 / Seeds: 3（Phase G の rolling engine を使用）

**実装メモ（環境制約への対応）**:
- SU2: ビルド不要の **prebuilt OpenMP バイナリ**（v8.5.0 omp、静的リンク、MPI 不要）を使用。
- メッシュ: gmsh が libGLU に依存し sudo 不可で導入できないため、
  **純 Python 構造 O-mesh 生成器**（`benchmarks/su2/airfoil_mesh.py`）を自作。
  境界層クラスタリング（first cell 1e-5, y+~0.4）+ 翼型重心を射線中心とする放射押し出し。

**サブタスク**:

| # | 作業 | ステータス |
|---|------|-----------|
| H-2-1 | SU2 インストール（prebuilt omp）・NACA0012 で Cl/Cd 確認 | ✅ Euler CL/CD=15.2 / RANS CL=0.231,CD=0.0127 |
| H-2-2 | 構造 O-mesh 生成器（純 Python、SU2 形式、BL クラスタ） | ✅ `airfoil_mesh.py`（全セル正面積） |
| H-2-3 | SU2 RANS 設定テンプレート + Python 実行ラッパー | ✅ `su2_runner.py`（Cauchy 収束, Cl/Cd パース, ~49s/eval） |
| H-2-4 | CST → mesh → SU2 RANS 一気通貫 + 任意形状頑健性 | ✅ 有効翼型 5/5 feasible、無効形状はグレースフルに infeasible |
| H-2-5 | `SU2LocalEvaluator`（JobEvaluator）+ Phase G rolling 統合 | ✅ `su2_evaluator.py`（rolling 検証: best Cl/Cd=25.3） |
| H-2-6 | ベンチハーネス（4 手法・バッチ並列）作成 | ✅ `su2_cfd_benchmark.py` |
| H-2-7 | 本番ベンチ実行 → `docs/BENCHMARK.md` 更新 | ✅ budget=100・4 手法・3 seeds 完了（2026-06-13）。TRust-BO+P2 median Cl/Cd=171.6 で全手法中最安定。 |

### 合格基準
- [ ] H-1: TRust-BO が NACA シリーズを上回る Cl/Cd を発見
- [ ] H-2: TRust-BO が BoTorch と同等以上の設計を 5× 以上速く達成
- [ ] メッシュ生成失敗（feasibility=False）を feasibility surrogate で適切に処理

---

## Phase K: 多目的最適化

### 目的
揚力最大化・抗力最小化などの**複数目的を同時最適化**し Pareto フロントを得る。  
CFD/エアロの実用ケースの多くは多目的問題（揚抗比のトレードオフ等）。

### 2 段階構成

| ステージ | 内容 | Rust 変更 |
|---------|------|----------|
| **K-1** | Chebyshev スカラー化 + Pareto 追跡 | なし（Python 層のみ） |
| **K-2** | EHVI 獲得関数（2 目的完全実装） | あり（新モジュール） |

### K-1: スカラー化（Python 層）

```python
# tell() に複数目的値を渡す
engine.tell(candidates, [
    {"values": [cl, -cd], "feasible": True},
    ...
])
# 内部で Chebyshev スカラー: max_i(w_i * |f_i - z*_i|) を最小化

# Pareto フロント取得
front = engine.pareto_front()  # 非支配解のリスト
```

**サブタスク**:

| # | 作業 | ステータス |
|---|------|-----------|
| K-1-1 | `tell()` の multi-value 対応（Python 層） | ✅ `multiobjective.py` |
| K-1-2 | Chebyshev / weighted-sum スカラー化実装 | ✅ ask-time re-scalarization 方式 |
| K-1-3 | Pareto 判定・フロント追跡（Python 側） | ✅ `_pareto_mask` + `hypervolume_2d` |
| K-1-4 | ZDT1 でスモークテスト | ✅ ZDT1 5D ratio 3.6×〜4.3× (>=2.0) |
| K-1-5 | H-1 の Cl/Cd 同時最適化で検証 | ✅ 16D ratio 1.2×〜1.6× (>=1.2) |

### K-2: EHVI 獲得関数（Rust コア）

**追加ファイル**:

```
src/
  hypervolume.rs   新規: 2D 超体積計算（WFG 簡略版）
  pareto.rs        新規: 非支配ソート・参照点管理
  acquisition.rs   変更: ehvi() 追加
  lib.rs           変更: 多目的 tell 対応
  types.rs         変更: values: Vec<f32> 追加
```

**サブタスク**:

| # | 作業 | ステータス |
|---|------|-----------|
| K-2-1 | `types.rs`: ProposeMoConfig / ProposeMoOutput 追加 | ✅ |
| K-2-2 | `pareto.rs`: 非支配ソート・2D フロント構築 | ✅ 単体テスト 4 件 |
| K-2-3 | `hypervolume.rs`: 2D 超体積計算（スイープライン） | ✅ 単体テスト 5 件 |
| K-2-4 | `acquisition.rs`: 2D EHVI 閉形式 | ✅ **MC 積分と一致**（テスト 6 件） |
| K-2-5 | `lib.rs`: `propose_mo`（目的別サロゲート + EHVI-CEM） | ✅ |
| K-2-6 | Python `MultiObjectiveEngine(method="ehvi")` + ZDT/高次元ベンチ | ✅ `zdt_ehvi_benchmark.py` |
| K-2-7 | NSGA-II 比較ベンチ実行・`BENCHMARK.md` 更新 | ✅ §11 追記 |
| K-2-8 | H-2 の Cl/Cd 同時最適化への適用 | 🔲（H-2 完了済み、着手可） |

### 合格基準
- [ ] K-1: ZDT1 で Pareto フロントの hypervolume が Random の 2× 以上
- [ ] K-2: ZDT2 で NSGA-II と同等以上の hypervolume を 10× 速く達成
- [ ] H-2（実 CFD）の Pareto フロントで意味のある揚抗トレードオフが確認できる

---

## 依存関係

```
A（n_hypers キャップ）
  └─ 全フェーズの前提。200D 対応なしで H・K に進むとボトルネックが残る。

G（並列評価ハーネス）
  └─ H-2 の前提。SU2 の 30min/eval を現実的にするには rolling eval が必須。

H-1（XFOIL）
  └─ H-2 の前提。パイプラインを先に完成させてから SU2 に移行。

K-1（スカラー化）
  └─ K-2 の前提。多目的の全体フローを先に確立してから EHVI に移行。
     H-1 と並行して進められる。
```

## 全体タイムライン（目安）

```
Week 1      : [A] 実装・ベンチ確認
Week 2〜3   : [G-1〜G-5] RollingEngine コア実装・テスト
Week 3〜4   : [G-6〜G-8] エラーハンドリング・SLURM テンプレート
Month 2     : [H-1] XFOIL パイプライン・ベンチ実行
              [K-1] スカラー化（H-1 と並行）
Month 2〜3  : [H-2-1〜H-2-4] SU2 環境構築・パイプライン
Month 3〜4  : [H-2-5〜H-2-7] SU2 本番ベンチ（G と統合）
Month 4     : [K-2-1〜K-2-5] EHVI Rust 実装
Month 5     : [K-2-6〜K-2-7] ベンチ・CFD への適用
```
