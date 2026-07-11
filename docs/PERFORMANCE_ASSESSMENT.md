# TRust-BO 性能アセスメント（OSS 公開判断用）

> 目的: 散在するベンチマーク結果を 1 か所に集約し、**証拠で言えること／言えないこと**を
> 正直に切り分けて、OSS 公開の GO/NO-GO 判断材料とする。
> 集計日: 2026-06-14。追加実行なし、既存 CSV のみから算出。
> **注(2026-07-11 追記)**: 本ファイルの数値は 2026-07-11 の rayon 並列化・獲得関数デフォルト
> "ei"→"ts" 変更(コミット d6a02f5)より前の測定。§3.1 の TRust-BO+P2 数値は特に、
> 当時のデフォルト "ei" で測定されたものであり **今のデフォルト "ts" では再現しない**
> (詳細は §3.1 末尾の注記、`docs/BENCHMARK.md` §15.2 参照)。

---

## 1. 結論（要約）

**「BoTorch TuRBO 比で 5–10× 高速、かつ高次元（≥50D）・中〜大 budget（≥100）で同等以上の品質を
出す CPU-only BO エンジン」** という主張はデータで裏付けられている。
一方「業界最高性能」「あらゆる問題で最良」とは言えない（低次元では BoTorch がやや上、SAASBO 未比較）。

公開判断としては、**主張範囲を限定すれば公開に値する**。詳細を以下に示す。

---

## 2. データで言えること（実証済み）

### 2.1 速度: BoTorch TuRBO 比 5–10× 高速 ★強い
出典: `midbudget_results.csv`（合成関数 50/100D, budget 100–300, 全 18 条件）。

| 次元 | 速度比（TRust / BoTorch、中央値） |
|---|---|
| 50D | **7.0 – 10.7×** |
| 100D | **5.0 – 6.1×** |

全条件で一貫。再現性が高く、GP の O(n³) を MLP アンサンブルで回避した設計の直接的帰結。

### 2.2 高次元・中〜大 budget の品質: BoTorch TuRBO 超え ★強い
出典: `midbudget_results.csv`（budget 100–300）, `large_budget_results.csv`（budget 500）。

- 合成関数 midbudget **16/18 条件で TRust-BO が品質勝ち**（中央値）。
- budget=500 では検証した **5/5 条件で TRust-BO 勝ち**（Ackley/Rastrigin/Levy, 50/100D）。
- 例（Ackley 100D b=300）: TRust=7.13 vs BoTorch=8.51。

### 2.3 実 CFD（SU2 RANS, H-2）: 最も安定 ★中（seeds 少）
出典: `su2_benchmark_results.csv`（16D CST, budget=100, 4 手法×3 seed）。

| 手法 | median Cl/Cd | 物理域(<300)に収まった seed |
|---|---|---|
| **TRust-BO+P2** | **171.6** | **3/3** |
| BoTorch_TuRBO | 126.1 | 3/3 |
| CMA-ES | 774.6 | 1/3 |
| Random | 316.1 | 1/3 |

TRust-BO は実 RANS（ノイジー・制約付き）で BoTorch を +36% 上回り、全 seed が物理的に妥当。

### 2.4 多目的 EHVI（K-2-8）: Chebyshev 超え ★弱い（2 seeds）
出典: `su2_mo_results.csv`（Cl 最大 + Cd 最小, budget=60, 2 seed）。
EHVI median HV=0.0239 vs Chebyshev 0.0165（**+45%**）。ただし 2 seed のみ、Chebyshev の方が
Pareto 点の多様性は高い。

---

## 3. データで言えないこと（未実証・限界）

### 3.1 低次元では BoTorch がやや上 ⚠ 重要
出典: `neuralfoil_benchmark_results.csv`（H-1, 16D 滑らか surrogate, Cl/Cd）。

| 手法 | median Cl/Cd | best |
|---|---|---|
| BoTorch_TuRBO | **241.4** | 245.4 |
| TRust-BO+P2 | 227.9 | **267.4** |
| CMA-ES | 223.3 | 265.5 |

**16D の滑らかな問題では BoTorch が中央値で上**（TRust は best では上回るが seeds 差あり）。
H-2（同じ 16D でも実 RANS でノイジー）では逆に TRust が勝つ → **TRust の優位は「高次元」か
「ノイジー・制約付き」の問題に出る**、という設計どおりの結果。低次元・滑らか・小 budget は不得手。

**注(2026-07-11)**: 上記 TRust-BO+P2 の 227.9 は当時のデフォルト獲得関数 `"ei"` での測定値。
2026-07-11 に獲得関数デフォルトが `"ts"` に変わったため、**今のデフォルト設定ではこの数値は
再現を期待できない**。実際 `docs/BENCHMARK.md` §15.2 で同系統の問題(同じ CST 16D, Cl/Cd 目的。
ただし seeds・budget は別条件の A/B)を "ts" vs "ei" で再検証したところ "ts" は "ei" に明確に
劣る(幾何平均比 0.86–0.92)ことを確認しており、
今のデフォルト設定のままだと BoTorch との差はむしろ**開く**方向になる。CFD 系のベンチマークで
TRust-BO を評価・再現する際は `config={"acquisition": "ei"}` を明示することを推奨。

### 3.2 小 budget の弱さ
Rastrigin budget=100（50D・100D とも）で BoTorch に負け。クロスオーバーは budget≈100–200。
それ未満は MLP の不確かさ推定が甘く不利。

### 3.3 SAASBO 未比較
高次元 BO のデファクト標準 SAASBO との比較が未実施（環境の OOM で保留）。
**「高次元 BO で最良」は現状主張できない。**

### 3.4 統計的厚みの不足
CFD は 3 seeds、MO は 2 seeds。学術的主張には最低 10 seeds が望ましい。CFD seed 間の
分散も大きい（TRust: 230/172/57）。

### 3.5 非物理アーティファクト
feasibility 判定が `CD>1e-6` のみのため、超薄翼で SA が CD≈0 を返し Cl/Cd=4707（CMA-ES）,
1142（Random）等の非物理値が混入。最小厚み・最小面積制約の追加が必要（能力拡張の課題）。

### 3.6 多目的は 2D 限定
EHVI 閉形式は 2 目的のみ。3 目的以上は未対応。

---

## 4. 公開判断の論点

### 胸を張れる主張（限定版）
> 「**GPU 不要・CPU-only で、高次元（≥50D）または実 CFD のようなノイジー・制約付き問題において、
> BoTorch TuRBO より 5–10× 速く、同等以上の品質を出す Trust-Region BO エンジン**。
> 翼型最適化の実 CFD パイプライン（SU2 RANS）と多目的最適化を同梱。」

### 避けるべき主張
- 「業界最高性能」「最速の BO」（SAASBO 未比較）
- 「あらゆる問題で最良」（低次元・小 budget・滑らかな問題では BoTorch が上）

### 公開の是非
- **賛**: 速度優位は即体感価値。CPU-only + SLURM + SU2 同梱は他に少なく訴求力あり。
  コード・CI・パッケージングは整備済み。アルファ版として十分。
- **要注意**: 「SAASBO と比べて？」は必ず来る → README に「比較対象は BoTorch TuRBO /
  CMA-ES / Random / NSGA-II、SAASBO 比較は今後」と明記して誠実に出す。
  seeds の少なさも limitation として明記。

### 推奨
**`Development Status :: 3 - Alpha` のまま、主張を限定して公開 GO** が妥当。
誇張せず実証範囲を正確に書けば、アルファ OSS として信頼を損なわない。最終判断はユーザー。
