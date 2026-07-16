# TRust-BO

**[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md)**

**今手元にあるハードウェアだけで動くベイズ最適化。**

> **TRust-BOはCFDソルバーではありません。** 最適化ループの中で動作するツールであり、
> CFDソルバー自体を置き換えるものではなく、設計探索中に必要なCFD実行回数を減らします。

[![CI](https://github.com/K092203/TRust-BO/actions/workflows/ci.yml/badge.svg)](https://github.com/K092203/TRust-BO/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)]()
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange)]()

---

## なぜ作ったか

最良の設計に、最良のハードウェアは要らないはずだ。

CFD最適化は長らくHPCクラスタやワークステーション、組織的な予算に縛られてきた ——
それらは優れたツールだが、多くの人の手には届かない。
TRust-BOは、ある高校生が実際に抱えていた課題を解くために作られた:
Formula Studentカーの空力を、GPUなしのラップトップで最適化すること。
目標は、ベイズ最適化をどこでも動くほど高速に、博士号なしで使えるほどシンプルに、
そして実際の設計現場で意味を持つほど正確にすることだ。

TRust-BOは、**Rustで実装したトラスト領域ベイズ最適化エンジン**を、PyO3経由でPythonに公開したものだ。
GPU不要、クラウド不要。`pip install trust-bo` してローカルで実行できる。

**正確に何をするか:** 高次元(50D以上)、または ノイズあり・制約ありの問題(実CFDなど)において、
TRust-BOは**BoTorch TuRBO比で5〜10倍高速でありながら、同等以上の品質**を達成する。
低次元・滑らか・小予算の問題ではGPベースの手法(BoTorch, HEBO)の方が通常優れており、
TRust-BOは万能最良を主張しない。正直なデータに基づく評価は
[docs/PERFORMANCE_ASSESSMENT.md](docs/PERFORMANCE_ASSESSMENT.md)(英語)を参照。

> **ベンチマークの範囲(比較する前に読むこと):** 以下の結果は **BoTorch TuRBO, CMA-ES,
> Random Search, NSGA-II** との比較で測定したものだ。**SAASBO**(強力な高次元BOベースライン)
> との比較は**今後の課題**。CFDベンチマークは**3シード**、多目的は**2シード**であり、
> 統計的な厳密性のためにさらにシード数を増やす予定。

---

## 他手法との比較

| 特徴 | **TRust-BO** | BoTorch | HEBO | Optuna |
|---|:---:|:---:|:---:|:---:|
| GPU必須 | ✗ | 任意 | ✗ | ✗ |
| CPU最適化されたコア | ✓ (Rust) | △ | △ | ✓ |
| 高次元対応(50D+) | ✓ | ✓ | △ | △ |
| 最小限のask/tell API | ✓ | △ | △ | ✓ |
| CFDワークフロー特化 | ✓ | ✗ | ✗ | ✗ |

> BoTorchやOptunaもCFD駆動の最適化に使うことはできる。TRust-BOは学生や小規模な
> エンジニアリングチーム向けに、軽量・CPU優先のワークフローを念頭に設計されている点が特徴だ。

---

## インストール

```bash
pip install trust-bo
```

ビルド済みホイール(abi3, Python 3.9以上)をLinux/macOS/Windows向けに配布しており、
通常のインストールにはRustツールチェーンは不要。

ソースからビルドする場合([Rustツールチェーン](https://rustup.rs/)が必要):

```bash
git clone https://github.com/K092203/TRust-BO
cd TRust-BO
python -m venv .venv && source .venv/bin/activate
pip install .
```

開発時は[maturin](https://github.com/PyO3/maturin)を使うと再ビルドが速い:
`pip install maturin && maturin develop --release`。

---

## 使い方

```python
from trust_bo import TRustBOEngine, Float

# 1. 探索空間を定義
space = [Float(f"x{i}", -5.0, 5.0) for i in range(10)]

# 2. エンジンを作成
engine = TRustBOEngine(space=space, direction="minimize", seed=42)

# 3. ask → evaluate → tell
for _ in range(20):                          # 20ラウンド × batch_size=10
    candidates = engine.ask(batch_size=10)   # 次の候補点を提案
    results = [
        {"value": your_cfd_solver(c), "feasible": True}
        for c in candidates
    ]
    engine.tell(candidates, results)         # 結果をエンジンへ返す

# 4. 最良の結果を取得
print(engine.best())
# {'parameters': {'x0': 0.12, ...}, 'objective_values': [3.47]}
```

### 制約付き

```python
engine.tell(candidates, [
    {"value": solver(c), "feasible": constraint_ok(c)}
    for c in candidates
])
```

### 多目的最適化(パレート)

`MultiObjectiveEngine`で複数目的を同時に最適化する。`method="ehvi"`はRustで実装された
閉形式の2目的Expected Hypervolume Improvementを使い、`method="chebyshev"`はスカラー化
(任意の目的数に対応)を使う。

```python
from trust_bo import MultiObjectiveEngine, Float

space = [Float(f"x{i}", 0.0, 1.0) for i in range(20)]
engine = MultiObjectiveEngine(
    space=space, directions=["maximize", "minimize"],  # 例: Cl ↑, Cd ↓
    method="ehvi", seed=0,
)
for _ in range(15):
    cands = engine.ask(batch_size=4)
    results = [{"values": [cl(c), cd(c)], "feasible": True} for c in cands]
    engine.tell(cands, results)

print(engine.pareto_front())                 # 非劣解の設計群
print(engine.hypervolume(ref=[0.0, 0.05]))   # 品質指標
```

### 保存と再開

```python
engine.save("study.zip")
engine = TRustBOEngine.load("study.zip")
```

### Optunaサンプラーとして使う

TRust-BOを[Optuna](https://optuna.org/)のサンプラーとしてそのまま使える。`pip install optuna`が必要。

```python
import optuna
from trust_bo.integrations.optuna import TrustBoOptunaSampler

study = optuna.create_study(direction="minimize", sampler=TrustBoOptunaSampler(seed=42))

def objective(trial):
    x = [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(10)]
    return sum(v**2 for v in x)

study.optimize(objective, n_trials=100)
print(study.best_value)
```

---

## ベンチマーク

![TRust-BO vs BoTorch TuRBO on Ackley 100D: better optimum, 5.7× faster, CPU-only](docs/assets/benchmark_ackley100d.png)

詳細データと手法: [docs/BENCHMARK.md](docs/BENCHMARK.md)、
[docs/PERFORMANCE_ASSESSMENT.md](docs/PERFORMANCE_ASSESSMENT.md)(いずれも英語)。
比較対象は**BoTorch TuRBO, CMA-ES, Random Search, NSGA-II**。SAASBOとの比較は今後の課題。

### 合成高次元関数(得意分野)

Ackley / Rastrigin / Levy の50D・100D、予算100〜500、`enable_phase2=True`。
値は小さいほど良い。品質はシード間の中央値best、速度はラウンドあたりの実時間。

| 条件 | TRust-BO | BoTorch TuRBO | 品質 | 速度 |
|---|:---:|:---:|:---:|:---:|
| Ackley 50D, b=300 | **5.96** | 7.25 | TRust | **8.9×** |
| Levy 50D, b=300 | **152.4** | 194.5 | TRust | **10.7×** |
| Ackley 100D, b=300 | **7.13** | 8.51 | TRust | **5.7×** |
| Levy 100D, b=300 | **284.1** | 723.2 | TRust | **5.7×** |
| Ackley 50D, b=500 | **4.64** | 6.38 | TRust | — |

中予算条件**18件中16件**、budget=500では**5件中5件**でTRust-BOが勝ち、**5〜10倍高速**。
唯一の敗北はbudget=100のRastrigin(小予算ではGPが有利)。

### 実CFD — 翼型形状最適化

**H-1(NeuralFoil, 16D CST, Cl/Cd最大化, 10シード):** クリーンなサロゲート、良設定問題。

| 手法 | median Cl/Cd | best |
|---|:---:|:---:|
| BoTorch TuRBO | **241.4** | 245.4 |
| **TRust-BO+P2** | 227.9 | **267.4** |
| CMA-ES | 223.3 | 265.5 |
| Random | 148.7 | 161.1 |

16D・滑らかな問題では中央値でBoTorchが上回るが、TRust-BOは単独最良設計に到達している。

> 測定は当時のデフォルトだった`acquisition="ei"`で実施。現在のデフォルトは`"ts"`であり、
> 同じ滑らかなCST 16D Cl/Cd目的の別のA/Bでは`"ei"`に劣る
> ([docs/BENCHMARK.md §15](docs/BENCHMARK.md)参照、上表とはシード・予算が異なるため、
> この227.9という数値そのものの再現は期待できない) —
> CFD形状の目的関数には`config={"acquisition": "ei"}`を明示することを推奨。

**H-2(SU2 RANS, 実ナビエ・ストークス方程式, 16D, 3シード):** ノイズあり・メッシュ制約あり。

| 手法 | median Cl/Cd | 物理的に妥当な範囲のシード |
|---|:---:|:---:|
| **TRust-BO+P2** | **171.6** | **3 / 3** |
| BoTorch TuRBO | 126.1 | 3 / 3 |
| CMA-ES | 774.6 ⚠ | 1 / 3 |
| Random | 316.1 ⚠ | 1 / 3 |

より難しくノイジーなRANS問題ではTRust-BOが優位(+36%)で、最も安定している。

> ⚠ **SU2(H-2)の例についての注記:** feasibility判定は現状シンプル(`Cd > 0` +
> メッシュ有効性)。極薄形状がすり抜けて非物理的なCl/Cd(上表の⚠値)を出すことがある。
> 幾何制約(最小厚み/面積)を予定している。
> **クリーンで即使えるCFD例としてはNeuralFoil(H-1)パイプラインを推奨。**

### 多目的(Cl↑とCd↓を同時に)

Rustで実装した閉形式2目的EHVI(Expected Hypervolume Improvement)とChebyshevスカラー化を
SU2上で比較(budget=60、**2シード**): EHVIのmedian hypervolumeは**0.0239 vs 0.0165(+45%)**。
Chebyshevの方がパレートフロントの多様性は高い。詳細は[docs/BENCHMARK.md](docs/BENCHMARK.md)。

### ネイティブPhase 2

純Rust実装のMatern 5/2マイクロGPがMLPアンサンブルの*残差*を最良点近傍でフィットし、
終盤の精密化を行う。フラグ一つ、sklearn不要、追加依存なし:

```python
engine = TRustBOEngine(space=space, config={"enable_phase2": True})
```

> **TRust-BOを使うべき場面:** 高次元(50D+)、またはノイズあり/制約ありの問題、
> 中〜大予算(100〜1000)、そしてCFDのように1評価が既に数分〜数時間かかり
> BOラウンドごとの実時間が重要になる場面。

---

## 動作原理

TRust-BOは**TuRBO**(トラスト領域ベイズ最適化)を、独自サロゲートで実装している:

```
コールドスタート → Halton準乱数サンプリング(n_init点)
ウォームパス     → MLPブートストラップ・アンサンブルサロゲート(5メンバー)
                 + トラスト領域内でのCross-Entropy Method(CEM)
                 + トラスト領域の動的更新(成功で拡大、失敗で縮小)
```

主な設計判断:
- **Rustコア** — 内側ループ(サロゲート学習、CEM、TR管理)はPyO3経由でコンパイル済みRustが実行し、CPU負荷を低く抑える
- **ウォームスタート** — サロゲートの重みをラウンド間でシリアライズし、学習時間を約41%削減
- **中立な中心の安定性** — TR中心は相対1%以上の改善がない限り動かず、小さな揺らぎによる不安定化を防ぐ
- **GPU不要** — `burn`の`ndarray`バックエンドを使用、最近のノートPC用CPUで十分

---

## ロードマップ

### 現状(v0.3.0)
- [x] 単一トラスト領域(exploitation重視) + TuRBO-M マルチTR(実験的)
- [x] ウォームスタート付きMLPブートストラップ・アンサンブルサロゲート
- [x] 制約処理(feasibilityサロゲート) + CFD向けfail-fast幾何形状制約(最小厚み/面積、メッシュ生成前)
- [x] 128テストスイート(Python 84 + Rust 44)、CPUのみ、PyO3 Pythonバインディング
- [x] BoTorch TuRBO / CMA-ES / HEBO / Random / NSGA-II との比較ベンチマーク
- [x] ネイティブPhase 2(Rust Tandem Residual-GP): 50Dで+32%、10Dで+55%、追加依存ゼロ
- [x] 高価なソルバー向けの非同期並列/rolling評価(SLURM対応)
- [x] マルチフィデリティ・カスケード(LF→HF、NeuralFoilで高忠実度評価70%削減)
- [x] 実CFD翼型最適化 — NeuralFoil(H-1)**および**SU2 RANS(H-2)パイプライン
- [x] 多目的最適化: Chebyshevスカラー化 + 閉形式2目的EHVI(Rust)
- [x] Optunaサンプラー統合
- [x] PyPI公開(Linux/macOS/Windows向けビルド済みabi3ホイール)

### 予定
- [ ] SAASBOとの比較 + ベンチマークシード数の追加(統計的厳密性)
- [ ] 2目的を超える多目的最適化
- [ ] マルチTR(TuRBO-M) — 優先度は下げている。CFD規模の予算では単一TRの方が安定
- [ ] 軽量エンジニアリング設計向けBOに関するリサーチ・レポート

---

## FAQ

### TRust-BOはCFDソルバーですか?
いいえ。TRust-BOは外部のCFDソルバーと**一緒に**動く最適化エンジンです。
ナビエ・ストークス方程式は解きません。次にどの設計候補を評価すべきかを決定し、
高価なCFD実行の総回数を減らします。

### なぜBoTorchを使わないのですか?
BoTorchは優れており、はるかに柔軟です。TRust-BOはその代替ではなく、
学生や小規模チーム向けの軽量CFD駆動ワークフローを念頭に置いた、
シンプルなask/tell APIを持つ、より小さなCPU優先エンジンです。

### 実CFDでの検証実績はありますか?
はい、翼型形状最適化で検証済みです。**NeuralFoil**(H-1、高速な学習済み空力サロゲート)と
**SU2 RANS**(H-2、実の定常ナビエ・ストークス方程式、Ma=0.3、Re=3×10⁶、SA乱流モデル)
という2つのパイプラインを同梱しています。ベンチマークの節と
[docs/BENCHMARK.md](docs/BENCHMARK.md)を参照してください。
現時点の検証はSU2で3シードのみ — さらなるシード数とSAASBO比較を予定しています。

### 誰のためのツールですか?
学生、Formula Studentチーム、そしてHPCやGPUなしでCFD駆動の設計最適化に
取り組みたい小規模エンジニアリングチーム。

### BoTorch / HEBO / Optuna との関係は?
いずれも優れた最適化エンジンであり、TRust-BOはそれらの置き換えを目指していません。
CPUのみ・軽量で、CFD駆動のエンジニアリング設計ワークフロー向けにパッケージ化された
最適化エンジンという特定のギャップに焦点を当てています。

---

## 既知の限界

- **GP比でのサロゲート精度:** MLPブートストラップ・アンサンブルは不確実性の較正精度を速度と引き換えにしている。低次元・滑らか・小予算の問題ではGPベースの手法(BoTorch, HEBO)が通常上回る(16D NeuralFoilベンチマークで確認済み)。TRust-BOの優位性は50D以上、またはノイズあり/制約ありの問題にある。
- **ベンチマークシード数:** 合成関数の結果は10シードだが、実CFD(SU2)は3シード、多目的は2シードと統計的には薄い。シード数の追加を予定している。
- **SAASBOとの比較は未実施:** 強力な高次元BOベースラインであるSAASBOとのベンチマークは(環境上の制約で)未実施。主張はBoTorch TuRBO / CMA-ES / Random / NSGA-IIとの比較に限定される。
- **CFDのfeasibility判定には既知のギャップがある:** SU2(H-2)パイプラインは現在、メッシュ生成前に不正な形状(最小厚み/面積、自己交差)をfail-fastで弾き、閾値未満の非物理的な`Cd`を事後的に棄却するようになっており、以前見られた主要なアーティファクトの原因は塞いだ([docs/BENCHMARK.md](docs/BENCHMARK.md) §20.3参照)。より粗い3Dフロントウィングメッシュ(下記参照)には残存リスクがある。
- **多目的最適化はEHVIについて2目的まで:** 閉形式EHVIは2目的限定。3目的以上ではChebyshevスカラー化を使うこと。
- **ウォームスタートの重み転送:** サロゲートの重みは16進文字列(ラウンドあたり約1MB)としてシリアライズされる。機能するが非効率であり、バイナリ転送機構を予定している。
- **マルチTR(`n_trs > 1`)は実験的:** 実装・テスト済みだが、CFD規模の予算では単一TRの方が安定するため優先度を下げている。

---

## 開発ログ

設計判断・フェーズごとの実験・ベンチマークの記録を含む完全な開発履歴は
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)(日本語)にまとめている。

---

## Contributing

貢献を歓迎する。現状は個人開発プロジェクトであり、バグ報告・実問題でのベンチマーク結果・
ドキュメント・コードなど、どんな形の協力も心から歓迎する。

TRust-BOを実際のCFD問題に使って結果(良い結果でも悪い結果でも)を得た場合は、
ぜひissueを開いて共有してほしい。現段階では実世界からのフィードバックが最も価値がある。

---

## License

MIT © 2026 Kotaro Ozawa
