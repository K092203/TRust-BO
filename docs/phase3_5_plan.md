# Phase 3.5 作業計画

> **歴史的記録**: Phase 3.5 完了時点(2026-06)の計画書。Phase 4(制約付き)以降、
> Phase G/H/K(実 CFD・多目的・rayon 並列化・ts デフォルト化等)まで完了済み。
> 現状は `CLAUDE.md` / `docs/ROADMAP.md` / `docs/DEVELOPMENT.md` を参照。

目的: Phase 3 の安定化・評価強化。Phase 4 (制約付き) に進む前の品質基盤を作る。
実装変更は最小限に抑え、診断・評価ツールの充実を優先する。

**ステータス: 全タスク完了 (2026-06)**

---

## タスク一覧 (優先順)

### P0 — 診断ツール

#### T1. surrogate_loss を実測値に変える ✅
`Ensemble::train()` の戻り値を `(Self, f32)` に変更。
各メンバーの最終 epoch MSE を平均して返す。`ProposeOutput.surrogate_loss` に反映。
`surrogate.rs:52-112`

#### T2. TR 状態を Python 側から観察できるようにする ✅
`TrustBoEngine.tr_state() -> dict | None` を追加。`engine.py:97-109`

---

### P1 — 評価強化

#### T3. seed 数を 10 に増やし median / IQR でレポート ✅
`tests/test_phase3_eval.py` 追加。`pytest -m eval` でのみ実行。
50D Ackley: TRM median=8.333 vs Random 9.573 (**+13%**)
50D Rosenbrock: TRM median=197,835 vs Random 375,308 (**+47%**)
`benchmarks/phase3_eval.csv` に出力。

#### T4. best-so-far curve のレポート生成 ✅
`TrustBoEngine.best_so_far_curve() -> list[float]` を追加。`engine.py:111-123`

#### T5. Ablation: TR dynamic vs TR frozen ✅
`tests/test_phase3_ablation.py` 追加。`pytest -m eval` でのみ実行。

**結果 (10D, budget=200, 5 seeds):**
| 条件 | Ackley median | Rosenbrock median | vs Random |
|---|---|---|---|
| TR dynamic (l=0.5, τ_fail=5) | 4.161 | 5,826 | **+42%, +43%** |
| TR frozen (l=1.0, no dynamics) | 5.242 | 7,421 | **+27%, +27.5%** |
| Random | 7.202 | 10,236 | — |

→ TR dynamics が **約+15%** の追加改善。

**注意**: 現行デフォルト (l_init=1.0=l_max, tau_fail=10) は TR frozen と同等動作。
性能を最大化するには `l_init=0.5, tau_fail=5` を推奨。

---

### P2 — Halton 列への切り替え (実装あり)

#### T6. candidate.rs に Halton 列を追加 ✅
計画では Sobol だったが、外部 crate 不要な **スクランブル Halton 列** を実装。
- 64D まで対応する素数テーブルを使用
- van der Corput 数列 + 乱数オフセット (additive scrambling)
- `cold_start_output` を `candidate::lhs()` → `candidate::halton()` に切り替え
- `candidate.rs:1-70` 参照

---

### P3 — rayon 並列化 (調査完了、見送り)

#### T7. rayon による ensemble 並列学習 → Sequential に戻した ✅
- rayon `into_par_iter()` を実装したが `B::seed()` が NdArray バックエンドの
  **グローバル状態** を操作するため、並列呼び出しで非決定論的になることが判明。
- `test_reproducibility_phase3` が連続実行で毎回異なる値を返すことで確認。
- 結論: 再現性を優先して Sequential に戻す。rayon の有効化には burn の
  thread-local RNG バックエンドへの切り替えが必要 (Phase 4 以降で検討)。

---

### P4 — EI の数値安定性改善

#### T8. EI の std=0 エッジケース処理 ✅
`acquisition.rs` の `ei()` に `if *s < 1e-6 { return 0.0; }` ガードを追加。
`acquisition.rs:8-16`

---

## 実装済み変更まとめ

| ファイル | 変更内容 |
|---|---|
| `src/surrogate.rs` | `train()` → `(Ensemble, f32)` 返り値変更 |
| `src/acquisition.rs` | EI std ガード追加 |
| `src/candidate.rs` | `halton()` 実装 (scrambled Halton) |
| `src/lib.rs` | cold_start を halton に切り替え, surrogate_loss 使用 |
| `python/trust_bo/engine.py` | `tr_state()`, `best_so_far_curve()` 追加 |
| `python/trust_bo/integrations/optuna.py` | TR 状態維持 + acquisition="ei" に修正 |
| `tests/test_phase3_eval.py` | 10-seed 評価テスト |
| `tests/test_phase3_ablation.py` | TR ablation テスト |

---

## Phase 3.5 exit criteria (全て達成)

- [x] `surrogate_loss` が実測値で記録され、学習収束が確認できる
- [x] 10 seeds × median(TRM) < median(Random) が 50D Ackley/Rosenbrock で成立
- [x] best-so-far curve が warm path で単調改善を示す
- [x] ablation で TR 有効性が定量確認できる (TR dynamic で +15% vs frozen)
- [x] Halton 切り替え後も全既存テスト (Phase 1/2/3, 13 tests) が通る
- [x] rayon 問題を特定・解決 (Sequential に戻し、再現性復元)
- [x] Optuna sampler の acquisition="ei" 修正 + TR 状態維持

---

## Phase 4 前提条件

Phase 3.5 完了により以下が確立された:

1. **信頼できる surrogate**: bootstrap EI + multi-start CEM が random を大幅に上回る
2. **診断ツール**: surrogate_loss, TR state, best-so-far curve が外部から観察可能
3. **再現性**: 全 13 テストが決定論的に通過
4. **評価基盤**: 10-seed median/IQR 評価と ablation 測定のフレームワーク

Phase 4 (制約付き最適化) への移行が可能。

---

## 参考: TR 設定ガイドライン

現行デフォルト (l_init=1.0=l_max) は TR dynamics を実質無効化している。
用途別の推奨設定:

| 用途 | l_init | l_max | tau_succ | tau_fail |
|---|---|---|---|---|
| 現行デフォルト (安全) | 1.0 | 1.0 | 3 | 10 |
| 推奨 (ablation 最適) | 0.5 | 1.0 | 3 | 5 |
| 高次元長期最適化 | 0.3 | 0.8 | 3 | 5 |
