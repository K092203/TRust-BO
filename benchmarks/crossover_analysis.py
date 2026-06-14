"""
crossover_analysis.py — 3 つの CSV を統合してクロスオーバーポイントを特定する。

入力 CSV:
  cfd_scale_results.csv   : budget=20/30/50,   50D のみ,       n_init=10, 0.1s遅延
  midbudget_results.csv   : budget=100/200/300, 50D・100D,      n_init=10, 遅延なし
  large_budget_results.csv: budget=500,         50/100/200D,    n_init=50, 遅延なし

出力:
  crossover_summary.csv   : 全条件の median/mean/std + n_init フラグ

クロスオーバー: 各 (problem, dim) で TRust-BO+P2 が BoTorch_TuRBO を
               初めて下回る budget を表示する。
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent  # trm-engine/

# ── CSV 読み込みと正規化 ───────────────────────────────────────────────────────

def load_rows(path: Path, n_init: int) -> list[dict]:
    """各 CSV を共通スキーマに正規化して返す。
    共通キー: problem, dim, method, budget, seed, best_value, n_init
    """
    rows = []
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return rows

    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                val = float(r["best_value"])
            except ValueError:
                continue  # "error" 行をスキップ

            # cfd_scale は problem 列に "Ackley_50D" 形式（dim 埋め込み）
            problem_raw = r["problem"]
            if "dim" in r:
                dim = int(r["dim"])
                problem = problem_raw
            else:
                # "Ackley_50D" → problem="Ackley", dim=50
                m = re.match(r"^(.+?)_(\d+)D$", problem_raw)
                if m:
                    problem, dim = m.group(1), int(m.group(2))
                else:
                    problem, dim = problem_raw, 0

            rows.append({
                "problem": problem,
                "dim":     dim,
                "method":  r["method"],
                "budget":  int(r["budget"]),
                "seed":    int(r["seed"]),
                "best":    val,
                "n_init":  n_init,
            })
    print(f"  {path.name}: {len(rows)} rows loaded")
    return rows


# ── 集計 ──────────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    """(problem, dim, method, budget) → stats dict"""
    g = defaultdict(list)
    ni = {}
    for r in rows:
        key = (r["problem"], r["dim"], r["method"], r["budget"])
        g[key].append(r["best"])
        ni[key] = r["n_init"]

    result = {}
    for key, vals in g.items():
        arr = np.array(vals)
        result[key] = {
            "median": float(np.median(arr)),
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "n":      len(arr),
            "n_init": ni[key],
        }
    return result


# ── crossover_summary.csv 出力 ────────────────────────────────────────────────

def write_summary_csv(agg: dict, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "dim", "method", "budget",
                    "n_init", "median_best", "mean_best", "std_best", "n_seeds"])
        for (problem, dim, method, budget), s in sorted(agg.items()):
            w.writerow([problem, dim, method, budget,
                        s["n_init"], f"{s['median']:.6f}",
                        f"{s['mean']:.6f}", f"{s['std']:.6f}", s["n"]])
    print(f"\ncrossover_summary.csv saved to: {out_path.resolve()}")


# ── クロスオーバー特定 ────────────────────────────────────────────────────────

def find_crossovers(agg: dict):
    problems = sorted({k[0] for k in agg})
    dims     = sorted({k[1] for k in agg})
    budgets  = sorted({k[3] for k in agg})

    print("\n" + "=" * 60)
    print("  クロスオーバー分析: TRust-BO+P2 が BoTorch を初めて下回る budget")
    print("=" * 60)

    for problem in problems:
        for dim in dims:
            trust_data  = {b: agg[(problem, dim, "TRust-BO+P2",   b)]["median"]
                           for b in budgets if (problem, dim, "TRust-BO+P2",   b) in agg}
            botorch_data = {b: agg[(problem, dim, "BoTorch_TuRBO", b)]["median"]
                            for b in budgets if (problem, dim, "BoTorch_TuRBO", b) in agg}

            if not trust_data or not botorch_data:
                continue

            common = sorted(set(trust_data) & set(botorch_data))
            crossover = None
            for b in common:
                if trust_data[b] < botorch_data[b]:
                    crossover = b
                    break

            print(f"\n{problem} {dim}D:")
            print(f"  {'budget':>7}  {'TRust-BO+P2':>13}  {'BoTorch':>10}  {'勝者'}")
            for b in common:
                t = trust_data.get(b)
                bo = botorch_data.get(b)
                if t is None or bo is None:
                    continue
                winner = "TRust-BO ✓" if t < bo else "BoTorch"
                marker = " ← crossover" if b == crossover else ""
                print(f"  {b:>7}  {t:>13.4f}  {bo:>10.4f}  {winner}{marker}")

            if crossover:
                print(f"  → クロスオーバー: budget = {crossover}")
            else:
                avail = common
                if avail:
                    last = avail[-1]
                    if trust_data.get(last, float("inf")) < botorch_data.get(last, float("inf")):
                        print(f"  → budget={last} 時点で TRust-BO が既に優位（下限不明）")
                    else:
                        print(f"  → budget={last} 時点でまだ BoTorch が優位（上限不明）")

    print("\n" + "=" * 60)


# ── サマリ表示 ────────────────────────────────────────────────────────────────

def print_full_table(agg: dict):
    problems = sorted({k[0] for k in agg})
    dims     = sorted({k[1] for k in agg})
    budgets  = sorted({k[3] for k in agg})
    methods  = ["TRust-BO+P2", "BoTorch_TuRBO", "HEBO", "Random"]

    print("\n全条件サマリ（median, lower is better）")
    print("-" * 80)
    hdr = f"{'Problem':<11} {'dim':>4} {'budget':>7}"
    for m in methods:
        hdr += f"  {m[:12]:>12}"
    print(hdr)
    print("-" * 80)

    for problem in problems:
        for dim in dims:
            for budget in budgets:
                row = f"{problem:<11} {dim:>4} {budget:>7}"
                any_data = False
                for m in methods:
                    s = agg.get((problem, dim, m, budget))
                    if s:
                        row += f"  {s['median']:>12.4f}"
                        any_data = True
                    else:
                        row += f"  {'—':>12}"
                if any_data:
                    print(row)
            print()
    print("-" * 80)


# ── メイン ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CSV 読み込み中...")
    rows = []
    rows += load_rows(BASE / "cfd_scale_results.csv",    n_init=10)
    rows += load_rows(BASE / "midbudget_results.csv",    n_init=10)
    rows += load_rows(BASE / "large_budget_results.csv", n_init=50)

    print(f"総行数: {len(rows)}")

    agg = aggregate(rows)
    write_summary_csv(agg, BASE / "crossover_summary.csv")
    find_crossovers(agg)
    print_full_table(agg)
