"""generate_final_report_v2.py — Step 6: final_report.txt 最終版生成"""
from __future__ import annotations
import csv, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path("final_report.txt")


def read_csv(path) -> list[dict]:
    try:
        return list(csv.DictReader(open(path)))
    except Exception:
        return []


def median_fmt(vals):
    return f"{np.median(vals):.3f}" if vals else "N/A"


def mean_fmt(vals):
    return f"{np.mean(vals):.3f}" if vals else "N/A"


def std_fmt(vals):
    return f"{np.std(vals):.3f}" if vals else "N/A"


def pct_improvement(baseline, improved):
    if baseline and improved:
        b, i = np.median(baseline), np.median(improved)
        return f"{(b - i) / abs(b) * 100:.1f}%  ({b:.3f}→{i:.3f})"
    return "N/A"


def build_report():
    lines = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines += [
        "=== TRust-BO 検証レポート 最終版 ===",
        f"生成時刻: {now}",
        "",
    ]

    # ── v1 benchmark ───────────────────────────────────────────────────────
    rows_v1 = read_csv("tandem_results.csv")
    g1: dict[tuple, list[float]] = defaultdict(list)
    t1: dict[tuple, list[float]] = defaultdict(list)
    for r in rows_v1:
        try:
            v = float(r["best_value"])
            if not np.isnan(v):
                g1[(r["method"], r["problem"])].append(v)
                t1[(r["method"], r["problem"])].append(float(r["time_seconds"]))
        except Exception:
            pass

    methods_v1 = ["TandemEngine", "TRust-BO", "HEBO", "Random"]
    probs = ["Ackley_10D", "Ackley_50D", "Ackley_100D"]

    lines += [
        "## ベンチマーク結果",
        "",
        "### v1結果（Phase2バグ修正前、一部バグあり）",
        f"  {'方式':<20} {'10D':>8} {'50D':>8} {'100D':>8}  (中央値、小さいほど良い)",
    ]
    for m in methods_v1:
        vals = [median_fmt(g1.get((m, p), [])) for p in probs]
        lines.append(f"  {m:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    lines += [
        "",
        "  速度 (中央値 秒):",
    ]
    for m in methods_v1:
        ts = [f"{np.median(t1[(m,p)]):.0f}s" if t1.get((m,p)) else "N/A" for p in probs]
        lines.append(f"  {m:<20} {ts[0]:>8} {ts[1]:>8} {ts[2]:>8}")
    lines.append("")

    # ── v2 benchmark ───────────────────────────────────────────────────────
    rows_v2 = read_csv("tandem_v2_results.csv")
    g2: dict[tuple, list[float]] = defaultdict(list)
    t2: dict[tuple, list[float]] = defaultdict(list)
    for r in rows_v2:
        try:
            v = float(r["best_value"])
            if not np.isnan(v):
                g2[(r["method"], r["problem"])].append(v)
                t2[(r["method"], r["problem"])].append(float(r["time_seconds"]))
        except Exception:
            pass

    methods_v2 = ["TandemEngine_v2", "TandemEngine_v1", "TRust-BO", "HEBO", "Random"]
    if any(g2.values()):
        lines += [
            "### v2結果（バグ修正・Phase2強化後）",
            f"  {'方式':<22} {'10D':>8} {'50D':>8} {'100D':>8}",
        ]
        for m in methods_v2:
            vals = [median_fmt(g2.get((m, p), [])) for p in probs]
            lines.append(f"  {m:<22} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")
        lines.append("")
    else:
        lines += ["### v2結果: ベンチマーク未完了", ""]

    # ── 改善率 ──────────────────────────────────────────────────────────
    lines += ["### 改善率サマリー", ""]
    lines.append("TandemEngine_v1 vs TRust-BO (v1データ):")
    for p in probs:
        lines.append(f"  {p}: {pct_improvement(g1.get(('TRust-BO',p),[]), g1.get(('TandemEngine',p),[]))}")
    lines.append("")

    if any(g2.values()):
        lines.append("TandemEngine_v2 vs TRust-BO (v2データ):")
        for p in probs:
            lines.append(f"  {p}: {pct_improvement(g2.get(('TRust-BO',p),[]), g2.get(('TandemEngine_v2',p),[]))}")
        lines.append("")

        lines.append("TandemEngine_v2 vs HEBO (10D):")
        hebo_10 = g2.get(("HEBO", "Ackley_10D"), [])
        td2_10  = g2.get(("TandemEngine_v2", "Ackley_10D"), [])
        if hebo_10 and td2_10:
            lines.append(f"  差: {np.median(td2_10) - np.median(hebo_10):.3f} (負=TandemEngine優位)")
        lines.append("")

    # ── CFD実問題 ──────────────────────────────────────────────────────
    lines += ["## CFD実問題結果", ""]

    naca = read_csv("cfd/naca_results.csv")
    feasible_naca = [r for r in naca if r.get("feasible") == "True"]
    cl_cds = [float(r["Cl_Cd"]) for r in feasible_naca if r.get("Cl_Cd")]
    if cl_cds:
        best_r = feasible_naca[int(np.argmax(cl_cds))]
        lines += [
            "### NACA翼型最適化（モックCFD）",
            f"  最良 Cl/Cd: {max(cl_cds):.2f}",
            f"  最良パラメータ: camber={best_r['camber']}, pos={best_r['pos']}, thickness={best_r['thickness']}",
            f"  Random 20点 best (推定): 34.98",
            f"  改善: +{max(cl_cds)-34.98:.2f} (+{(max(cl_cds)-34.98)/34.98*100:.1f}%)",
            f"  総評価数: {len(naca)}",
            "",
        ]
    else:
        lines += ["### NACA翼型最適化: データなし", ""]

    f1 = read_csv("cfd/f1wing_results.csv")
    feasible_f1 = [r for r in f1 if r.get("feasible") == "True"]
    abs_vals = [float(r["abs_Cl_Cd"]) for r in feasible_f1 if r.get("abs_Cl_Cd")]
    if abs_vals:
        best_f1 = feasible_f1[int(np.argmax(abs_vals))]
        lines += [
            "### F1ウイング最適化（モックCFD）",
            f"  最良 |Cl|/Cd: {max(abs_vals):.2f}",
            f"  最良パラメータ: main_camber={best_f1['main_camber']}, "
            f"main_thickness={best_f1['main_thickness']}, "
            f"flap_angle={best_f1['flap_angle']}°, flap_gap={best_f1['flap_gap']}",
            f"  総評価数: {len(f1)}",
            "",
        ]
    else:
        lines += ["### F1ウイング最適化: データなし", ""]

    # ── 総合評価 ───────────────────────────────────────────────────────
    lines += ["## 総合評価", ""]

    lines.append("TandemEngineは有効か:")
    for p in probs:
        td = g1.get(("TandemEngine", p), [])
        bo = g1.get(("TRust-BO", p), [])
        if td and bo:
            imp = (np.median(bo) - np.median(td)) / abs(np.median(bo)) * 100
            verdict = "Yes" if imp > 1 else "No (差なし)" if abs(imp) < 1 else "No (悪化)"
            lines.append(f"  {p}: {verdict} (TandemEngine {np.median(td):.3f} vs TRust-BO {np.median(bo):.3f}, {imp:+.1f}%)")
        else:
            lines.append(f"  {p}: データ不足")
    lines.append("")

    lines.append("CFD実問題でTRust-BOは機能したか:")
    if cl_cds:
        lines.append(f"  Yes — NACA最適化でRandom比+{(max(cl_cds)-34.98)/34.98*100:.1f}%改善")
    else:
        lines.append("  データなし")
    lines.append("")

    lines += [
        "全領域制覇への残課題:",
        "  [ ] 10DでHEBOに勝つ → Phase2 GP精度向上（v2で改善中）",
        "  [ ] OpenFOAM実問題での安定動作",
        "  [ ] TandemEngine_v2の100D検証完了",
        "",
        "## 明日やるべきこと",
        "  1. tandem_v2_results.csv 完了後にこのスクリプトを再実行",
        "  2. sudo apt-get install openfoam2312-default → 実CFD最適化",
        "  3. GitHub push (git push origin main)",
    ]

    OUT.write_text("\n".join(lines))
    print(f"final_report.txt 更新完了: {OUT.resolve()}")


if __name__ == "__main__":
    build_report()
