#!/bin/bash
# Steps 1→4→5→6 を自律実行するスクリプト
set -e
cd /home/k0903/trm-engine

log() { echo "$(date '+%H:%M:%S') $*" | tee -a task_log.txt; }

# ── Step 1: ベンチマーク完了待ち ─────────────────────────────────────────
log "=== Step 1: tandem_results.csv 完了待ち ==="
while [ $(wc -l < tandem_results.csv 2>/dev/null || echo 0) -lt 61 ]; do
    sleep 30
done
log "Step 1 完了: $(wc -l < tandem_results.csv) rows"

# ── Step 2-3: テスト実行 ────────────────────────────────────────────────
log "=== Step 2-3: pytest tests/test_tandem.py ==="
.venv/bin/pytest tests/test_tandem.py -v --tb=short 2>&1 | tee /tmp/test_tandem_results.txt || true
log "テスト完了"

# ── Step 4: benchmark_tandem_v2.py 実行 ─────────────────────────────────
log "=== Step 4: benchmark_tandem_v2.py 開始 ==="
.venv/bin/python benchmark_tandem_v2.py > /tmp/bench_v2.log 2>&1 &
BENCH_V2_PID=$!
log "PID=$BENCH_V2_PID"

# 2時間ごとに進捗をログ
while kill -0 $BENCH_V2_PID 2>/dev/null; do
    sleep 7200
    ROWS=$(wc -l < tandem_v2_results.csv 2>/dev/null || echo 0)
    log "Step 4 進捗: $ROWS rows"
done
wait $BENCH_V2_PID || true
log "Step 4 完了: $(wc -l < tandem_v2_results.csv) rows"

# ── Step 5: CFD結果確認 ──────────────────────────────────────────────────
log "=== Step 5: CFD結果確認 ==="
.venv/bin/python - <<'EOF'
import csv, numpy as np
from pathlib import Path

def check_csv(path, val_col):
    rows = list(csv.DictReader(open(path))) if Path(path).exists() else []
    if not rows:
        print(f"  {path}: データなし")
        return
    try:
        vals = [float(r[val_col]) for r in rows if r.get(val_col)]
        print(f"  {path}: {len(rows)}行, best={max(vals):.3f}, mean={np.mean(vals):.3f}")
    except Exception as e:
        print(f"  {path}: エラー {e}")

check_csv("cfd/naca_results.csv", "Cl_Cd")
check_csv("cfd/f1wing_results.csv", "abs_Cl_Cd")
EOF

# ── Step 6: 最終レポート更新 ─────────────────────────────────────────────
log "=== Step 6: final_report.txt 更新 ==="
.venv/bin/python generate_final_report_v2.py
log "=== 全Step完了 ==="
