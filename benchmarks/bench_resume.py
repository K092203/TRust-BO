"""bench_resume.py — ベンチ共通の resume（チェックポイント）ヘルパ。

長時間ベンチは WSL の OOM・自動シャットダウンや手動 kill で途中停止しうる。
CSV を逐次追記しておき、再実行時に **完了済みの組合せ(key)をスキップ** することで、
同じコマンドの再実行だけで続きから再開できるようにする（ユーザー操作不要）。

使い方:
    from bench_resume import resume_or_init, is_done
    done = resume_or_init(CSV_PATH, ("method", "problem", "budget", "seed"), csv_write_header)
    ...
    if is_done(done, method, problem, budget, seed):
        print("[skip]"); continue
    ...  # 実行 → csv_append
"""

from __future__ import annotations

import csv
import os


def load_done(csv_path, key_cols) -> set[tuple[str, ...]]:
    """CSV から完了済み key の集合を返す。無ければ空集合。key は全て str に正規化。"""
    done: set[tuple[str, ...]] = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add(tuple(str(r[c]) for c in key_cols))
    except Exception:
        pass
    return done


def resume_or_init(csv_path, key_cols, write_header) -> set[tuple[str, ...]]:
    """CSV があれば完了 key の集合を返す。無ければ write_header() を呼び空集合を返す。"""
    if os.path.exists(csv_path):
        done = load_done(csv_path, key_cols)
        print(f"  [resume] CSV found — {len(done)} run(s) already done, skipping")
        return done
    write_header()
    return set()


def is_done(done: set[tuple[str, ...]], *key_values) -> bool:
    """key_values（任意型）を str 化して done に含まれるか判定。"""
    return tuple(str(v) for v in key_values) in done
