"""
rolling_engine.py — 並列非同期評価ハーネス。

HPC（SLURM/PBS）や並列 CFD ワークフロー向けに
max_concurrent ジョブを常時並走させ、完了次第 tell → ask の
ローリングウィンドウを実現する。

基本的な使い方:
    class MyCFDEvaluator(JobEvaluator):
        def submit(self, candidate: dict) -> str:
            # 形状ファイル書き出し → sbatch 投入 → job_id 返す
            ...
        def poll(self) -> list[tuple[str, float, bool]]:
            # squeue 確認 → 完了分を (job_id, value, feasible) で返す
            ...

    engine = RollingTRustBOEngine(
        base_engine=TRustBOEngine(space, direction="maximize", seed=42),
        evaluator=MyCFDEvaluator(),
        max_concurrent=8,
        poll_interval=30.0,
    )
    best = engine.run(budget=200)
    print(best)
"""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any


# ── 抽象基底クラス ─────────────────────────────────────────────────────────────

class JobEvaluator(ABC):
    """
    HPC ジョブキューへのアダプタ基底クラス。

    ユーザーは submit() と poll() を実装することで、
    任意のジョブスケジューラ（SLURM/PBS/LSF/ローカルプロセス）に接続できる。
    """

    @abstractmethod
    def submit(self, candidate: dict) -> str:
        """
        候補点 1 件をジョブとして投入する。

        Parameters
        ----------
        candidate : パラメータ名 → 値 の dict（engine.ask() の戻り値の 1 要素）

        Returns
        -------
        job_id : ジョブを一意に識別する文字列
        """

    @abstractmethod
    def poll(self) -> list[tuple[str, float, bool]]:
        """
        完了済みジョブを非ブロッキングで返す。

        Returns
        -------
        list of (job_id, value, feasible)
            job_id   : submit() が返したジョブ ID
            value    : 目的関数値（base_engine の direction のまま生の値を渡す）
            feasible : 制約を満たすかどうか（制約なし問題では常に True）

        注意: このメソッドはブロックしてはいけない。
              完了がなければ空リストを返すこと。
        """


# ── ローリングウィンドウエンジン ────────────────────────────────────────────────

class RollingTRustBOEngine:
    """
    TRustBOEngine を非同期並列評価で動かすラッパー。

    max_concurrent スロットを常時埋め続け、
    1 件完了 → tell → ask(1) → 即投入 のサイクルを回す。

    Parameters
    ----------
    base_engine    : TRustBOEngine インスタンス
    evaluator      : JobEvaluator サブクラスのインスタンス
    max_concurrent : 同時実行ジョブ数（HPC のスロット数に合わせる）
    poll_interval  : poll() の呼び出し間隔 [秒]
    job_timeout    : この秒数以内に完了しないジョブを失敗扱いにする (None = 無制限)
    verbose        : 進捗を標準出力に表示するか
    """

    def __init__(
        self,
        base_engine: Any,
        evaluator: JobEvaluator,
        max_concurrent: int = 4,
        poll_interval: float = 5.0,
        job_timeout: float | None = None,
        verbose: bool = True,
    ) -> None:
        self._engine = base_engine
        self._evaluator = evaluator
        self._max_concurrent = max_concurrent
        self._poll_interval = poll_interval
        self._job_timeout = job_timeout
        self._verbose = verbose
        # job_id -> (candidate, submit_time)
        self._pending: dict[str, tuple[dict, float]] = {}

    # --- public API ---

    def run(self, budget: int) -> dict:
        """
        budget 件の評価が完了するまでローリング実行する。

        Returns
        -------
        best : base_engine.best() の結果 dict（評価がゼロの場合は {}）
        """
        evaluated = 0

        # 初期スロットを埋める
        for _ in range(min(self._max_concurrent, budget)):
            self._submit_next()

        while evaluated < budget:
            newly_done = self._collect(budget - evaluated)

            if newly_done == 0:
                time.sleep(self._poll_interval)
                continue

            evaluated += newly_done

            # 空いたスロットを再充填（残り budget を超えないよう制限）
            can_submit = min(
                self._max_concurrent - len(self._pending),
                budget - evaluated - len(self._pending),
            )
            for _ in range(max(0, can_submit)):
                self._submit_next()

        return self._engine.best() or {}

    def base_engine(self) -> Any:
        """内部の TRustBOEngine を返す。履歴・best() へのアクセスに使う。"""
        return self._engine

    # --- internal ---

    def _submit_next(self) -> None:
        try:
            cand = self._engine.ask(batch_size=1)[0]
            jid = self._evaluator.submit(cand)
            self._pending[jid] = (cand, time.monotonic())
        except Exception as e:
            if self._verbose:
                print(f"  [warn] submit failed: {e}")

    def _collect(self, max_collect: int) -> int:
        """poll() を呼び完了分を tell する。タイムアウトも処理する。"""
        # タイムアウト済みジョブを除去し、budget カウントに加算する
        timed_out_count = 0
        if self._job_timeout is not None:
            now = time.monotonic()
            timed_out = [
                jid for jid, (_, t) in self._pending.items()
                if now - t > self._job_timeout
            ]
            for jid in timed_out:
                self._pending.pop(jid)
                timed_out_count += 1
                if self._verbose:
                    print(f"  [warn] job {jid} timed out, dropping")

        try:
            completed = self._evaluator.poll()
        except Exception as e:
            if self._verbose:
                print(f"  [warn] poll() raised: {e}")
            return 0

        done = 0
        for jid, value, feasible in completed:
            if jid not in self._pending:
                continue  # 二重通知またはタイムアウト済みを無視
            cand, _ = self._pending.pop(jid)
            try:
                self._engine.tell([cand], [{"value": value, "feasible": feasible}])
            except Exception as e:
                if self._verbose:
                    print(f"  [warn] tell() failed for {jid}: {e}")
                continue
            done += 1
            if self._verbose:
                best = self._engine.best()
                bv = best["objective_values"][0] if best else float("nan")
                print(f"  [done] {jid}  value={value:.4f}  best={bv:.4f}")
            if done >= max_collect:
                break

        return done + timed_out_count


# ── Mock 評価器（テスト・デモ用）─────────────────────────────────────────────

class MockEvaluator(JobEvaluator):
    """
    スレッドで非同期に目的関数を評価するテスト用評価器。

    実行時間をランダムにばらつかせることで、HPC の実ジョブを模擬する。

    Parameters
    ----------
    fn           : candidate (dict) → float の評価関数
    min_delay    : 最小実行時間 [秒]
    max_delay    : 最大実行時間 [秒]
    failure_rate : 0〜1 でジョブが完了せず消える確率（障害シミュレーション）
    seed         : 遅延・失敗のランダムシード
    """

    def __init__(
        self,
        fn: callable,
        min_delay: float = 0.05,
        max_delay: float = 0.5,
        failure_rate: float = 0.0,
        seed: int | None = 0,
    ) -> None:
        self._fn = fn
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._failure_rate = failure_rate
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._completed: deque[tuple[str, float, bool]] = deque()
        self._counter = 0

    def submit(self, candidate: dict) -> str:
        jid = f"mock-{self._counter:05d}"
        self._counter += 1
        delay = self._rng.uniform(self._min_delay, self._max_delay)
        fail = self._rng.random() < self._failure_rate

        def _worker():
            time.sleep(delay)
            if fail:
                return  # ジョブが消える（job_timeout テスト用）
            try:
                val = float(self._fn(candidate))
                with self._lock:
                    self._completed.append((jid, val, True))
            except Exception:
                with self._lock:
                    self._completed.append((jid, float("inf"), False))

        threading.Thread(target=_worker, daemon=True).start()
        return jid

    def poll(self) -> list[tuple[str, float, bool]]:
        with self._lock:
            results = list(self._completed)
            self._completed.clear()
        return results


# ── SLURM アダプタテンプレート ────────────────────────────────────────────────

class SlurmEvaluator(JobEvaluator):
    """
    SLURM ジョブキュー向けアダプタのテンプレート。

    実際の CFD ワークフローに合わせて以下の 3 メソッドを実装すること:
        _write_input(candidate, workdir)        : 入力ファイル書き出し
        _sbatch_args(workdir) -> list[str]      : sbatch に渡す引数リスト
        _read_result(workdir) -> (float, bool)  : 結果ファイルから値を読む

    使用例:
        class SU2Evaluator(SlurmEvaluator):
            def _write_input(self, candidate, workdir):
                write_cst_coords(candidate, workdir / "airfoil.dat")
                generate_mesh(workdir / "airfoil.dat", workdir / "mesh.su2")
                shutil.copy("su2_template.cfg", workdir / "run.cfg")

            def _sbatch_args(self, workdir):
                return ["--nodes=1", "--ntasks=8", f"--chdir={workdir}",
                        "--wrap=SU2_CFD run.cfg"]

            def _read_result(self, workdir):
                cl, cd = parse_su2_history(workdir / "history.csv")
                return cl / cd, True  # 最大化
    """

    def __init__(self, workdir_root: str = "slurm_jobs") -> None:
        import os
        self._root = workdir_root
        self._running: dict[str, str] = {}  # slurm_job_id -> workdir
        os.makedirs(workdir_root, exist_ok=True)

    def submit(self, candidate: dict) -> str:
        import os, subprocess, uuid
        workdir = os.path.join(self._root, str(uuid.uuid4()))
        os.makedirs(workdir)
        self._write_input(candidate, workdir)
        args = ["sbatch", "--parsable"] + self._sbatch_args(workdir)
        slurm_id = subprocess.check_output(args).decode().strip()
        self._running[slurm_id] = workdir
        return slurm_id

    def poll(self) -> list[tuple[str, float, bool]]:
        import subprocess
        completed = []
        still_running = {}
        for jid, workdir in list(self._running.items()):
            # squeue が出力を返さない = ジョブが存在しない = 完了
            out = subprocess.run(
                ["squeue", "-j", jid, "-h"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out:
                still_running[jid] = workdir
            else:
                try:
                    value, feasible = self._read_result(workdir)
                    completed.append((jid, value, feasible))
                except Exception as e:
                    print(f"  [warn] read_result failed for {jid}: {e}")
                    completed.append((jid, float("inf"), False))
        self._running = still_running
        return completed

    def _write_input(self, candidate: dict, workdir: str) -> None:
        raise NotImplementedError("_write_input を実装してください")

    def _sbatch_args(self, workdir: str) -> list[str]:
        raise NotImplementedError("_sbatch_args を実装してください")

    def _read_result(self, workdir: str) -> tuple[float, bool]:
        raise NotImplementedError("_read_result を実装してください")
