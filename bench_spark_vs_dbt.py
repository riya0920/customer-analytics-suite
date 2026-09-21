"""Benchmark the DuckDB and Spark paths on identical inputs, honestly.

Both engines run the SAME staging + mart SQL (see src/pipeline_spark.py) over the
SAME raw frames (src.pipeline.raw_frames), and this script asserts their marts are
identical before reporting a single timing -- a speed number for a wrong answer is
worse than no number.

What it measures, and why each:
  * DuckDB in-process   -- the pure engine running the mart SQL. The fair,
                           apples-to-apples compute number.
  * Spark startup        -- cost of bringing up the JVM + SparkSession. Paid once
                           per job, and at this data scale it dominates everything.
  * Spark transform      -- the same SQL on Spark, cold (first run) and warm
                           (steady state on a live session), collected to pandas.
  * dbt build            -- the actual project pipeline step (subprocess): dbt
                           process start + DuckDB compute + writing marts.

Peak memory is the max resident set of the process tree (this process + its Java
children for Spark; the dbt subprocess tree for dbt), sampled every 40ms.

Run:  python bench_spark_vs_dbt.py     (needs a Java 17+ runtime for the Spark rows)
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import pipeline as PL          # noqa: E402
from src import pipeline_spark as PS    # noqa: E402

OUT = os.path.join(HERE, "out")
CAL = 546
MB = 1024 * 1024


class PeakSampler(threading.Thread):
    """Track the peak RSS of a process tree while a block runs."""
    def __init__(self, pid=None, interval=0.04):
        super().__init__(daemon=True)
        self.proc = psutil.Process(pid or os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = False

    def _tree_rss(self):
        tot = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except psutil.Error:
                pass
        return tot

    def run(self):
        while not self._stop:
            try:
                self.peak = max(self.peak, self._tree_rss())
            except psutil.Error:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop = True
        self.join(timeout=1)


def _parity(a: dict, b: dict) -> bool:
    import numpy as np
    import pandas as pd
    keys = {"customer_rfm": ["customer_id"], "customer_holdout": ["customer_id"],
            "channel_daily": ["channel", "day"]}
    for n in PS.MART_NAMES:
        x = a[n].sort_values(keys[n]).reset_index(drop=True)[sorted(a[n].columns)]
        y = b[n].sort_values(keys[n]).reset_index(drop=True)[sorted(b[n].columns)]
        if x.shape != y.shape:
            return False
        for c in x.columns:
            xa, ya = x[c], y[c]
            if pd.api.types.is_float_dtype(xa) or pd.api.types.is_float_dtype(ya):
                if not np.allclose(xa.astype(float), ya.astype(float),
                                   rtol=1e-9, atol=1e-6, equal_nan=True):
                    return False
            elif not np.array_equal(xa.to_numpy(), ya.to_numpy()):
                return False
    return True


def bench_duckdb(tx, touch, reps=3):
    times = []
    sampler = PeakSampler(); sampler.start()
    first = None
    for _ in range(reps):
        t0 = time.time()
        out = PS.build_marts_duckdb(tx, touch, CAL)
        times.append(time.time() - t0)
        first = first or out
    sampler.stop()
    return dict(median_s=statistics.median(times), runs=times,
                peak_mb=sampler.peak / MB), first


def bench_spark(tx, touch, reps=3):
    sampler = PeakSampler(); sampler.start()
    t0 = time.time()
    spark = PS.spark_session()
    startup = time.time() - t0
    times, first = [], None
    for _ in range(reps):
        t0 = time.time()
        out = PS.build_marts_spark(spark, tx, touch, CAL)
        times.append(time.time() - t0)
        first = first or out
    spark.stop()
    sampler.stop()
    return dict(startup_s=startup, cold_transform_s=times[0],
                warm_transform_s=statistics.median(times[1:]) if len(times) > 1 else times[0],
                transform_runs=times, peak_mb=sampler.peak / MB), first


def bench_dbt():
    """Time the real pipeline step: land raw into DuckDB, then `dbt build`."""
    t0 = time.time()
    PL.land()
    land_s = time.time() - t0
    env = dict(os.environ); env["DBT_PROFILES_DIR"] = PL.DBT
    cmd = [sys.executable, "-m", "dbt.cli.main", "build", "--project-dir", PL.DBT,
           "--vars", json.dumps({"calibration_days": CAL})]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=env, cwd=PL.DBT, text=True)
    sampler = PeakSampler(pid=p.pid); sampler.start()
    p.communicate()
    build_s = time.time() - t0
    sampler.stop()
    return dict(land_s=land_s, build_s=build_s, total_s=land_s + build_s,
                peak_mb=sampler.peak / MB, ok=p.returncode == 0)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Building identical raw frames ...")
    tx, touch = PL.raw_frames()
    print("  transactions=%d  touches=%d" % (len(tx), len(touch)))

    print("\nDuckDB (in-process, same SQL) ...")
    duck, duck_marts = bench_duckdb(tx, touch)
    print("  median %.3fs   peak %.0f MB" % (duck["median_s"], duck["peak_mb"]))

    have_java = PS.java_available()
    spark = spark_marts = None
    if have_java:
        print("\nSpark (local[*]) ...")
        spark, spark_marts = bench_spark(tx, touch)
        print("  startup %.1fs  cold %.2fs  warm %.2fs  peak %.0f MB"
              % (spark["startup_s"], spark["cold_transform_s"],
                 spark["warm_transform_s"], spark["peak_mb"]))
        assert _parity(spark_marts, duck_marts), "Spark and DuckDB marts differ!"
        print("  parity: Spark marts == DuckDB marts  (OK)")
    else:
        print("\nSpark: SKIPPED (no Java 17+ runtime found)")

    print("\ndbt build (subprocess, the real pipeline step) ...")
    dbt = bench_dbt()
    print("  land %.2fs  build %.1fs  total %.1fs  peak %.0f MB  ok=%s"
          % (dbt["land_s"], dbt["build_s"], dbt["total_s"], dbt["peak_mb"], dbt["ok"]))

    result = dict(rows=dict(transactions=len(tx), touches=len(touch)),
                  duckdb=duck, spark=spark, dbt=dbt,
                  java_available=have_java)
    with open(os.path.join(OUT, "bench_spark.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print("\n-> out/bench_spark.json")


if __name__ == "__main__":
    main()
