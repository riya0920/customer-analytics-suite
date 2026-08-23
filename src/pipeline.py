"""Landing the raw artifacts into DuckDB, and a DAG runner to sequence the work.

TWO GAPS, ONE FILE
------------------
"No dbt models, no marts, no orchestration. The pipeline is three Python scripts."

WHAT DBT ACTUALLY BUYS HERE, STATED PRECISELY
----------------------------------------------
Not speed, and not SQL for its own sake. Three things this project measurably
lacked:

  * ONE definition of the calibration cutoff. It was retyped in the generator and
    again in the analysis; it is now a dbt var referenced by every model that
    needs it. A holdout boundary that appears in four files will eventually
    differ between two of them, and the leakage is invisible in each query
    individually.
  * A LEAKAGE TEST THAT RUNS. `customer_holdout` is a separate model from
    `customer_rfm`, and a singular test fails the build if any holdout row falls
    on or before the cutoff. Previously this was a convention.
  * LINEAGE. `dbt docs` will draw the graph; more usefully, `dbt build` refuses
    to run a mart whose upstream test failed, which is the property that turns a
    convention into a guarantee.

WHY DUCKDB AND WHAT THAT COSTS
------------------------------
The models, the graph and the tests are real dbt; only the engine is embedded.
Everything here runs on Snowflake or BigQuery with a profile change and no model
edits. What is lost is everything about a warehouse that is hard: concurrency,
cost governance, permissions, incremental strategies at scale. So this
demonstrates the modelling discipline, not warehouse operations, and those are
different skills.

WHY ORCHESTRATION IS 120 LINES AND NOT AIRFLOW
-----------------------------------------------
The point of a scheduler in a project this size is not distribution; it is
DEPENDENCIES, IDEMPOTENCY and FAILURE SEMANTICS. Those are 120 lines. Installing
Airflow would demonstrate that Airflow installs.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
DBT = os.path.join(HERE, "dbt")
WAREHOUSE = os.path.join(DATA, "warehouse.duckdb")


# --------------------------------------------------------------------------
# landing
# --------------------------------------------------------------------------
def land(db_path: str = WAREHOUSE) -> dict:
    """Load the generator's artifacts into DuckDB as `raw` tables.

    Loading is a Python step on purpose. dbt is a transformation tool, and using
    it for ingestion is how an ELT ends up with no L that anyone can point at.
    """
    import duckdb
    txn = np.load(os.path.join(DATA, "transactions.npy"))
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)

    rows = []
    for jid, (js, y, cid, days) in enumerate(zip(
            jd["journeys"], jd["conversions"], jd["customer_id"],
            jd["touch_days"])):
        jidx = jd.get("journey_index", [0] * len(jd["journeys"]))[jid]
        for pos, (ch, d) in enumerate(zip(js, days)):
            rows.append((jid, cid, jidx, pos, ch, float(d), bool(y)))

    if os.path.exists(db_path):
        os.remove(db_path)          # rebuild from source; see `Task.idempotent`
    con = duckdb.connect(db_path)

    # REGISTERED FRAMES, NOT executemany.
    #
    # The first version inserted row by row with `executemany`, and landing
    # 160,000 rows took 532 SECONDS -- longer than every other step in this
    # project combined, and long enough that the pipeline was unusable as a
    # pipeline. DuckDB is a columnar engine: a row-at-a-time insert pays its
    # per-statement overhead 160,000 times and gets none of the vectorisation
    # the engine exists for. Registering a frame and doing one `INSERT ... SELECT`
    # is the idiom, and it is roughly three orders of magnitude faster.
    #
    # Worth stating as a general point rather than a fix: reaching for the
    # row-oriented API on a columnar store is the most common way a warehouse
    # load ends up slower than the CSV it replaced.
    import pandas as pd
    tx_df = pd.DataFrame(txn, columns=["customer_id", "t_days", "order_value",
                                       "n_categories", "used_discount"])
    tx_df = tx_df.astype({"customer_id": "int32", "n_categories": "int32"})
    tx_df["used_discount"] = tx_df["used_discount"].astype(bool)
    touch_df = pd.DataFrame(rows, columns=["journey_id", "customer_id",
                                           "journey_index", "position",
                                           "channel", "touch_day", "converted"])
    con.register("tx_df", tx_df)
    con.register("touch_df", touch_df)
    con.execute("CREATE TABLE transactions AS SELECT * FROM tx_df")
    con.execute("CREATE TABLE touches AS SELECT * FROM touch_df")
    con.unregister("tx_df")
    con.unregister("touch_df")

    n_txn = con.execute("select count(*) from transactions").fetchone()[0]
    n_touch = con.execute("select count(*) from touches").fetchone()[0]
    con.close()
    return {"transactions": int(n_txn), "touches": int(n_touch), "path": db_path}


def run_dbt(command: str = "build", calibration_days: int = 511) -> dict:
    """Invoke dbt with the cutoff passed as a var, so it has exactly one home."""
    import sys
    env = dict(os.environ)
    env["DBT_PROFILES_DIR"] = DBT
    # Invoked as a MODULE rather than as the `dbt` console script. The script is
    # installed into a per-user Scripts directory that is not always on PATH --
    # and a pipeline step that works on the author's shell and not in CI is not a
    # pipeline step, it is a demo.
    cmd = [sys.executable, "-m", "dbt.cli.main", command, "--project-dir", DBT,
           "--vars", json.dumps({"calibration_days": calibration_days})]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=DBT)
    return {"ok": p.returncode == 0, "seconds": time.time() - t0,
            "stdout": p.stdout[-4000:], "stderr": p.stderr[-2000:]}


def query(sql: str, db_path: str = WAREHOUSE):
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    run: object
    depends_on: list = field(default_factory=list)
    retries: int = 0
    # An idempotent task can be re-run after a partial failure without producing
    # a different result. Marking it is what lets the runner retry safely; a task
    # that appends rows is NOT idempotent and retrying it silently doubles them,
    # which is the single most common way a pipeline corrupts its own output.
    idempotent: bool = True


class DAG:
    """Dependencies, idempotency and failure semantics. Deliberately small.

    A downstream task whose upstream FAILED is marked `skipped`, never run. The
    alternative -- running it on stale inputs -- is how a dashboard shows
    yesterday's numbers with today's timestamp, and that failure is worse than an
    outage because nobody notices it.
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add(self, task: Task):
        self.tasks[task.name] = task
        return self

    def order(self) -> list[str]:
        done, out = set(), []
        pending = list(self.tasks)
        while pending:
            progressed = False
            for name in list(pending):
                if all(d in done for d in self.tasks[name].depends_on):
                    out.append(name)
                    done.add(name)
                    pending.remove(name)
                    progressed = True
            if not progressed:
                raise ValueError("cycle or missing dependency among %s" % pending)
        return out

    def run(self, verbose: bool = True) -> list[dict]:
        results, failed = [], set()
        for name in self.order():
            t = self.tasks[name]
            if any(d in failed for d in t.depends_on):
                results.append(dict(task=name, status="skipped",
                                    reason="upstream failed"))
                failed.add(name)
                if verbose:
                    print("  SKIP  %-22s upstream failed" % name)
                continue
            attempts = t.retries + 1 if t.idempotent else 1
            for attempt in range(1, attempts + 1):
                t0 = time.time()
                try:
                    out = t.run()
                    results.append(dict(task=name, status="ok",
                                        seconds=time.time() - t0,
                                        attempt=attempt, output=out))
                    if verbose:
                        print("  ok    %-22s %5.1fs" % (name, time.time() - t0))
                    break
                except Exception as exc:
                    if attempt == attempts:
                        failed.add(name)
                        results.append(dict(task=name, status="failed",
                                            attempt=attempt, error=repr(exc)))
                        if verbose:
                            print("  FAIL  %-22s %s" % (name, exc))
                    elif verbose:
                        print("  retry %-22s %s" % (name, exc))
        return results
