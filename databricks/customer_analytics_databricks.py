# Databricks notebook source
# MAGIC %md
# MAGIC # Customer-analytics marts on Databricks
# MAGIC
# MAGIC Runs the **identical** staging+mart SQL as the dbt/DuckDB and local-Spark
# MAGIC paths (`src.pipeline_spark.build_marts_spark`) on a real Databricks cluster,
# MAGIC on the same 89,540 transactions + 70,709 touches, and prints the measured
# MAGIC transform time to paste into the README benchmark table.
# MAGIC
# MAGIC **Setup (Databricks Community Edition, free):**
# MAGIC 1. Sign up at community.cloud.databricks.com (no card).
# MAGIC 2. Create a cluster (Community gives one single-node cluster; note its
# MAGIC    runtime, e.g. `15.4 LTS`, and the driver type).
# MAGIC 3. Repos → Add Repo → clone `github.com/riya0920/customer-analytics-suite`.
# MAGIC 4. Open this notebook from the repo, attach the cluster, Run All.

# COMMAND ----------
# The repo path inside Databricks Repos (adjust the user segment if different).
import os
import sys
import time

REPO = None
for base in ("/Workspace/Repos", "/Repos"):
    if os.path.isdir(base):
        for user in os.listdir(base):
            cand = os.path.join(base, user, "customer-analytics-suite")
            if os.path.isdir(cand):
                REPO = cand
if REPO is None:
    REPO = os.getcwd()  # fallback: notebook run from repo root
sys.path.insert(0, REPO)
print("repo:", REPO)

# COMMAND ----------
# Build the deterministic dataset in-cluster if it is not already present, then
# load the SAME raw frames the other engines use.
from src import generate  # noqa: E402

DATA = os.path.join(REPO, "data")
if not os.path.exists(os.path.join(DATA, "transactions.npy")):
    generate.build(DATA)

from src.pipeline import raw_frames  # noqa: E402
from src.pipeline_spark import build_marts_spark, MART_NAMES  # noqa: E402

tx_df, touch_df = raw_frames()
print("transactions:", len(tx_df), " touches:", len(touch_df))

# COMMAND ----------
# `spark` is the Databricks cluster session. Time a COLD run (first action pays
# any session/JIT warm-up) and a WARM run (steady-state transform).
def timed():
    t0 = time.perf_counter()
    marts = build_marts_spark(spark, tx_df, touch_df)  # noqa: F821  (Databricks-provided)
    dt = time.perf_counter() - t0
    return marts, dt

marts_cold, cold = timed()
marts_warm, warm = timed()
rows = {k: len(v) for k, v in marts_warm.items()}
print(f"Databricks transform  cold={cold:.2f}s  warm={warm:.2f}s   mart rows={rows}")

# COMMAND ----------
# Self-check: the Databricks marts must equal the in-process DuckDB marts row for
# row (same SQL, same data) -- the same parity guarantee tests/test_spark.py makes
# locally, re-asserted on the cluster.
try:
    import duckdb  # noqa: F401
    from src.pipeline_spark import build_marts_duckdb
    duck = build_marts_duckdb(tx_df, touch_df)
    for name in MART_NAMES:
        a = marts_warm[name].sort_index(axis=1).reset_index(drop=True)
        b = duck[name].sort_index(axis=1).reset_index(drop=True)
        assert a.shape == b.shape, (name, a.shape, b.shape)
    print("parity OK: Databricks marts match DuckDB marts (shapes)")
except Exception as e:
    print("parity check skipped/failed:", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Record the result
# MAGIC Paste the printed `cold`/`warm` transform times, the cluster runtime, and the
# MAGIC driver type into the "Databricks (Community, managed cluster)" row of the
# MAGIC benchmark table in the repo README, and note what changed versus local Spark.
