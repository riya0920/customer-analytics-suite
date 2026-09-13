# Running the Spark marts on Databricks (Community Edition)

This runs the **identical** staging + mart SQL as the DuckDB and local-Spark
paths (`src.pipeline_spark.build_marts_spark`) on a real Databricks managed
cluster, to add the third row to the benchmark table in the repo README.

## Steps

1. **Sign up** for Databricks Community Edition at
   `community.cloud.databricks.com`: free, no credit card.
2. **Create a cluster.** Community gives one single-node cluster. Note its
   **runtime version** (e.g. `15.4 LTS`, Spark 3.5) and driver type; those go in
   the README row.
3. **Clone the repo.** Repos → *Add Repo* →
   `https://github.com/riya0920/customer-analytics-suite`.
4. **Open** `databricks/customer_analytics_databricks.py` from the repo, **attach**
   the cluster, and **Run All**. The notebook:
   - builds the deterministic dataset in-cluster (`src.generate`) if absent,
   - loads the same raw frames (`src.pipeline.raw_frames`),
   - times `build_marts_spark(spark, …)` cold and warm on the cluster's session,
   - re-asserts the DuckDB↔Spark mart parity (same guarantee as
     `tests/test_spark.py`, on the cluster),
   - prints `Databricks transform  cold=…s  warm=…s`.
5. **Record it.** Paste the warm (and cold) time, the runtime version, and the
   driver type into the "Databricks (Community, managed cluster)" row of the repo
   README, and note what changed versus local Spark.

## Why this can't be run from the repo's CI

Databricks Community requires an interactive account and a hosted cluster; there
is no unattended/API path on the free tier. So the notebook and steps are here,
and the measured number is filled in after a run; the row stays blank rather
than carrying an invented figure.
