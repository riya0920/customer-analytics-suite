# Databricks notebook source
# MAGIC %md
# MAGIC # Customer-analytics marts on Databricks
# MAGIC
# MAGIC The **same** staging + mart transformations as the local Spark path
# MAGIC (`src/pipeline_spark.py`) and the dbt/DuckDB path - the SQL below is copied
# MAGIC verbatim from `pipeline_spark.STAGING_SQL` / `mart_sql`, so this is a change
# MAGIC of *where Spark runs*, not of the logic. Kept alongside the local path.
# MAGIC
# MAGIC **What changed vs local `spark_session()` (documented in the repo README):**
# MAGIC - No `SparkSession.builder…getOrCreate()` - Databricks provides `spark`.
# MAGIC - No JVM/JDK to install and no `local[*]` master - a managed cluster.
# MAGIC - Data is read from a **volume / DBFS** with `spark.read.csv`, not built from
# MAGIC   a pandas frame via `createDataFrame` (so no pandas↔JVM serialisation on the
# MAGIC   way in).
# MAGIC - The Databricks Runtime pins the Spark + Python versions (and may add Photon).
# MAGIC
# MAGIC **Setup:** upload `databricks/data/transactions.csv` and `touches.csv`
# MAGIC (regenerate with `python databricks/export_raw_csv.py`) to the `RAW_DIR`
# MAGIC below (a Unity Catalog volume, or DBFS `/FileStore`).

# COMMAND ----------

import time

# Point this at wherever the two CSVs were uploaded.
RAW_DIR = "/Volumes/workspace/default/cas_raw"   # or "dbfs:/FileStore/cas_raw"
CAL = 546                                          # calibration_days (one home, as in dbt)

# COMMAND ----------

# Read the raw CSVs into Spark and register them under the names the SQL expects.
t0 = time.time()
(spark.read.option("header", True).option("inferSchema", True)
      .csv(f"{RAW_DIR}/transactions.csv").createOrReplaceTempView("transactions"))
(spark.read.option("header", True).option("inferSchema", True)
      .csv(f"{RAW_DIR}/touches.csv").createOrReplaceTempView("touches"))

# COMMAND ----------

# --- staging (verbatim from pipeline_spark.STAGING_SQL) ---
spark.sql("""
    select
        cast(customer_id  as integer) as customer_id,
        cast(t_days       as double)  as t_days,
        cast(order_value  as double)  as order_value,
        cast(n_products as integer) as n_products,
        cast(had_return as boolean) as had_return
    from transactions
""").createOrReplaceTempView("stg_transactions")

spark.sql("""
    select
        cast(journey_id    as integer) as journey_id,
        cast(customer_id   as integer) as customer_id,
        cast(journey_index as integer) as journey_index,
        cast(position      as integer) as touch_position,
        channel                        as channel,
        cast(touch_day     as double)  as touch_day,
        cast(converted     as boolean) as converted
    from touches
""").createOrReplaceTempView("stg_touches")

# COMMAND ----------

# --- marts (verbatim from pipeline_spark.mart_sql; return_rate summands cast to
# --- double so avg matches DuckDB, not Spark's DECIMAL rounding) ---
customer_rfm = spark.sql(f"""
    with t as (select * from stg_transactions where t_days <= {CAL})
    select
        customer_id,
        count(*)                    as frequency,
        max(t_days)                 as recency_day,
        {CAL} - max(t_days)         as days_since_last,
        min(t_days)                 as first_purchase_day,
        avg(order_value)            as avg_order_value,
        sum(order_value)            as total_value,
        avg(n_products)           as avg_products,
        avg(case when had_return then cast(1 as double)
                 else cast(0 as double) end) as return_rate
    from t group by customer_id
""")

customer_holdout = spark.sql(f"""
    select customer_id, count(*) as holdout_orders,
           sum(order_value) as holdout_value, max(t_days) as holdout_last_day
    from stg_transactions where t_days > {CAL} group by customer_id
""")

channel_daily = spark.sql("""
    select channel, cast(floor(touch_day) as integer) as day,
           count(*) as touches, count(distinct journey_id) as journeys_touched,
           sum(case when converted then 1 else 0 end) as touches_on_converting_journeys
    from stg_touches group by channel, cast(floor(touch_day) as integer)
""")

# Force execution (Spark is lazy) and time the whole transform.
n_rfm = customer_rfm.count()
n_holdout = customer_holdout.count()
n_channel = channel_daily.count()
elapsed = time.time() - t0

print(f"transform wall-clock: {elapsed:.2f}s")
print(f"customer_rfm      : {n_rfm} rows")
print(f"customer_holdout  : {n_holdout} rows")
print(f"channel_daily     : {n_channel} rows")

# COMMAND ----------

# Parity with the dbt/DuckDB warehouse and the local-Spark run: same marts.
assert n_rfm == 4899, n_rfm
assert n_holdout == 2592, n_holdout
assert n_channel == 303, n_channel
# leakage guard, as in the dbt singular test + tests/test_spark.py
assert customer_holdout.filter("holdout_last_day <= %d" % CAL).count() == 0
print("PARITY OK - Databricks marts match the DuckDB/local-Spark marts "
      "(4899 / 2592 / 303, no leakage).")

# COMMAND ----------

# MAGIC %md
# MAGIC Record the printed `transform wall-clock` in the repo README's benchmark
# MAGIC table, next to DuckDB and local Spark. Expect it to be **slower** than local
# MAGIC Spark for a cold cluster (managed cluster / serverless start-up dominates at
# MAGIC 160k rows) and far slower than DuckDB - the same finding as the local
# MAGIC benchmark, one more rung out: the managed platform's fixed costs only pay off
# MAGIC at data sizes this dataset is nowhere near.
