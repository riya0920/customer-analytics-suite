"""A PySpark port of the dbt/DuckDB transformations, kept ALONGSIDE the dbt path
rather than replacing it.

The five dbt models (two staging views, three marts) are re-expressed here as a
set of SQL strings that run **unchanged on both engines** -- DuckDB in-process
and Spark local -- so the port is provably the same transformation, not a
lookalike. `tests/test_spark.py` asserts the Spark marts equal what dbt actually
materialised into the warehouse, row for row.

Why this exists (the honest version): distributed engines earn their keep at a
scale this dataset is nowhere near (89k transactions, 71k touches). The point of
the port is to show the transformations translate cleanly to the Spark DataFrame/
SQL API and to *measure* the crossover cost -- see `bench_spark_vs_dbt.py`, whose
finding is that Spark is far slower here, dominated by JVM start-up and task
scheduling. A documented negative result is the deliverable.

The SQL is dialect-neutral on purpose: only casts both engines share (integer,
double, boolean), `floor`, `count(distinct ...)`, and `case when`. That is what
lets one string serve both and makes the parity guarantee meaningful.
"""
from __future__ import annotations

import glob
import os
import sys


# --------------------------------------------------------------------------
# JVM discovery -- Spark needs a Java 17+ runtime
# --------------------------------------------------------------------------
def ensure_java() -> str | None:
    """Return a usable java.exe path, setting JAVA_HOME/PATH if needed.

    Honours an existing JAVA_HOME; otherwise looks for a portable Temurin 17 in
    ~/.jdks (where the setup step unpacks it). Returns None if no JVM is found,
    so callers can skip cleanly rather than crash.
    """
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.exists(os.path.join(jh, "bin", "java.exe")):
        _prepend_path(os.path.join(jh, "bin"))
        return os.path.join(jh, "bin", "java.exe")
    if jh and os.path.exists(os.path.join(jh, "bin", "java")):  # posix
        _prepend_path(os.path.join(jh, "bin"))
        return os.path.join(jh, "bin", "java")
    for pat in (os.path.expanduser("~/.jdks/jdk-17*"),
                os.path.expanduser("~/.jdks/jdk-21*")):
        for cand in sorted(glob.glob(pat)):
            for exe in ("bin/java.exe", "bin/java"):
                p = os.path.join(cand, exe)
                if os.path.exists(p):
                    os.environ["JAVA_HOME"] = cand
                    _prepend_path(os.path.dirname(p))
                    return p
    return None


def _prepend_path(d: str):
    if d not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def java_available() -> bool:
    return ensure_java() is not None


# --------------------------------------------------------------------------
# the transformations -- one SQL body, two engines
# --------------------------------------------------------------------------
# raw view names match dbt's source() names; staging/mart names match the models.
STAGING_SQL = {
    "stg_transactions": """
        select
            cast(customer_id  as integer) as customer_id,
            cast(t_days       as double)  as t_days,
            cast(order_value  as double)  as order_value,
            cast(n_products as integer) as n_products,
            cast(had_return as boolean) as had_return
        from transactions
    """,
    "stg_touches": """
        select
            cast(journey_id    as integer) as journey_id,
            cast(customer_id   as integer) as customer_id,
            cast(journey_index as integer) as journey_index,
            cast(position      as integer) as touch_position,
            channel                        as channel,
            cast(touch_day     as double)  as touch_day,
            cast(converted     as boolean) as converted
        from touches
    """,
}


def mart_sql(calibration_days: int) -> dict:
    """The three marts, parameterised by the one cutoff -- mirroring the dbt
    models and their `var('calibration_days')`."""
    cal = int(calibration_days)
    return {
        "customer_rfm": f"""
            with t as (
                select * from stg_transactions where t_days <= {cal}
            )
            select
                customer_id,
                count(*)                    as frequency,
                max(t_days)                 as recency_day,
                {cal} - max(t_days)         as days_since_last,
                min(t_days)                 as first_purchase_day,
                avg(order_value)            as avg_order_value,
                sum(order_value)            as total_value,
                avg(n_products)           as avg_products,
                -- cast the summands to double: Spark parses `1.0` as DECIMAL and
                -- avg(decimal) rounds to 5 places, silently disagreeing with
                -- DuckDB's double. Forcing double keeps the two engines identical.
                avg(case when had_return then cast(1 as double)
                         else cast(0 as double) end) as return_rate
            from t
            group by customer_id
        """,
        "customer_holdout": f"""
            select
                customer_id,
                count(*)          as holdout_orders,
                sum(order_value)  as holdout_value,
                max(t_days)       as holdout_last_day
            from stg_transactions
            where t_days > {cal}
            group by customer_id
        """,
        "channel_daily": """
            select
                channel,
                cast(floor(touch_day) as integer) as day,
                count(*)                          as touches,
                count(distinct journey_id)        as journeys_touched,
                sum(case when converted then 1 else 0 end)
                                                  as touches_on_converting_journeys
            from stg_touches
            group by channel, cast(floor(touch_day) as integer)
        """,
    }


MART_NAMES = ["customer_rfm", "customer_holdout", "channel_daily"]


# --------------------------------------------------------------------------
# DuckDB engine (in-process) -- the same SQL, for an apples-to-apples baseline
# --------------------------------------------------------------------------
def build_marts_duckdb(tx_df, touch_df, calibration_days: int = 546) -> dict:
    """Run the staging + mart SQL on DuckDB in-process. Returns pandas frames."""
    import duckdb
    con = duckdb.connect()
    try:
        con.register("transactions", tx_df)
        con.register("touches", touch_df)
        for name, sql in STAGING_SQL.items():
            con.execute(f"create or replace temp view {name} as {sql}")
        out = {}
        for name, sql in mart_sql(calibration_days).items():
            out[name] = con.execute(sql).fetchdf()
        return out
    finally:
        con.close()


# --------------------------------------------------------------------------
# Spark engine (local) -- the port
# --------------------------------------------------------------------------
def spark_session(app: str = "cas-spark"):
    """A small local SparkSession tuned for a single machine and a small dataset.

    Low shuffle-partition count (the default 200 is absurd at this scale and adds
    scheduling overhead), UI off, and the driver python pinned to this
    interpreter so the workers agree with the driver.
    """
    if not java_available():
        raise RuntimeError("no Java 17+ runtime found (set JAVA_HOME)")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[*]").appName(app)
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8")
             # Arrow makes the pandas<->Spark interchange columnar; without it,
             # createDataFrame/toPandas fall back to a row-at-a-time path that is
             # ~4x slower here. (pandas 3.x prints a "not yet fully supported"
             # warning under Arrow, but the marts come out byte-identical -- the
             # parity test proves it.)
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_marts_spark(spark, tx_df, touch_df, calibration_days: int = 546) -> dict:
    """Run the identical staging + mart SQL on Spark. Returns pandas frames
    (collected with .toPandas(), which is what forces Spark's lazy plan to run)."""
    spark.createDataFrame(tx_df).createOrReplaceTempView("transactions")
    spark.createDataFrame(touch_df).createOrReplaceTempView("touches")
    for name, sql in STAGING_SQL.items():
        spark.sql(sql).createOrReplaceTempView(name)
    out = {}
    for name, sql in mart_sql(calibration_days).items():
        out[name] = spark.sql(sql).toPandas()
    return out
