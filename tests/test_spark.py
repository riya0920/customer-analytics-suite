"""Tests for the PySpark port (src/pipeline_spark.py).

These are the Spark equivalents of the three dbt singular tests --
  * the leakage guard        (assert_holdout_after_cutoff.sql)
  * RFM-inside-calibration   (assert_rfm_inside_calibration.sql)
  * channel_daily is reach   (assert_channel_daily_is_reach_not_attribution.sql)
plus the guarantee that matters most for a port: the Spark marts equal, row for
row, what dbt actually materialised into the DuckDB warehouse. If Spark ever
drifts from the canonical SQL, that parity test fails.

The whole module skips cleanly when pyspark or a Java 17+ runtime is absent, so
the existing 69 tests never depend on a JVM being installed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline as PL          # noqa: E402
from src import pipeline_spark as PS    # noqa: E402

pytestmark = pytest.mark.skipif(not PS.java_available(),
                                reason="no Java 17+ runtime (set JAVA_HOME)")

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CAL = 511
KEYS = {"customer_rfm": ["customer_id"], "customer_holdout": ["customer_id"],
        "channel_daily": ["channel", "day"]}


@pytest.fixture(scope="module")
def frames():
    return PL.raw_frames()


@pytest.fixture(scope="module")
def spark_marts(frames):
    """Build the Spark marts once (a SparkSession is expensive to start)."""
    tx, touch = frames
    spark = PS.spark_session("cas-tests")
    try:
        yield PS.build_marts_spark(spark, tx, touch, CAL)
    finally:
        spark.stop()


@pytest.fixture(scope="module")
def duck_marts(frames):
    tx, touch = frames
    return PS.build_marts_duckdb(tx, touch, CAL)


def _equal(a: pd.DataFrame, b: pd.DataFrame, key) -> bool:
    a = a.sort_values(key).reset_index(drop=True)[sorted(a.columns)]
    b = b.sort_values(key).reset_index(drop=True)[sorted(b.columns)]
    if a.shape != b.shape:
        return False
    for c in a.columns:
        xa, ya = a[c], b[c]
        if pd.api.types.is_float_dtype(xa) or pd.api.types.is_float_dtype(ya):
            if not np.allclose(xa.astype(float), ya.astype(float),
                               rtol=1e-9, atol=1e-6, equal_nan=True):
                return False
        elif not np.array_equal(xa.to_numpy(), ya.to_numpy()):
            return False
    return True


# -- the port is faithful ---------------------------------------------------
def test_spark_marts_equal_duckdb_inprocess(spark_marts, duck_marts):
    """Same SQL, two engines, identical answers."""
    for n in PS.MART_NAMES:
        assert _equal(spark_marts[n], duck_marts[n], KEYS[n]), n


def test_spark_marts_equal_the_dbt_warehouse(spark_marts):
    """The strongest guarantee: Spark reproduces what dbt materialised."""
    if not os.path.exists(PL.WAREHOUSE):
        pytest.skip("run `python run_complete.py` first to build the warehouse")
    for n in PS.MART_NAMES:
        wh = PL.query("select * from %s" % n)
        assert _equal(spark_marts[n], wh, KEYS[n]), n


# -- the three dbt singular tests, re-expressed on Spark output -------------
def test_spark_holdout_has_no_leakage(spark_marts):
    ho = spark_marts["customer_holdout"]
    assert int((ho["holdout_last_day"] <= CAL).sum()) == 0


def test_spark_rfm_inside_calibration(spark_marts):
    rfm = spark_marts["customer_rfm"]
    assert int((rfm["recency_day"] > CAL).sum()) == 0


def test_spark_channel_daily_is_reach_not_attribution(spark_marts, frames):
    _, touch = frames
    credited = int(spark_marts["channel_daily"]["touches_on_converting_journeys"].sum())
    actual = int(touch.groupby("journey_id")["converted"].max().sum())
    assert credited > actual


# -- basic shape guarantees -------------------------------------------------
def test_spark_rfm_is_keyed_by_customer(spark_marts):
    rfm = spark_marts["customer_rfm"]
    assert len(rfm) > 0
    assert rfm["customer_id"].is_unique
