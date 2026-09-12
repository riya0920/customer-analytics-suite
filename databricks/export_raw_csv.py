"""
Write the raw (transactions, touches) frames to CSV for upload to Databricks.

The Databricks notebook (cas_marts_databricks.py) reads these two files from a
Unity Catalog volume / DBFS instead of building Spark DataFrames from pandas the
way the local path does -- that difference is the point of the comparison. Uses
the SAME src.pipeline.raw_frames() builder, so Databricks transforms the identical
inputs DuckDB and local Spark do.

    python databricks/export_raw_csv.py    # -> databricks/data/{transactions,touches}.csv
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.pipeline import raw_frames  # noqa: E402

OUT = os.path.join(HERE, "data")


def main():
    os.makedirs(OUT, exist_ok=True)
    tx, touch = raw_frames()
    tx_path = os.path.join(OUT, "transactions.csv")
    touch_path = os.path.join(OUT, "touches.csv")
    tx.to_csv(tx_path, index=False)
    touch.to_csv(touch_path, index=False)
    print("wrote %s (%d rows)" % (tx_path, len(tx)))
    print("wrote %s (%d rows)" % (touch_path, len(touch)))


if __name__ == "__main__":
    main()
