"""
Land the raw (transactions, touches) tables into Snowflake -- the Snowflake
equivalent of `src.pipeline.land` (which lands into DuckDB). It reuses the exact
same `raw_frames()` builder, so both warehouses get byte-identical inputs and the
only thing that changes between them is the engine.

After this, the same dbt models build on Snowflake with only a target switch:

    export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
    python land_snowflake.py
    cd dbt && dbt build --target snowflake

Requires `snowflake-connector-python[pandas]` and a Snowflake account (a free
trial is enough). Kept optional: the connector is imported lazily so the default
DuckDB path never needs it. See the README's Snowflake section.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import raw_frames  # noqa: E402


def _conn():
    import snowflake.connector  # lazy: optional dependency

    return snowflake.connector.connect(
        account=_req("SNOWFLAKE_ACCOUNT"),
        user=_req("SNOWFLAKE_USER"),
        password=_req("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "CUSTOMER_ANALYTICS"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "main"),
    )


def _req(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"{var} is not set (see the README's Snowflake section).")
    return val


def land_snowflake() -> dict:
    """Create/replace the raw tables in Snowflake from the shared raw frames."""
    from snowflake.connector.pandas_tools import write_pandas  # lazy

    tx_df, touch_df = raw_frames()
    db = os.getenv("SNOWFLAKE_DATABASE", "CUSTOMER_ANALYTICS")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "main")

    con = _conn()
    try:
        cur = con.cursor()
        cur.execute(f'CREATE DATABASE IF NOT EXISTS "{db}"')
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{db}"."{schema}"')
        cur.execute(f'USE SCHEMA "{db}"."{schema}"')
        # write_pandas creates the table (auto_create_table) matching the frame,
        # overwriting any prior load so re-runs are idempotent (like land()).
        n_tx, _, rows_tx, _ = write_pandas(
            con, tx_df, "transactions", auto_create_table=True, overwrite=True,
            quote_identifiers=False,
        )
        n_touch, _, rows_touch, _ = write_pandas(
            con, touch_df, "touches", auto_create_table=True, overwrite=True,
            quote_identifiers=False,
        )
        return {"transactions": rows_tx, "touches": rows_touch,
                "database": db, "schema": schema}
    finally:
        con.close()


if __name__ == "__main__":
    result = land_snowflake()
    print("Landed into Snowflake:", result)
    print("Now run:  cd dbt && dbt build --target snowflake")
