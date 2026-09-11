"""
Verify the dbt models are Snowflake-dialect-clean WITHOUT a Snowflake account.

The project's README claims the models "would run on Snowflake with a profile
change and no model edits." This turns that claim into a check: every compiled
model's SQL is parsed as DuckDB and transpiled to the Snowflake dialect with
sqlglot; any construct Snowflake could not accept surfaces as a transpile error.

It is a static portability guarantee, not a substitute for a live run (that needs
an account -- see the README's Snowflake section). Run:

    python run_complete.py            # first, so dbt/target/compiled exists
    python check_snowflake_portability.py
"""
from __future__ import annotations

import glob
import os
import sys

import sqlglot

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILED = os.path.join(
    HERE, "dbt", "target", "compiled", "retail_customer_analytics", "models"
)


def compiled_models() -> list[str]:
    return sorted(glob.glob(os.path.join(COMPILED, "**", "*.sql"), recursive=True))


def transpiles_to_snowflake(sql: str) -> tuple[bool, str]:
    try:
        out = sqlglot.transpile(sql, read="duckdb", write="snowflake")
        return (bool(out and out[0].strip()), "")
    except Exception as exc:  # sqlglot.errors.ParseError / UnsupportedError
        return (False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    models = compiled_models()
    if not models:
        print("No compiled SQL found. Run `python run_complete.py` first "
              "(it builds dbt/target/compiled/).")
        return 2

    ok = 0
    print(f"Transpiling {len(models)} compiled model files DuckDB -> Snowflake:\n")
    for path in models:
        with open(path, encoding="utf-8") as f:
            sql = f.read()
        good, err = transpiles_to_snowflake(sql)
        name = os.path.relpath(path, COMPILED)
        print(f"  {'ok ' if good else 'FAIL'}  {name}" + (f"  -- {err}" if err else ""))
        ok += good

    print(f"\n{ok}/{len(models)} model files transpile to Snowflake with no errors.")
    return 0 if ok == len(models) else 1


if __name__ == "__main__":
    sys.exit(main())
