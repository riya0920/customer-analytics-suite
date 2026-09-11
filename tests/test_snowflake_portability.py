"""
Static Snowflake-portability guarantee for the dbt models.

Turns the README's "would run on Snowflake with a profile change and no model
edits" from a claim into a test: every compiled model's SQL must transpile from
DuckDB to the Snowflake dialect with no errors (sqlglot). This is not a live run
(that needs an account -- see the README's Snowflake section); it is a guard that
no DuckDB-only construct sneaks into a model.

Skips cleanly if the compiled SQL is absent (dbt/target/ is gitignored) -- run
`python run_complete.py` first to build it, same as the other warehouse-dependent
tests here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("sqlglot")

import check_snowflake_portability as CP  # noqa: E402


def _models():
    return CP.compiled_models()


@pytest.mark.skipif(not _models(), reason="run `python run_complete.py` to compile dbt")
def test_all_compiled_models_transpile_to_snowflake():
    failures = []
    for path in _models():
        with open(path, encoding="utf-8") as f:
            good, err = CP.transpiles_to_snowflake(f.read())
        if not good:
            failures.append((os.path.basename(path), err))
    assert not failures, f"non-Snowflake-portable models: {failures}"


@pytest.mark.skipif(not _models(), reason="run `python run_complete.py` to compile dbt")
def test_there_are_models_to_check():
    # guard against the check silently passing on zero files
    assert len(_models()) >= 5  # 2 staging + 3 marts (+ generic test SQL)
