"""
The customer-analytics pipeline as an Airflow DAG — a port of the hand-built
120-line runner in ``src/pipeline.py`` (the ``DAG`` / ``Task`` classes), kept
ALONGSIDE it, not replacing it.

Same three tasks, same order, same idempotency intent:

    land_raw  ->  dbt_build  ->  read_marts

Each task wraps the *identical* function the hand-built runner calls
(``src.pipeline.land`` / ``run_dbt`` / ``query``), so this is a change of
orchestrator, not of logic. What Airflow adds and what it costs is written up in
the README ("Orchestration: 120-line runner vs Airflow").

Run it without a scheduler or webserver:

    export AIRFLOW_HOME=~/cas-airflow-home
    export AIRFLOW__CORE__DAGS_FOLDER=<repo>/airflow/dags
    export AIRFLOW__CORE__LOAD_EXAMPLES=False
    airflow db migrate
    airflow dags test cas_pipeline 2025-01-01
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# The repo root is two levels up from this file (airflow/dags/cas_pipeline.py),
# so the same src/ package the hand-built runner uses is importable here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import pipeline as PL  # noqa: E402


def _land(**_):
    return PL.land()


def _dbt_build(**_):
    result = PL.run_dbt("build")
    if not result["ok"]:
        # Surface dbt's failure as a task failure so Airflow marks downstream
        # tasks upstream_failed -- the framework's equivalent of the hand-built
        # runner's "skip on upstream failure".
        raise RuntimeError("dbt build failed:\n" + result.get("stderr", "")[-1500:])
    return {"seconds": result["seconds"]}


def _read_marts(**_):
    rfm = len(PL.query("select * from customer_rfm"))
    holdout = len(PL.query("select * from customer_holdout"))
    return {"customer_rfm": rfm, "customer_holdout": holdout}


default_args = {
    "owner": "customer-analytics",
    # Airflow gives retries + exponential backoff for free; the hand-built runner
    # had a plain retry count and only for tasks flagged idempotent.
    "retries": 1,
    "retry_delay": __import__("datetime").timedelta(seconds=10),
}

with DAG(
    dag_id="cas_pipeline",
    description="Land raw -> dbt build -> read marts (port of the 120-line runner)",
    start_date=datetime(2025, 1, 1),
    schedule=None,          # triggered/tested, not scheduled
    catchup=False,
    default_args=default_args,
    tags=["customer-analytics", "dbt", "duckdb"],
) as dag:
    land_raw = PythonOperator(
        task_id="land_raw",
        python_callable=_land,
        # land() rebuilds the warehouse from source each run -> safe to retry.
        retries=1,
    )
    dbt_build = PythonOperator(
        task_id="dbt_build",
        python_callable=_dbt_build,
        # `dbt build` is idempotent (it rebuilds tables), so retry is safe.
        retries=1,
    )
    read_marts = PythonOperator(
        task_id="read_marts",
        python_callable=_read_marts,
        retries=0,
    )

    # The dependency graph the hand-built DAG.order() computed by hand.
    land_raw >> dbt_build >> read_marts
