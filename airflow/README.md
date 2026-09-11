# Airflow port of the pipeline

`dags/cas_pipeline.py` is the Airflow version of the hand-built runner in
`../src/pipeline.py`. It is kept **alongside** the runner, not in place of it. See
the main README ("Orchestration: the 120-line runner vs Airflow") for the
comparison and the measured run.

Airflow is **not** in the project's `requirements.txt` — it is a heavy, Linux-only
dependency with a hard Python ≤3.12 constraint, so it stays optional.

## Run it (Python ≤3.12, on Linux / WSL / Docker)

This machine is Windows + Python 3.14, on which Airflow will not install; it was
run under a `uv`-provisioned Python 3.12 inside WSL:

```bash
# 1. a supported interpreter without sudo
pip install --user uv && uv python install 3.12
uv venv --python 3.12 ~/cas-airflow && source ~/cas-airflow/bin/activate

# 2. Airflow (constrained) + the pipeline's own deps in the same venv
CON=https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.12.txt
uv pip install "apache-airflow==2.10.4" --constraint "$CON"
uv pip install duckdb dbt-duckdb numpy pandas

# 3. point Airflow at this repo's dags and run the DAG end-to-end (no scheduler)
export AIRFLOW_HOME=~/cas-airflow-home
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"   # from the repo root
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow db migrate
airflow dags test cas_pipeline 2025-01-01
```

Expected: all three tasks succeed and `read_marts` returns
`{'customer_rfm': 7894, 'customer_holdout': 2847}` — the same marts the hand-built
runner produces.
