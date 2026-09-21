# Engineering notes

How the analysis is wired: a dbt pipeline on DuckDB, and the same pipeline run on
Airflow, Snowflake (checked statically), local Spark and Databricks to measure what
those tools add at this size.

> **Note on timings.** The runtimes below were measured on the earlier, larger
> simulated version of the data (89,540 transactions + 70,709 touches). The
> project now runs on real Online Retail II data (30,816 purchase days + 42,613
> touches). The SQL is unchanged and cross-engine parity is re-checked by
> `tests/test_spark.py`; row counts below are the current ones.

## A pipeline, not three scripts

```
ok  land_raw     1.7s     ok  dbt_build  31.3s     ok  read_marts  0.2s
dbt: PASS=15 WARN=0 ERROR=0 SKIP=0 TOTAL=15
marts: customer_rfm 4,899 rows · customer_holdout 2,592 rows
```

Five dbt models (two staging views, three marts) on DuckDB, ten dbt tests, and a
120-line DAG runner. What dbt actually buys here, and it is not SQL for its own
sake:

- **One definition of the calibration cutoff.** It was retyped in the generator
  and again in the analysis; it is now a dbt var. A cutoff living in four files
  will eventually differ between two of them, and the leakage is invisible in each
  query on its own.
- **A leakage test that runs.** `customer_holdout` is a separate model from
  `customer_rfm`, and a singular test **fails the build** if any holdout row falls
  on or before the cutoff. That was previously a convention, and a convention is
  a thing people follow until a deadline.
- **A test that asserts a table is what it claims to be.** `channel_daily` credits
  every channel that touched a converting journey, so its conversion column must
  sum to *more* than the true conversion count - 12,076 against 2,994. A test
  asserts exactly that. If it ever stops holding, someone has quietly turned a
  reach table into an attribution table, which is the most common analytics error
  in this domain.

The orchestrator is 120 lines and not Airflow on purpose: what a scheduler is
*for* at this size is dependencies, idempotency and failure semantics - a task
whose upstream failed is marked `skipped` and never runs on stale inputs, and a
task marked non-idempotent is never retried, because retrying an append silently
doubles it. Installing Airflow would demonstrate that Airflow installs.

> **The load step took 532 seconds** before it took 1.7. It inserted 160,000 rows
> with `executemany`, paying per-statement overhead on a columnar engine and
> getting none of the vectorisation DuckDB exists for. Registering a frame and
> doing one `INSERT … SELECT` is roughly three orders of magnitude faster.
> Reaching for the row-oriented API on a columnar store is the most common way a
> warehouse load ends up slower than the CSV it replaced.

**Honest limit:** DuckDB. The models, the graph and the tests are real dbt and
would run on Snowflake with a profile change. What is absent is everything about
a warehouse that is hard - concurrency, cost governance, permissions, incremental
strategies at scale.

## Orchestration: the 120-line runner vs Airflow

The section above says installing Airflow would "demonstrate that Airflow
installs." So here it is, ported and **run** rather than asserted. The same DAG
(`land_raw → dbt_build → read_marts`) is expressed as an Airflow DAG
(`airflow/dags/cas_pipeline.py`), kept **alongside** the hand-built runner in
`src/pipeline.py` - each Airflow task wraps the identical function the runner
calls, so it is a change of orchestrator, not of logic.

**It runs, measured:** `airflow dags test cas_pipeline` executed all three tasks
green and `read_marts` returned **customer_rfm = 4,899, customer_holdout = 2,592**
- the same marts the hand-built runner produces - with `dbt_build` taking ~54s.

**Getting it to run is itself the first finding.** Airflow is POSIX-only and
supports Python ≤3.12, so it will **not install on this Windows + Python 3.14 box**
(nor on the WSL Ubuntu, which is also on 3.14). It ran under a `uv`-provisioned
Python 3.12 venv inside WSL. The 120-line runner needs none of that: one file,
standard library only, runs anywhere Python does.

**What Airflow handles that the 120 lines didn't:**

- Scheduling & backfill (cron / data-intervals, catchup) - the runner is on-demand.
- Per-task retries with backoff (the runner had a flat count, only for idempotent tasks).
- A metadata DB + UI: task history, logs, durations, run states, and re-running a
  single failed task - the runner prints and forgets.
- Parallelism via pluggable executors (Local / Celery / Kubernetes) - the runner is sequential.
- Connections/hooks/pools, secrets, SLAs & alerting, XCom passing - none of which the runner models.

**What it added in complexity:**

- A whole runtime - metadata database + scheduler + (optionally) webserver: three
  services vs zero.
- A heavy, version-pinned dependency tree (installed via Airflow's official
  constraints file) and a hard OS / Python-version constraint that stopped it
  running natively here at all.
- Config surface (`AIRFLOW_HOME`, executor, DB) and operator boilerplate for a
  three-task DAG the runner expressed in ~120 lines.

**The honest read** matches the runner's original comment: at this size Airflow's
value is latent - you pay its operational cost now for scheduling, observability
and parallelism you do not yet need. The port earns its keep the day this pipeline
needs a schedule, a shared UI, or a cluster; until then the 120 lines are the right
tool - now shown both ways, not asserted.

```bash
# Airflow needs a Python <=3.12 interpreter (Linux / WSL / Docker); see the DAG docstring.
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
airflow db migrate && airflow dags test cas_pipeline 2025-01-01
```

## Running on Snowflake - a profile change, verified statically

The models run on DuckDB but are written as portable dbt. A `snowflake` target in
`dbt/profiles.yml` runs the **same** models, graph and tests on Snowflake with no
model edits (`dbt build --target snowflake`), after loading the raw tables with
`land_snowflake.py` - which reuses the same `raw_frames()` the DuckDB loader uses,
so both warehouses get byte-identical inputs.

**Verified without an account.** `check_snowflake_portability.py` transpiles every
compiled model from the DuckDB dialect to the Snowflake dialect with sqlglot:
**12/12 model files transpile with zero errors** (2 staging + 3 marts + 7 generated
test queries), and `tests/test_snowflake_portability.py` pins it. That is a static
guarantee that no DuckDB-only construct leaked into a model - turning the "would
run on Snowflake" claim into a check. It is not a live run.

**Not run live - honestly.** A Snowflake free trial needs an account (and card
verification) that can't be created here, so there are no live Snowflake numbers.
The exact path once an account exists:

```bash
export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
python land_snowflake.py                  # loads raw tables (reuses raw_frames)
cd dbt && dbt build --target snowflake    # same 5 models + 10 tests, on Snowflake
```

**DuckDB vs Snowflake vs Redshift - where they differ:**

| | DuckDB (this repo) | Snowflake | Redshift |
|---|---|---|---|
| runs as | in-process, one file | managed cloud service | managed cloud cluster |
| scaling | one machine's RAM/cores | elastic virtual warehouses; compute/storage separated | node cluster (RA3 separates storage) |
| concurrency | single process | multi-cluster, auto-suspend/resume | WLM queues + concurrency scaling |
| cost | free | per-second warehouse credits | per-hour nodes |
| dialect | near-standard | standard + rich semi-structured (VARIANT) | Postgres-derived; some function gaps |
| this data (8k customers, 160k rows) | ~0.1s (measured) | seconds + credits + network round-trips | similar + cluster overhead |

Only the DuckDB timing and the 12/12 transpile result are measured; the
Snowflake/Redshift cells are qualitative, not benchmarked (no accounts). The
honest read matches the Airflow one: at this scale a cloud warehouse buys nothing
the embedded engine doesn't already do faster and for free - Snowflake and
Redshift earn their keep on concurrency, governance, and data that outgrows one
machine. Portable dbt is what makes the switch a profile change rather than a
rewrite.

## The same transformations on PySpark - a documented negative result

The five dbt models are also implemented on **local Spark**
(`src/pipeline_spark.py`), kept **alongside** the dbt/DuckDB path, not replacing
it. The staging + mart logic is written as SQL strings that run **unchanged on
both engines**, so the port is provably the same transformation rather than a
lookalike - `tests/test_spark.py` asserts the Spark marts equal, row for row,
what dbt actually materialised into the warehouse (**75 tests pass**, up from 69:
the six new ones are the three dbt singular tests re-expressed on Spark output,
plus cross-engine parity). Getting there surfaced one real dialect trap: Spark
parses `1.0` as `DECIMAL`, so `avg(...)` silently rounded `discount_rate` to five
places and disagreed with DuckDB's `double` until the summands were cast to
`double`. The parity test is what caught it.

**Benchmark, same 89,540 transactions + 70,709 touches, one run on this machine
(2 cores, Arrow enabled):**

| path | startup | transform | peak memory |
|---|---|---|---|
| **DuckDB** (in-process, same SQL) | - | **0.10 s** | **141 MB** |
| **Spark** `local[*]` (Arrow) | 10.3 s (JVM) | 3.0 s warm · 11.7 s cold | 1,137 MB |
| dbt build (subprocess, orchestrated) | - | 16.8 s | 184 MB |

**DuckDB wins decisively at this scale, and that is the interesting finding.** Even
after the JVM is warm, Spark's transform is ~30× slower (3.0 s vs 0.10 s) and it
holds ~8× the memory (1.1 GB vs 141 MB); cold, you also pay ~10 s just to start
the JVM - per job. The reason is not that Spark is bad; it is that ~160k rows fit
in L3-cache-sized working sets where a vectorised single-node engine has no
coordination to do, and Spark's fixed costs (JVM, task scheduling, shuffle
plumbing, pandas↔JVM serialisation) are pure overhead with nothing to amortise
them against.

**Where the port would flip to Spark's favour:** when the data no longer fits in
one machine's memory (tens of GB+), when the transform must spill to disk, or when
it already lives in a cluster/lakehouse (S3 + Spark) and moving it out to a
single node would cost more than the compute. None of those hold here, so forcing
Spark would be cargo-culting - the honest artifact is the measurement that says
so. The value of the port is that the transformations translate cleanly to the
Spark SQL/DataFrame API and the crossover cost is now measured, not guessed.

```bash
# optional: needs a Java 17+ runtime; pyspark is not in requirements.txt
pip install pyspark
python bench_spark_vs_dbt.py      # runtimes + peak memory, asserts parity first
python -m pytest tests/test_spark.py -q
```

### The same PySpark path on Databricks

`databricks/cas_marts_databricks.py` runs the **identical** staging + mart SQL on
Databricks instead of local Spark - a notebook that copies the `pipeline_spark`
SQL verbatim and asserts the same parity (**4,899 / 2,592 / 303**, no leakage).
`databricks/export_raw_csv.py` writes the two raw CSVs to upload.

**What changes vs the local `spark_session()` path:**

- No `SparkSession` is created - Databricks provides `spark`; no JVM/JDK to install,
  no `local[*]` master, a managed cluster instead.
- Data is read from a **volume / DBFS** with `spark.read.csv`, not built from a
  pandas frame via `createDataFrame` - so no pandas↔JVM serialisation on ingest.
- The Databricks Runtime pins the Spark + Python versions (and may add Photon), vs
  the `uv`-provisioned local stack.

| path | transform | status |
|---|---|---|
| DuckDB (in-process) | **0.10 s** | measured |
| local Spark (`local[*]`, Arrow) | 3.0 s warm / 11.7 s cold | measured |
| Databricks (Free Edition, serverless) | **40.13 s** | measured |

On Databricks Free Edition serverless the same transform takes 40.13 s, with the
same parity (4,899 / 2,592 / 303, no leakage). It is the same work, not a
shortcut. That is about 13x slower than local Spark warm (3.0 s) and about 400x
slower than DuckDB (0.10 s). At 160k rows the platform's fixed costs are what you
pay for: acquiring a session, reading from the remote volume, planning the query,
moving data across the network. This is the local benchmark result again, one step
further out.

Databricks is not slow. It is built for data thousands of times larger, where
those fixed costs disappear into the run and a single machine cannot hold the data
at all. At this size it only adds latency. The figure is one serverless run and
will move around with how warm the session is, but the order of magnitude will
not. It is recorded, not estimated.

```bash
python databricks/export_raw_csv.py    # -> databricks/data/{transactions,touches}.csv
# upload both CSVs to a Databricks volume, import databricks/cas_marts_databricks.py,
# then Run All and read the printed transform wall-clock.
```

---

# Part 5 - Dashboards
