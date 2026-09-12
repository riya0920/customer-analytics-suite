# DATA-1 Customer Analytics: Segmentation → CLV → Attribution

Three questions on one dataset with the handoffs
computed, attribution validated against known ground truth, **a real dbt pipeline
with a leakage test that fails the build**, k chosen rather than asserted,
**Shapley at twelve channels with its sampled approximation checked and then
repaired**, higher-order Markov, CAC and ROAS against incremental truth, and an
**unobserved confounder that no method here can beat**.

Three of the sections below report that something does not work. Two of those are
the most useful results in the project.

```bash
python src/generate.py       # ~5s    8,000 customers, 15,238 journeys, 12 channels
python run_analytics.py      # ~1min  the original report
python run_complete.py       # ~7min  the completion pass (dbt build + k sweep)
python -m pytest tests -q    # 69 tests
```

8,000 customers, 89,540 transactions, **15,238 journeys (1.9 per customer)**,
**12 channels**, 730 days.

## Published dashboards (Tableau Public + Looker Studio)

The same three result areas are published as an interactive dashboard on **two**
BI platforms - pick whichever a given job posting names.

**Tableau Public** (three tabs: Segmentation, CLV, Attribution; no sign-in to view):
https://public.tableau.com/app/profile/riya.ashokbhai.soni/viz/CustomerAnalytics-SegmentationCLVAttribution/Segmentation

Built in Tableau's browser web-authoring on the same exported CSVs: Segmentation =
customers by segment, CLV = share of predicted value by segment (Segment 0 = 37%,
Segment 4 = 51%), Attribution = the 7-method × 12-channel credit matrix with the
planted-truth row, so the methods' disagreement is visible cell by cell.

**Looker Studio** (unlisted - no sign-in required):
https://lookerstudio.google.com/reporting/97a07987-f61e-4930-9219-fa5d02c239cc

Three pages, one per result area, built on CSVs exported straight from the
pipeline:

- **Segmentation** - customers, churn rate, and share of predicted value by
  segment. The story the numbers tell: segment 0 is 7.6% of customers but 37% of
  value at 31% churn; segments 1 and 3 are 45% of customers holding under 4% of
  value at ~93% churn.
- **CLV** - predicted-CLV index by acquisition channel (which channels bring
  higher-value customers; spread 0.94-1.08).
- **Attribution** - each method's credit split across the 12 channels, so the
  disagreement between methods (and against planted truth) is visible at a
  glance.

The data is exported by `export_bi.py`, which reads the pipeline's own metric
JSON and recomputes the customer-level frames with the **same** `src.clv`
functions the report uses - and **asserts** its recomputed segment sizes match
the report before writing, so the dashboard cannot silently drift from the
analysis:

```bash
python export_bi.py          # writes bi_export/*.csv (17 tidy files)
```

Looker Studio was built first because it reaches a public link fastest (browser-
only, native link-sharing); the Tableau Public version was added so the exact
"Tableau" keyword is covered too. Both read the same `bi_export/` CSVs, so every
figure on either dashboard is something the code produced - no estimates.

**Power BI** — build steps in [`docs/POWERBI.md`](docs/POWERBI.md), over the same
`bi_export/` CSVs. Public link: _pending publish_ — the report builds in Power BI
Desktop (free), but a public "Publish to web" URL needs a Power BI **service**
account, which requires a work/school email (consumer Gmail cannot register). The
build and publish steps are documented; the link goes here once published. Why
Tableau/Looker were done first: both reach a free public URL from any account in
the browser, which Power BI's public path does not.

For a code-native, offline view of the same attribution result:

```bash
python -m src.plotly_view     # writes out/attribution_plotly.html (self-contained Plotly)
```

`src.plotly_view` recomputes the credited share each method assigns per channel
(the same `src.attribution` functions the report uses) and renders a grouped bar
of every method against the TRUTH row, annotating the planted zero-effect channel
that every observational method over-credits. The HTML bundles Plotly inline, so
it opens with no server and no network.

## A pipeline, not three scripts

```
ok  land_raw     1.7s     ok  dbt_build  31.3s     ok  read_marts  0.2s
dbt: PASS=15 WARN=0 ERROR=0 SKIP=0 TOTAL=15
marts: customer_rfm 7,894 rows · customer_holdout 2,847 rows
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
  sum to *more* than the true conversion count - 20,344 against 4,971. A test
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
green and `read_marts` returned **customer_rfm = 7,894, customer_holdout = 2,847**
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
SQL verbatim and asserts the same parity (**7,894 / 2,847 / 303**, no leakage).
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
same parity (7,894 / 2,847 / 303, no leakage). It is the same work, not a
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

## Choosing k - six criteria, five answers

| criterion | k |
|---|---|
| silhouette (max) | 3 |
| Calinski-Harabasz (max) | 2 |
| Davies-Bouldin (min) | 8 |
| elbow (inertia knee, computed) | 4 |
| stability (max ARI over bootstraps) | 2 |
| **forward separation (max adjusted η²)** | **6** |

They disagree for principled reasons. Silhouette rewards compact spheres.
Stability rewards *coarse* partitions - k=2 is stable on almost any data because
there is little to disagree about. Forward separation rewards whatever correlates
with the outcome.

**The one that should decide is forward separation**, because it is the only
criterion tied to what the segments are *for*. The rest measure whether the
geometry is tidy, which is a question nobody in the business asked. And it is
reported **adjusted**, because raw η² rises with k mechanically and an unadjusted
table always recommends the largest k on offer.

### HDBSCAN, swept rather than asserted

| min_cluster_size | clusters | noise share | largest cluster |
|---|---|---|---|
| 25 | 8 | 0.849 | 362 |
| 50 | 2 | **0.556** | 3,228 |
| 100 | 2 | 0.691 | 2,328 |
| 200 | 0 | 1.000 | 0 |

**HDBSCAN leaves the majority unassigned at every setting tried**, and that is a
statement about this customer base rather than about the algorithm: RFM features
on a retail panel are one diffuse cloud with a thin high-value tail, not a set of
dense islands. There is no density structure to find, so a density method
correctly finds none - and a marketing team handed a clustering that covers 44% of
customers will go back to k-means by the end of the week.

**The comparison is not "which is better".** k-means forces a partition and is
therefore always actionable and sometimes fictional; it will cheerfully cut a
single cloud into five wedges and name them. HDBSCAN refuses to invent structure
and is therefore sometimes honest and often unusable. The useful output of running
both is knowing which one you are buying - here, k-means is inventing the
segments, and that is worth knowing before anyone builds a campaign on them.

## Shapley at twelve channels - and the check that failed

Exact Shapley over 12 channels is 4,096 coalitions: large enough to be interesting,
small enough that the exact answer still exists. So the sampled estimator can be
scored against the thing it approximates.

| permutations | mean abs error |
|---|---|
| 25 | 0.0713 |
| 100 | 0.0730 |
| 400 | 0.0596 |
| 800 | **0.0572** |

**32× more permutations should cut Monte-Carlo error by 5.66×. Measured: 1.25×.**

It does not converge. The error falls a little and plateaus, which means the gap
is **bias, not variance** - the sampler is converging to a different number, not
noisily to the same one.

The cause is that only **2,519 of 4,096 coalitions were ever observed**. A
permutation walks the lattice one channel at a time and cannot advance when the
next coalition was never seen, so permutations *stall*, and they stall more often
for channels appearing in rare combinations. The exact estimator has the same
missing data but reweights by the coalitions it did use; the sampler cannot,
because it never learns which ones it skipped.

**So they are not the same estimator at two sample sizes - they are different
estimators.** Sampled Shapley on a sparse coalition lattice needs a different
value function, not more permutations. This is precisely the check the usual
justification for sampling skips: "the exact version is intractable" is true at 30
channels and is also the regime where nobody can discover this.

The stall is now **counted rather than inferred**: the exact-set sampler stalls on
**1,828 of 4,800 permutation steps (38.1%)**. More than a third of every walk
lands on a coalition nobody was ever exposed to.

## The fix - two value functions, and they are not the same fix

The previous pass stopped at the diagnosis. This is the repair.

### Fix 1 - a value function defined on the whole lattice

`v(S)` = the conversion rate among journeys whose channel set is a **subset** of
S: *"what is achievable using only the channels in S"*. The old one asks *"what
happened to the customers who saw exactly this combination and nothing else"* - a
question about a rarer and rarer group as the coalition grows, and undefined once
that group is empty.

| | coalitions defined |
|---|---|
| exact-set | 2,519 of 4,096 (61.5%) |
| **subset-closure** | **4,095 of 4,096 (100.0%)** |

The one undefined coalition is the empty set, and that is correct: no channels is
no marketing, and the rate is zero by *definition* rather than by missing data.

Same estimator, same permutations, same seed logic - only the game being sampled
changed:

| permutations | exact-set error | closure error |
|---|---|---|
| 25 | 0.08459 | 0.02425 |
| 100 | 0.06154 | 0.01172 |
| 400 | 0.05393 | 0.00616 |
| 800 | **0.05034** (plateau) | **0.00458** |

**32× the permutations should cut a purely noisy error 5.66×. The exact-set
version manages 1.68×; the closure version manages 5.30×.** It converges, and the
stall count is zero by construction because there is no rung to fall off.

**Efficiency residual: −5.55e−17** against a grand-coalition value of 0.3262. The
credits add up to the thing being attributed, to machine precision - a check that
was not available for the exact-set version at all, whose grand coalition is
estimated from whichever handful of customers happened to see all twelve channels.

> **The number of seeds needed to measure a convergence rate is itself something
> that has to be checked.** One seed read **8.10×** and was non-monotone; six
> seeds read **6.44×**. Both are *above* the 5.66× ceiling that 1/√n sets - which
> is not a fast estimator, it is an unconverged measurement *of* an estimator. It
> took twelve seeds to settle underneath the ceiling where it belongs. The same
> mistake this section exists to catch, one level up.

### Fix 2 - Shapley inside each journey

Not the same fix. It changes what the lattice **is**: a journey with five touches
has 32 sub-coalitions whether the catalogue holds 12 channels or 300.

```
distinct channel sets      : 2,519
most channels in a journey : 9
sub-coalitions per journey : 512 at that maximum
marginal evaluations       : 454,200, computed EXACTLY
```

No sampling at all. **The intractability that justified sampling was a property of
the value function, not of the problem.**

> **The test caught this claim being false in my own implementation.** The first
> version called the dense builder for its value function, which materialises
> 2^n. At 12 channels that is 4,096 and invisible; the test that runs it at 30
> channels asked for **8 GiB**. The estimator whose entire claim is that it does
> not depend on channel count was depending on channel count - and the claim sat
> in the docstring through a full clean run before a test disagreed with it. It
> now builds one zeta transform per *journey*, over that journey's own bits, and
> a test pins that the local answer equals the global one to 1e−12.

### Against planted truth

| method | MAE vs truth |
|---|---|
| **per-journey** | **0.0247** |
| closure | 0.0268 |
| exact-set | 0.0292 |
| sampled (old) | 0.0466 |

Both fixes beat what they replaced, but they are not two approximations of one
number. They answer two different questions. Closure asks what a channel adds to
what is achievable. Per-journey asks how each observed journey's outcome divides
among the touches that were in it. Nothing makes them agree, and reporting
whichever scored better without saying they measure different things would be
picking an estimand by leaderboard.

And the zero-effect channel is still credited **0.0812** under closure and
**0.0690** per journey. **Fixing the estimator does not fix the data** - the same
conclusion the confounder section reaches from the other direction.

## Higher-order Markov - every channel gets exactly zero

| order | states | thin-state share | max removal effect | channels with zero credit |
|---|---|---|---|---|
| 1 | 13 | 0.000 | 0.000089 | **12 / 12** |
| 2 | 152 | 0.092 | 0.000010 | **12 / 12** |
| 3 | 1,354 | 0.287 | 0.000000 | **12 / 12** |

**That is not a bug, and it is the most useful thing in the section.**

This implementation removes a channel by **deleting the touch from the journeys**
and re-estimating - the counterfactual a marketer means by "what if we turned it
off". Done that way the conversion probability does not move, because in
observational path data the outcome is attached to the **journey**, not to the
path: a journey that converted still converted with one touch removed.

The textbook removal effect avoids that by deleting the **node from the graph** and
renormalising, which strands the removed node's inbound probability mass in the
null state. That produces a satisfying non-zero number - `markov_removal` scores
0.1082 MAE with it below - and the number comes from the graph representation
rather than from anything about the channel.

The two implementations disagree completely, and **the one that returns zeros is
the one being honest.** "Remove the channel from the graph" was never a causal
statement; this is what it looks like when you write down the counterfactual it
claims to compute and then actually compute it. Both are kept, and a test pins
each.

## CAC and ROAS - and why channel-level ones never reconcile

| channel | spend | observational CAC | **incremental CAC** | observational ROAS | incremental ROAS |
|---|---|---|---|---|---|
| paid_search | $2,984.80 | $1.18 | $2.92 | 119.75 | 133.18 |
| shopping_feed | $979.38 | $0.58 | $1.57 | 238.70 | 248.04 |
| affiliate | $484.47 | $0.30 | $1.42 | 462.53 | 273.51 |
| **retargeting** | $298.50 | **$0.16** | **∞** | **861.66** | **0.00** |

Blended CAC is $1.27. The channels' `conversions_touched` sum to **20,344 against
4,971 actual conversions**, because every conversion touched by four channels is
counted four times. That is what every channel-level CAC in every marketing deck
is, and it is why the numbers never reconcile to the blended figure.

**Look at `retargeting`**: a channel that causes nothing has a defensible $0.16
CAC and an 861× ROAS, and would survive any efficiency review. Its incremental CAC
is infinite. That gap is the business case for the experiment, denominated in
dollars rather than in credit shares.

## The unobserved confounder - why no method here can win

The previous README ended by admitting *"the attribution simulator has no
unobserved confounders beyond the one I planted - so every method here performs
better than it would on real data."* That is now false by construction.

`in_market` is a latent state affecting **32%** of customers. It raises conversion
probability by 0.16 **and** multiplies exposure to closing channels by 2.4×. It is
never written to disk.

| method | MAE vs truth | credit to the zero-effect channel |
|---|---|---|
| shapley | **0.0292** | 0.1065 |
| linear | 0.0344 | 0.0900 |
| time_decay | 0.0348 | 0.1243 |
| shapley_sampled | 0.0466 | 0.1257 |
| last_touch | 0.0620 | 0.1996 |
| first_touch | 0.0772 | 0.0163 |
| markov_removal | 0.1082 | 0.1166 |

**Every method credits the zero-effect channel, and every method has non-trivial
error.** The distinction from the planted retargeting confound matters: that one
is observable *in principle* - propensity is a customer attribute a good model
could proxy. This one is not, and no attribution system in the world can condition
on it: it is the thing the customer knows and the ad server does not.

The useful reading is not the ranking. It is that the ranking is now a comparison
of **how each method fails** rather than a search for one that succeeds - with an
unobserved common cause of exposure and outcome, none of them *can*. That is a
theorem, not a limitation of these implementations, and it is why the geo holdout
is the only instrument that answers the question at all.

## Uplift - who to target, graded against a known CATE

Average effects tell you whether to run a campaign; **uplift** tells you *who to
target*, and that is where a budget actually moves. `python -m src.uplift` plants
a treatment whose effect **varies** across customers - high-intent buyers get a
large positive lift, and a low-intent **"sleeping dogs"** segment is actively
hurt - then grades a T-learner and an S-learner against the known per-customer
`tau(x)`. The average effect is ~0 by design, so the whole value is in the
heterogeneity: the oracle that ranks by the true effect earns **+0.057** mean
true-tau in its top decile against **+0.001** for random targeting.

| learner | Spearman(pred, true tau) | Qini (fraction of oracle) | top-decile true tau: model / random / oracle |
| --- | --- | --- | --- |
| T-learner | 0.205 | -0.640 | +0.0224 / +0.0013 / +0.0573 |
| S-learner | 0.675 | 0.665 | +0.0330 / +0.0013 / +0.0573 |

The honest result is that **the S-learner beats the T-learner here**, which cuts
against the usual "T-learner is the flexible default for heterogeneity" heuristic.
With a near-zero base signal the T-learner differences two independently-fit
models whose errors do not cancel, and its ranking ends up *worse than random* on
Qini (-0.640) even though its very top decile still beats random. The S-learner,
carrying treatment as a feature, captures two-thirds of the oracle's Qini. Qini
is reported as a fraction of the oracle because the raw Qini coefficient is
ill-conditioned when the ATE is ~0 (its usual normaliser goes to zero) - and
individual treatment effects are noisy even under randomisation, so the rank
correlation is moderate, not near-1. That ceiling is reported, not tuned past.

## What is deliberately not here

- **Real touch data with ground truth does not exist and cannot**, because the
  ground truth is a causal quantity. That is why the generator is the point rather
  than an apology.
- **One journey per customer is fixed; repeat *acquisition* is not modelled.**
  Customers now have several journeys, but a customer who churns and is re-won is
  not represented as such.
- **No CUPED, no synthetic control, no geo experiment** - the report argues for
  one and does not run it; DATA-3 is where designs live.
- **The dbt project is five models.** No incremental materialisations, no
  snapshots, no exposures, no docs site.
- **Neither Shapley fix is causal**, and neither claims to be. Both are exact
  allocations of an *observational* quantity; the confounder section below is
  what says why that quantity is not the one anybody wants.
- **The generator is still a model.** BG/NBD is fitted to a BG/NBD process, and
  the confounder is one I chose - a real system has many, correlated, and none of
  them documented.

## Spec coverage

What the original brief asked for, and whether it is in this project:

| Asked for | In this project? |
|---|---|
| Use a real transactional dataset with a simulated multi-channel touch layer that has known channel effects | Partial - the touch layer is simulated with known effects as required; the transaction base is also simulated rather than Olist/UCI |
| RFM plus behavioral clustering, with bootstrap-stability validation | Yes |
| Prove segments actually predict future behavior (forward-test into the next period) | Yes |
| Probabilistic CLV (BG/NBD + Gamma-Gamma) validated on a temporal holdout, plus a GBM challenger | Yes |
| CLV deciles wired back into segments (the value-concentration curve) | Yes |
| All five attribution methods including Markov, each scored against the simulator's known truth | Yes |
| An incrementality demo: plant a channel that causes nothing, show every method over-credit it, and sketch the experiment that could settle it | Yes |
| A 2-page executive memo plus a reproducible dbt pipeline | Yes |
