# DATA-1 — Customer Analytics: Segmentation → CLV → Attribution

**Complete against the spec.** Three questions on one dataset with the handoffs
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
  sum to *more* than the true conversion count — 20,344 against 4,971. A test
  asserts exactly that. If it ever stops holding, someone has quietly turned a
  reach table into an attribution table, which is the most common analytics error
  in this domain.

The orchestrator is 120 lines and not Airflow on purpose: what a scheduler is
*for* at this size is dependencies, idempotency and failure semantics — a task
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
a warehouse that is hard — concurrency, cost governance, permissions, incremental
strategies at scale.

## Choosing k — six criteria, five answers

| criterion | k |
|---|---|
| silhouette (max) | 3 |
| Calinski-Harabasz (max) | 2 |
| Davies-Bouldin (min) | 8 |
| elbow (inertia knee, computed) | 4 |
| stability (max ARI over bootstraps) | 2 |
| **forward separation (max adjusted η²)** | **6** |

They disagree for principled reasons. Silhouette rewards compact spheres.
Stability rewards *coarse* partitions — k=2 is stable on almost any data because
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
correctly finds none — and a marketing team handed a clustering that covers 44% of
customers will go back to k-means by the end of the week.

**The comparison is not "which is better".** k-means forces a partition and is
therefore always actionable and sometimes fictional; it will cheerfully cut a
single cloud into five wedges and name them. HDBSCAN refuses to invent structure
and is therefore sometimes honest and often unusable. The useful output of running
both is knowing which one you are buying — here, k-means is inventing the
segments, and that is worth knowing before anyone builds a campaign on them.

## Shapley at twelve channels — and the check that failed

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
is **bias, not variance** — the sampler is converging to a different number, not
noisily to the same one.

The cause is that only **2,519 of 4,096 coalitions were ever observed**. A
permutation walks the lattice one channel at a time and cannot advance when the
next coalition was never seen, so permutations *stall*, and they stall more often
for channels appearing in rare combinations. The exact estimator has the same
missing data but reweights by the coalitions it did use; the sampler cannot,
because it never learns which ones it skipped.

**So they are not the same estimator at two sample sizes — they are different
estimators.** Sampled Shapley on a sparse coalition lattice needs a different
value function, not more permutations. This is precisely the check the usual
justification for sampling skips: "the exact version is intractable" is true at 30
channels and is also the regime where nobody can discover this.

The stall is now **counted rather than inferred**: the exact-set sampler stalls on
**1,828 of 4,800 permutation steps (38.1%)**. More than a third of every walk
lands on a coalition nobody was ever exposed to.

## The fix — two value functions, and they are not the same fix

The previous pass stopped at the diagnosis. This is the repair.

### Fix 1 — a value function defined on the whole lattice

`v(S)` = the conversion rate among journeys whose channel set is a **subset** of
S: *"what is achievable using only the channels in S"*. The old one asks *"what
happened to the customers who saw exactly this combination and nothing else"* — a
question about a rarer and rarer group as the coalition grows, and undefined once
that group is empty.

| | coalitions defined |
|---|---|
| exact-set | 2,519 of 4,096 (61.5%) |
| **subset-closure** | **4,095 of 4,096 (100.0%)** |

The one undefined coalition is the empty set, and that is correct: no channels is
no marketing, and the rate is zero by *definition* rather than by missing data.

Same estimator, same permutations, same seed logic — only the game being sampled
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
credits add up to the thing being attributed, to machine precision — a check that
was not available for the exact-set version at all, whose grand coalition is
estimated from whichever handful of customers happened to see all twelve channels.

> **The number of seeds needed to measure a convergence rate is itself something
> that has to be checked.** One seed read **8.10×** and was non-monotone; six
> seeds read **6.44×**. Both are *above* the 5.66× ceiling that 1/√n sets — which
> is not a fast estimator, it is an unconverged measurement *of* an estimator. It
> took twelve seeds to settle underneath the ceiling where it belongs. The same
> mistake this section exists to catch, one level up.

### Fix 2 — Shapley inside each journey

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
> not depend on channel count was depending on channel count — and the claim sat
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

**Read that carefully.** Both fixes beat what they replaced, but they are not two
approximations of one number — they are two different questions. Closure asks what
a channel adds to what is *achievable*; per-journey asks how each observed
journey's outcome divides among the touches that were in it. Nothing makes them
agree, and reporting whichever scored better without saying they measure different
things would be **picking an estimand by leaderboard**.

And the zero-effect channel is still credited **0.0812** under closure and
**0.0690** per journey. **Fixing the estimator does not fix the data** — the same
conclusion the confounder section reaches from the other direction.

## Higher-order Markov — every channel gets exactly zero

| order | states | thin-state share | max removal effect | channels with zero credit |
|---|---|---|---|---|
| 1 | 13 | 0.000 | 0.000089 | **12 / 12** |
| 2 | 152 | 0.092 | 0.000010 | **12 / 12** |
| 3 | 1,354 | 0.287 | 0.000000 | **12 / 12** |

**That is not a bug, and it is the most useful thing in the section.**

This implementation removes a channel by **deleting the touch from the journeys**
and re-estimating — the counterfactual a marketer means by "what if we turned it
off". Done that way the conversion probability does not move, because in
observational path data the outcome is attached to the **journey**, not to the
path: a journey that converted still converted with one touch removed.

The textbook removal effect avoids that by deleting the **node from the graph** and
renormalising, which strands the removed node's inbound probability mass in the
null state. That produces a satisfying non-zero number — `markov_removal` scores
0.1082 MAE with it below — and the number comes from the graph representation
rather than from anything about the channel.

The two implementations disagree completely, and **the one that returns zeros is
the one being honest.** "Remove the channel from the graph" was never a causal
statement; this is what it looks like when you write down the counterfactual it
claims to compute and then actually compute it. Both are kept, and a test pins
each.

## CAC and ROAS — and why channel-level ones never reconcile

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

## The unobserved confounder — why no method here can win

The previous README ended by admitting *"the attribution simulator has no
unobserved confounders beyond the one I planted — so every method here performs
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
is observable *in principle* — propensity is a customer attribute a good model
could proxy. This one is not, and no attribution system in the world can condition
on it: it is the thing the customer knows and the ad server does not.

The useful reading is not the ranking. It is that the ranking is now a comparison
of **how each method fails** rather than a search for one that succeeds — with an
unobserved common cause of exposure and outcome, none of them *can*. That is a
theorem, not a limitation of these implementations, and it is why the geo holdout
is the only instrument that answers the question at all.

## What is deliberately not here

- **Real touch data with ground truth does not exist and cannot**, because the
  ground truth is a causal quantity. That is why the generator is the point rather
  than an apology.
- **One journey per customer is fixed; repeat *acquisition* is not modelled.**
  Customers now have several journeys, but a customer who churns and is re-won is
  not represented as such.
- **No CUPED, no synthetic control, no geo experiment** — the report argues for
  one and does not run it; DATA-3 is where designs live.
- **The dbt project is five models.** No incremental materialisations, no
  snapshots, no exposures, no docs site.
- **Neither Shapley fix is causal**, and neither claims to be. Both are exact
  allocations of an *observational* quantity; the confounder section below is
  what says why that quantity is not the one anybody wants.
- **The generator is still a model.** BG/NBD is fitted to a BG/NBD process, and
  the confounder is one I chose — a real system has many, correlated, and none of
  them documented.
