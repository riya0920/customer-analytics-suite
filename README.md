# DATA-1 — Customer Analytics: Segmentation → CLV → Attribution

**Roughly 50% of the spec.** Three questions on one dataset with the handoffs
explicit, attribution **validated against known ground truth** - plus the four
things the first pass named as missing: the **customer link** it called its single
biggest structural gap, touch timestamps, Shapley attribution, and the executive
memo.

```bash
python src/generate.py       # ~40s  transactions + journeys, now with customer ids
python run_analytics.py      # ~1min
python -m pytest tests -q    # 29 tests
```

8,000 customers, 90,092 transactions, 730 days (511 calibration / 219 holdout).

## Why the touch data is simulated

Real multi-touch attribution data with ground truth does not exist publicly, and
it **cannot** — the ground truth is a causal quantity, so establishing it takes an
experiment nobody publishes. Without truth, attribution methods can only be
asserted, which is exactly why the field is full of confident numbers that
misallocate budget.

So channel effects are chosen and every method is **scored** against them. Three
things are planted: known incremental effects per channel; position bias (display
and social open journeys, email and search close them, so first- and last-touch
have systematically *opposite* biases); and a **zero-effect retargeting channel**
targeted at users who were going to convert anyway.

## Segmentation — and whether it predicts anything

Bootstrap stability over 20 resamples: **ARI 0.939** (min 0.895).

| segment | n | cal frequency | recency | **churn T+1** | holdout spend |
|---|---|---|---|---|---|
| 1 | 790 | 36.1 | 67 | **36.7%** | $600 |
| 2 | 548 | 0.7 | 95 | 37.8% | $68 |
| 3 | 3,083 | 8.7 | 115 | 41.3% | $184 |
| 0 | 1,357 | 3.1 | 404 | 91.3% | $11 |
| 4 | 2,222 | 3.2 | 426 | **94.6%** | $7 |

**57.9 points** of churn-rate spread and $7→$600 of holdout spend. Segments that
don't separate *future* behaviour are decoration — you can always partition a
cloud of points and name the parts. The only way to know is to hold out time and
look.

Operationally that spread is a **targeting prior, not a playbook**: it says where
retention budget has headroom, not what intervention works — that needs a test
per segment. And acting on a segment *changes* it, so these are pre-intervention
baselines that stop being true the moment anyone uses them.

## CLV — and a caveat on the caveat

BG/NBD (r=0.607, α=15.50, a=1.273, b=15.12) and Gamma-Gamma (p=2.976, q=6.519,
v=135.0), both written out from scratch since `lifetimes` isn't installed —
which forces the assumptions into view.

Gamma-Gamma assumes value is independent of frequency. **Tested, not assumed:**
measured correlation **+0.0035** — holds here.

Decile calibration on the temporal holdout:

| decile | predicted | actual | ratio |
|---|---|---|---|
| 5 | 43.3 | 51.8 | 0.84 |
| 7 | 169.1 | 164.0 | 1.03 |
| 9 | 791.4 | 804.5 | 0.98 |

The textbook warning is that these models rank populations well and mispredict
individuals. **That is not what this run shows** — the individual correlation is
0.80. Reporting that as evidence the model predicts individuals well would be the
most misleading thing in this project, because the reason is circular: my
generator draws inter-purchase times from an exponential with a Gamma-distributed
rate and applies a Beta dropout after each purchase. **That *is* the BG/NBD
process.** The model is scored on data satisfying its assumptions exactly.

What this section actually validates is that the estimator recovers parameters
and the calibration harness works — worth knowing, and not the same claim. A test
fits BG/NBD to data drawn with known parameters and asserts recovery.

| model | Spearman | MAE |
|---|---|---|
| BG/NBD + Gamma-Gamma | **0.739** | **88.6** |
| GBM challenger | 0.714 | 95.3 |

Recommendation is the probabilistic model, and not because it won every number:
it gives P(alive) and a purchase-count distribution rather than a point estimate,
extrapolates to horizons it never saw, and has four arguable parameters. The GBM
needs a labelled future, so it can only predict horizons you've already lived
through — and silently relearns whatever selection is in the label window.

## Attribution, scored against truth

| method | display | social | email | paid_search | **retargeting** | MAE |
|---|---|---|---|---|---|---|
| **TRUTH** | 0.089 | 0.200 | 0.311 | 0.400 | **0.000** | — |
| last_touch | 0.027 | 0.046 | 0.230 | 0.340 | **0.358** | 0.143 |
| first_touch | **0.647** | 0.250 | 0.058 | 0.030 | 0.016 | 0.249 |
| linear | 0.263 | 0.210 | 0.211 | 0.169 | 0.146 | 0.132 |
| time_decay | 0.157 | 0.155 | 0.243 | 0.232 | 0.213 | 0.113 |
| markov_removal | 0.000 | 0.203 | 0.277 | 0.338 | 0.183 | **0.074** |

Last-touch over-credits closers, first-touch over-credits openers (display 0.647
against a truth of 0.089) — systematically, because the generator gave channels
position bias.

## The planted channel

`retargeting` has a true causal effect of **exactly zero**.

```
conversion rate WITH retargeting    : 0.2832
conversion rate WITHOUT retargeting : 0.2346
apparent lift                       : +0.0486   (TRUE lift: 0.0000)
```

**Every method credits it** — last-touch hands it 35.8% of all credit; even
`markov_removal`, which sounds causal, gives it 18.3%. "Remove the channel from
the graph" is a statement about observed paths, not about the world: it assumes
the users who saw that channel would have walked the same graph minus one node.
When the channel was *targeted* at high-intent users, removing it also removes
the intent that arrived with it, and the method charges that intent to the
channel.

**The experiment that would settle it**, sized: geo holdout, matched markets,
retargeting off in control. Geo rather than user-level because cookie holdouts
leak across devices and the ad platform optimises delivery against the holdout,
breaking randomisation. Cost is the foregone spend in control markets — which, if
the effect really is zero, is not a cost at all. That asymmetry is the pitch.

## The budget decision, and a result that complicates it

$1,500 (~$0.19/prospect) allocated in proportion to credited share, evaluated in
the true world with concave reach:

| allocation | conversions | spend on zero-effect channel | lost vs truth |
|---|---|---|---|
| TRUTH | 2,165.7 | $0 | — |
| time_decay | 2,086.0 | $319 | 79.8 (3.7%) |
| last_touch | 2,072.1 | **$537 (35.8%)** | 93.6 (4.3%) |
| linear | 2,050.6 | $220 | 115.1 (5.3%) |
| markov_removal | 1,978.0 | $274 | 187.7 (8.7%) |
| first_touch | 1,939.8 | $24 | 225.9 (10.4%) |

Last-touch costs 94 conversions and puts **35.8% of the budget into a channel
that causes nothing**.

**But rank by attribution error and by budget outcome and the orders disagree.**
`markov_removal` has the *lowest* MAE (0.074) and the *second-worst* budget
outcome (−8.7%); `time_decay` is middling on MAE and best on decisions. Getting
credit shares closest to truth is not the same as making the best decision,
because the decision runs those shares through channel costs and a concave reach
curve — a method can be wrong in a direction that happens to be cheap.
**Choosing an attribution method on MAE alone optimises the wrong objective.**

> The first version of this section used a $1m budget, which saturated every
> channel's reach and made all six allocations tie at exactly 2,600 conversions.
> A section where every method wins equally is a broken section, not a finding.

## Second pass: four gaps the first pass named

### The CLV handoff - computed, not reasoned about

The first pass called this *the single biggest structural gap*: journeys weren't
linked to customer ids, so the budget objective couldn't be weighted by the value
of the customers a channel acquires - which is the whole point of computing CLV
in the same project. The link exists now.

| channel | converters touched | mean CLV | **CLV index** |
|---|---|---|---|
| display | 1,534 | $150.96 | **1.040** |
| social | 1,341 | $147.20 | 1.010 |
| retargeting | 1,091 | $143.37 | 0.990 |
| paid_search | 1,181 | $142.98 | 0.980 |
| email | 1,324 | $141.43 | 0.974 |

Scoring the same allocations two ways - conversions, and conversions weighted by
acquired-customer CLV - gives **the same winner**. And that's reported rather than
hidden: the CLV spread across channels is only 0.066, because this simulator
doesn't correlate channel with customer value. The join has nothing to bite on
*here*, which is a property of the lab, not a general result. On real data the
cheap acquisition channels are usually the low-value ones, which is exactly when
this weighting earns its keep.

**Honest limit on the weighting itself:** `clv_index` is correlational. A channel
above 1.0 may be *acquiring* better customers, or may simply be *touching*
customers who were already valuable - the same confound the retargeting section is
about. Weighting a budget by it inherits that confound, which makes the case for
the experiment **larger**, not smaller.

### Shapley - and the strongest form of the central finding

Shapley has a formal **dummy-player axiom**: a channel that changes no
coalition's conversion rate is guaranteed exactly zero credit. It's the only
method here with that property, and a test proves it holds when a dummy channel
is added *at random*.

On the planted data, **Shapley gives retargeting 0.168**.

The axiom isn't violated - it's satisfied, on a coalition function that is itself
confounded. Retargeting genuinely *does* raise the observed conversion rate of
every coalition it joins, because it joins the coalitions of customers who were
going to convert. **Shapley answers its question correctly; the question is the
wrong one.** No amount of methodological sophistication fixes data that contains
no variation in whether the channel ran.

Shapley is also the **most accurate** method (MAE 0.074) and the best-performing
allocation among the heuristics (2.2% of conversions lost vs truth).

### A correction the regenerated data forced

The first pass said *"every method credits it, including markov_removal"*. After
regeneration that became false - `markov_removal` reads **0.0000** for
retargeting. It is not detecting the confound:

```
markov_removal   display 0.0000  social 0.0802  email 0.2810  paid_search 0.6389  retargeting 0.0000
TRUTH            display 0.0889  social 0.2000  email 0.3111  paid_search 0.4000  retargeting 0.0000
```

It has collapsed the credit onto one channel and **zeroed display too**, whose
true share is 0.089. That's a known brittleness of removal effects: a channel
whose removal doesn't *disconnect* the graph scores zero regardless of
contribution. It happens to be right about retargeting and wrong about display for
the same reason. **A method that is accidentally right is not a method you can
deploy, because you cannot tell in advance which of its zeros are correct.**

`first_touch` also gives it little (0.016) - retargeting is a *closer*, so a
first-touch model barely sees it. That's the mirror image of the position bias
that makes last-touch over-credit it, not evidence that first-touch is
unconfounded. Both facts are now asserted in the test, which previously claimed
"every method" and was asserting an accident.

### Time decay over days, and the memo

Time decay now decays over **days** rather than journey position - the generator
carries touch timestamps. Position decay treats a touch three steps back the same
whether it was yesterday or three weeks ago, and those are different claims.

The **executive memo** the spec asks for is written to `out/EXECUTIVE_MEMO.md`:
recommendation, the geo-holdout pitch with its asymmetry argument (the experiment
is cheapest in exactly the world where the channel is worthless), what the
analysis is based on, an explicit *what we are not claiming* section, and the cost
of doing nothing.

## The other ~50% - what is still NOT here

- **No dbt models, no marts, no orchestration.** The pipeline is three Python
  scripts.
- **One journey per customer.** The link exists but the simulator gives each
  customer a single journey, so it cannot model repeat acquisition or the same
  customer being touched across campaigns.
- **The CLV weighting is correlational** (see above) and inherits the confound
  it is meant to help price.
- **No HDBSCAN**, no k selection procedure - k=5 is asserted, and only stability
  and forward-separation are measured, not optimality.
- **Markov is first-order only**; no higher-order path models.
- **Shapley is exact over 5 channels.** At 30 channels the 2^30 coalitions need
  sampling, which is not implemented.
- **No CAC or ROAS**, so the budget table is conversions and customer value, not
  profit.
- **The generator is the model.** BG/NBD is fitted to a BG/NBD process and the
  attribution simulator has no unobserved confounders beyond the one I planted —
  so every method here performs better than it would on real data.
