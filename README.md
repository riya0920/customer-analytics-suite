# DATA-1 — Customer Analytics: Segmentation → CLV → Attribution

**This is not deployable.** It is the first ~20% of the spec: three questions on
one dataset with the handoffs explicit, and — the differentiator — attribution
methods **validated against known ground truth** instead of asserted. Missing 80%
at the bottom.

```bash
python src/generate.py       # ~30s  transactions + journeys with planted effects
python run_analytics.py      # ~6min
python -m pytest tests -q    # 16 tests
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

## The other 80% — what is NOT here

- **The two-page executive memo** the spec asks for. There is a report, not a memo.
- **No dbt models, no marts, no orchestration.** The pipeline is three Python
  scripts.
- **Journeys are not linked to customer ids** — the single biggest structural gap.
  It means the CLV→attribution handoff is *reasoned about* and not computed: I
  cannot weight the budget objective by the predicted CLV of the customers each
  channel actually acquires, which is the whole point of joining the two.
- **No touch timestamps**, so time-decay is over journey *position*, not days.
- **No HDBSCAN**, no k selection procedure — k=5 is asserted, and only
  stability and forward-separation are measured, not optimality.
- **Markov is first-order only**; no higher-order paths, no Shapley comparison.
- **No CAC or ROAS**, so the budget table is conversions, not profit.
- **The generator is the model.** BG/NBD is fitted to a BG/NBD process and the
  attribution simulator has no unobserved confounders beyond the one I planted —
  so every method here performs better than it would on real data.
