# Customer Analytics: Segmentation, Lifetime Value and Marketing Attribution

**Live dashboards:** [Tableau Public](https://public.tableau.com/app/profile/riya.ashokbhai.soni/viz/CustomerAnalytics-SegmentationCLVAttribution/Segmentation) · [Looker Studio](https://lookerstudio.google.com/reporting/97a07987-f61e-4930-9219-fa5d02c239cc) · [Executive memo](out/EXECUTIVE_MEMO.md)

Who are this store's customers, what is each one worth, and which marketing
channels actually bring in sales?

## What we did

We took two years of real transactions from a UK online gift retailer
([UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii),
also on Kaggle) and answered three questions a marketing team asks:

1. **Who are our customers?** Group them into segments, and check that the
   segments predict what customers do next.
2. **What is each customer worth?** Predict each customer's future spend
   (customer lifetime value, CLV) and check it against what really happened.
3. **Which channels actually work?** When a customer buys after several ads,
   decide how much credit each channel gets, and check if that credit is right.

**Why part of the data is simulated.** Questions 1 and 2 run only on the real
transactions. Question 3 needs the *true* effect of each ad channel to check any
method against, and no public dataset has that (it would take an experiment
nobody publishes). So we simulated ad journeys **for the real customers**, with
channel effects we set, and tied each customer's intent to their real purchase
history. Two traps are built in: `retargeting` has **zero** real effect but is
shown to people already about to buy, and a hidden "ready to buy" state that no
method can see drives both ad exposure and buying.

## How we did it

**1. Cleaning the real data.** 1,067,371 invoice lines became 30,816 purchase
days for 4,899 customers:

| step | removed |
|---|---:|
| exact duplicate lines | 34,335 lines |
| no customer ID | 235,151 lines |
| non-product codes (postage, fees, manual adjustments, samples) | 3,662 lines |
| price of zero or less | 60 lines |
| purchases later cancelled in full (17,586 cancellation lines matched back to the original purchase) | 6,028 lines |
| purchase days wiped out by returns | 176 days |
| customers first seen in the test period (no history to predict from) | 940 customers |

Several invoices on the same day count as one purchase, so a customer who splits
a basket across two invoices doesn't look twice as loyal.

**2. Train and test split in time.** Models are fitted on the first 18 months
(Dec 2009 to May 2011) and scored on the last ~6 months (Jun to Dec 2011), which
they never saw.

**3. Segmentation.** k-means on recency, frequency, spend, products per order and
return rate. Checked for stability (bootstrap), and forward in time: do segments
formed on the first 18 months predict the next 6? Six different ways of choosing
the number of segments were compared, plus a density method (HDBSCAN) that
doesn't need one.

**4. Lifetime value.** BG/NBD (how many more purchases, and has the customer
quietly left?) times Gamma-Gamma (how much per purchase), written from scratch
and checked against a 50-digit reference calculation. Compared with a
gradient-boosting model.

**5. Attribution.** Seven methods (first-touch, last-touch, linear, time-decay,
Markov, Shapley, sampled Shapley) scored against the known truth. Then a budget
simulation: split £45,000 by each method's credit and count the sales it really
produces.

**6. Pipeline.** dbt on DuckDB with a test that fails the build if test-period
data leaks into training. The same logic also runs on Spark, Databricks and an
Airflow DAG, and a FastAPI + Java API + React front end serve the results. Details
in [docs/ENGINEERING.md](docs/ENGINEERING.md).

## What we found

**Segments** (numbered by k-means; the names are ours):

| segment | customers | orders so far | days since last order | no purchase in next 6 months | share of predicted value |
|---|---:|---:|---:|---:|---:|
| 3 · wholesale accounts | 34 (0.7%) | 62 | 27 | 9% | **21%** |
| 4 · regulars | 1,615 (33%) | 7.5 | 75 | 23% | **49%** |
| 1 · occasional | 603 (12%) | 1.8 | 185 | 49% | 7% |
| 2 · recent one-timers | 1,514 (31%) | 1.0 | 124 | 52% | 17% |
| 0 · lapsed one-timers | 1,133 (23%) | 1.0 | 354 | 76% | 5% |

- The segments predict the future: the chance of not buying again ranges from
  **9% to 76%**. They are stable too (bootstrap agreement 0.92).
- **34 wholesale customers hold 21% of predicted value.** The top 20% of
  customers hold 69%.
- The customers are **one continuous cloud, not natural groups**: HDBSCAN finds
  one dense core (63% of customers) and calls the rest noise. So the segments are
  useful cuts, not groups that exist on their own.

**Lifetime value:**

| model | rank correlation with real 6-month spend | mean abs error |
|---|---:|---:|
| **BG/NBD + Gamma-Gamma** | **0.63** | **£587** |
| gradient boosting | 0.54 | £808 |

- It ranks customers well but misses in a clear pattern: it **over-predicts the
  lowest 10% by 39%** and **under-predicts the top 10% by 26%**. The test period
  includes the Christmas season, when heavy buyers buy more than a steady-rate
  model expects.
- The raw per-customer correlation (0.92) looks excellent but is misleading: the
  top 1% of customers are 37% of all spend, so it mostly measures whether the
  model found the giant accounts. On a log scale it is 0.60.
- Gamma-Gamma assumes order size doesn't depend on how often someone buys. On this
  data it does (correlation +0.13), so its spend estimates lean high for frequent
  buyers.

**Attribution:**

- **Every method gives credit to `retargeting`, the channel that does nothing.**
  Last-touch gives it 20% of all credit. Shapley, the most principled method,
  still gives it 14%.
- Budgeting on last-touch instead of the truth loses **171 sales (9.4%)** and
  puts **£9,123 (20%) of the budget** into retargeting.
- The best methods are nearly tied: time-decay, linear and Shapley all have
  errors within 0.001 of each other. Shapley still gives the best *budget*,
  losing only 2.9% of sales.
- **Weighting by customer value makes it worse.** Retargeting reaches the
  store's best buyers, so on value it ranks *first* of all 12 channels (1.62x the
  average customer value).
- Standard sampled Shapley does not converge on this data (32x more samples cut
  the error only 2.2x, when 5.7x is expected). A different value function fixes
  it (5.4x).

**Real data also caught two bugs the simulated version never could:**

- **The CLV formula returned huge negative values.** It divides by (a - 1), and
  the code replaced that with 0.000001 whenever a < 1. The old fake data always
  had a > 1; real customers give a = 0.15. Fixed, with a test.
- **ROAS came out at 16,985x.** Each sale was credited with the customer's whole
  6 months of spend, not one order. Fixed: one sale is now worth one average
  order, and paid search comes out at a believable 12x.

## What we decided, and why

1. **Use 5 segments.** How well segments separate future spend (adjusted η²):
   0.039 at k=4, 0.065 at k=5, 0.071 at k=7. Five gets most of what seven gives,
   with fewer groups for a team to act on. Four (the pick of four geometric
   criteria) loses a lot.
2. **Use BG/NBD over gradient boosting.** It ranks better (0.63 vs 0.54), gives
   the chance each customer is still active, and can predict further ahead than
   its training window. Use it to rank and size groups, not to price one
   customer.
3. **Treat wholesale accounts separately.** 34 customers hold 21% of value, and
   they skew every average. They need account management, not campaigns.
4. **Stop budgeting on last-touch, and run a geo experiment on retargeting.**
   Switch it off in matched regions for four weeks. If it really does nothing,
   the test costs nothing.
5. **Don't weight the budget by customer value until that test is done.** Value
   weighting rewards retargeting twice for the same bias.
6. **Replace sampled Shapley** with the fixed value function (or exact
   per-journey Shapley), which converges and adds up exactly.

**Limits:** the ad-channel effects are simulated, so what carries over is how
the methods fail, not the exact percentages. The retailer is one UK business with
many wholesale buyers, and the one test period includes Christmas.

## How to run it

```bash
pip install -r requirements.txt

# 1. get the data (45 MB) into data/raw/
mkdir -p data/raw
curl -L -o data/raw/online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip data/raw/online_retail_ii.zip -d data/raw/

python src/build_data.py     # ~5 min first time (reads the Excel file), then ~30s
python run_analytics.py      # ~1 min   main report   -> out/analytics_report.txt, out/EXECUTIVE_MEMO.md
python run_complete.py       # ~5 min   deeper checks -> out/complete_report.txt (dbt build, k, Shapley)
python export_bi.py          #          CSVs for the dashboards -> bi_export/
python -m pytest tests -q    # 95 tests
```

Full reports: [`out/analytics_report.txt`](out/analytics_report.txt) and
[`out/complete_report.txt`](out/complete_report.txt).

## Code layout

```
src/clean_retail.py          cleaning rules for Online Retail II, each one counted
src/build_data.py            real transactions + simulated ad journeys for the same customers
src/segmentation.py          choosing k, stability, forward test, HDBSCAN
src/clv.py                   BG/NBD and Gamma-Gamma, written out
src/attribution.py           the seven attribution methods
src/scaling.py               Shapley fixes, higher-order Markov, CAC and ROAS
src/pipeline.py              load to DuckDB + dbt build + a small DAG runner
dbt/                         5 models, 10 tests, including the leakage test
api/, java-api/, frontend/   REST APIs and a React dashboard over bi_export/
```
