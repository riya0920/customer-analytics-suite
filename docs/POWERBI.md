# Power BI dashboard — build and publish

This builds a Power BI report over the **same `bi_export/*.csv` files** the
Tableau Public and Looker Studio dashboards use, so every figure is something the
pipeline produced (regenerate with `python export_bi.py`). Three pages mirror the
other two dashboards: Segmentation, CLV, Attribution.

## 0. Get the data

```bash
python export_bi.py        # writes bi_export/*.csv (17 tidy files)
```

## 1. Build the report (Power BI Desktop — free, Windows)

Install Power BI Desktop (free, Microsoft Store or download). Then **Home → Get
data → Text/CSV** and import these files. Power BI auto-detects types; set the
share/rate columns to *Percentage* or keep decimal and format the visuals.

**Page 1 — Segmentation** (`segments.csv`)
- Clustered bar: Axis = `segment`, Values = `share_of_customers` and
  `share_of_value` — the headline is Segment 0 at **7.6% of customers / 37.3% of
  value** while segments 1 and 3 are ~45% of customers holding under 4% of value.
- Column: `segment` vs `churn_rate`.
- Card: `kpis.csv[top20pct_value_share]` = **0.766** (top-20% of customers hold
  77% of predicted value).

**Page 2 — CLV** (`clv_by_channel.csv`, `clv_model_comparison.csv`)
- Bar: `channel` vs `clv_index` (which acquisition channels bring higher-value
  customers; spread ~0.94–1.08).
- Table/bar from `clv_model_comparison.csv`: BG/NBD vs the GBM challenger.
- Card: `kpis.csv[clv_rank_spearman]` = **0.7366** (holdout rank agreement).

**Page 3 — Attribution** (`attribution_credit_long.csv`, `attribution_mae.csv`,
`attribution_zero_effect_credit.csv`)
- Matrix: Rows = `method`, Columns = `channel`, Values = `credit` — include the
  `truth` rows so each method's disagreement with the planted truth is visible.
- Bar: `attribution_mae.csv` `method` vs `mae`, sorted ascending (Shapley best at
  **0.0292**; last-touch worst).
- Bar: `attribution_zero_effect_credit.csv` — every method credits the
  **retargeting** channel whose TRUE effect is zero. Attribution is not
  incrementality.

Save as `bi/customer_analytics.pbix`.

## 2. Publish to a public URL

**Publish to web (public)** gives a shareable link:

1. Sign in to the **Power BI service** (app.powerbi.com) and upload/open the report.
2. **File → Publish to web (public) → Create embed code**, then copy the public link.

**Account caveat (read before you start):** Power BI *service* sign-up requires a
**work or school (Azure AD) email** — consumer Gmail/Outlook.com accounts cannot
register for Power BI. If you only have a personal account, either use a work/
school email, or note that **Tableau Public and Looker Studio (both already
published here) are the reliable free public-URL paths** and Power BI stays as the
`.pbix` + exported PDF. "Publish to web" also makes the report fully public — fine
here because the data is entirely synthetic (no PII).

## 3. Record the link

Once published, paste the URL into the "Power BI" bullet in the README (replacing
the placeholder).
