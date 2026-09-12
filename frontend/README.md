# Customer Analytics Dashboard (React + TypeScript)

A front end over the `customer-analytics-suite` pipeline outputs. It answers the
three questions the pipeline was built for — **who** the customers are
(segmentation), **what** they are worth (CLV), and **which channels earned it**
(attribution) — reading the exported model results directly rather than
re-deriving or hard-coding any figure.

**Live:** https://riya0920.github.io/customer-analytics-suite/
**Stack:** React 18, TypeScript (strict), Vite, Recharts, Vitest + Testing Library.

```bash
cd frontend
npm install
npm run dev        # local dev server
npm test           # unit + component tests
npm run typecheck  # tsc --noEmit
npm run build      # production build to dist/
```

## What it shows

| View | Reads | Charts |
|---|---|---|
| **Segmentation** | `segments.csv`, `k_selection.csv` | Value share per segment, a value-concentration (Lorenz) curve, the segment table with forward-tested holdout outcomes, and the k-selection stability curve behind `k = 5` |
| **CLV** | `clv_by_channel.csv`, `clv_summary.csv`, `clv_model_comparison.csv`, `clv_per_customer.csv` | Predicted-CLV distribution over all 8,000 customers, BG/NBD + Gamma-Gamma vs the GBM challenger, and value by acquiring channel (sortable) |
| **Attribution** | `attribution_credit_long.csv`, `method_scores.csv`, `budget_outcomes.csv` | Per-method credit vs known truth, an accuracy scatter (MAE vs credit wrongly given to the zero-effect channel), and what each method costs in conversions — including the planted zero-effect control every method over-credits |

## Architecture decisions

- **The pipeline is the source of truth; the UI only presents.** The exported
  CSVs are copied verbatim into `public/data/` and shipped as static assets. The
  UI computes *presentation-layer* aggregates only (a Lorenz curve, a histogram,
  sorting) — all in `src/data/transforms.ts`, as pure functions, so those numbers
  are unit-tested without rendering. No model logic is re-implemented here, which
  keeps the front end honest: if a figure is wrong, it is wrong in the pipeline.

- **Typed at the data boundary.** Every CSV has an explicit row interface in
  `src/data/types.ts` whose fields match the column names exactly. The loaders in
  `src/data/loaders.ts` map raw cells through those types with a numeric coercer
  that *throws* on non-numeric input, so a malformed export fails loudly at load
  instead of silently rendering `NaN`. A change to the pipeline's output columns
  surfaces here as a compile error.

- **A hand-written CSV parser, deliberately.** `src/data/csv.ts` is ~90 lines and
  handles quoted fields with embedded commas (e.g. `"Spearman (rank, holdout)"`),
  escaped quotes, CRLF, and a BOM. It is fully unit-tested. Pulling in a CSV
  library for three columns of well-formed data would be more surface area than
  parser, so this is the smaller, auditable choice.

- **Loading is an explicit state machine.** `src/data/store.tsx` models the async
  load as `loading → ready | error` via `useReducer` behind a React context, so
  loading and error handling live in exactly one place and components never fetch.
  The provider takes an injectable `loader`, which is how the component tests hand
  the tree a dataset synchronously.

- **State management, scoped to what needs it.** Global (context + reducer): the
  loaded dataset and its lifecycle. Local (`useState`): the active tab and
  per-view controls (attribution method, channel sort). Server-state libraries
  (React Query, Redux) would be overkill for a single one-shot static load — the
  data never changes after mount.

- **Typed, reusable presentational components.** `DataTable<T>`, `StatCard`,
  `Card`, and `Callout` in `src/components/primitives.tsx` are generic and props-
  only; the three views compose them. Charts use Recharts with a fixed
  categorical palette (`colors.ts`) so a segment/channel keeps its colour across
  views.

## Testing

`npm test` runs four suites (30 assertions):

- `csv.test.ts` — the parser against quoted/escaped/CRLF/BOM edge cases.
- `transforms.test.ts` — the Lorenz curve, histogram binning/clipping, and
  attribution helpers, with numbers checked by hand.
- `loaders.test.ts` — the typed parsers against sample strings copied verbatim
  from the real exports (pins the column names).
- `App.test.tsx` — renders the app with an injected dataset and asserts the
  loading→ready→error states and tab switching (Testing Library + jsdom).

CI (`.github/workflows/ci.yml`) typechecks, tests, and builds the frontend on
every push; `deploy-dashboard.yml` builds with the Pages base path and publishes.

## Data provenance

The CSVs under `public/data/` are the committed outputs of the pipeline's
`bi_export` step. Regenerate them from the repo root and copy them in:

```bash
# from the repo root, after run_analytics.py / run_complete.py
python export_bi.py              # writes bi_export/*.csv
cp bi_export/{segments,k_selection,clv_by_channel,clv_summary,\
clv_model_comparison,clv_per_customer,attribution_credit_long,\
method_scores,budget_outcomes,kpis}.csv frontend/public/data/
```
