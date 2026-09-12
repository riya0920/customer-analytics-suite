# Customer Analytics REST API (FastAPI)

A typed REST layer over the `customer-analytics-suite` outputs. It serves the same
exported results the dashboard reads (segmentation, CLV, attribution) as JSON, so
a browser front end — or the Java service in Sprint 2 Task 4 — can consume them
over HTTP instead of reading CSVs directly.

**Stack:** FastAPI + Pydantic (typed responses + OpenAPI), stdlib `csv` for
loading, `uvicorn` to serve, `pytest` + `TestClient` for tests.

## Run it

```bash
# from the repo root
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
# interactive docs (OpenAPI): http://localhost:8000/docs
```

The data directory is resolved at startup in this order, so it works both from a
fresh clone and from a working tree that has just run the pipeline:

1. `$CAS_DATA_DIR` (if set)
2. `<repo>/bi_export` — the pipeline's own output (`python export_bi.py`)
3. `<repo>/frontend/public/data` — the committed copy the dashboard also ships

## Endpoints

Every route declares a Pydantic `response_model`, so `/openapi.json` and `/docs`
are generated from the same types the data layer produces.

| Method & path | Returns |
|---|---|
| `GET /health` | Data dir + per-table row counts |
| `GET /api/kpis` | Headline KPIs (one object) |
| `GET /api/segments` | The five segments |
| `GET /api/segments/k-selection` | k-selection stability metrics |
| `GET /api/clv/summary` | BG/NBD + Gamma-Gamma params and fit metrics |
| `GET /api/clv/models` | BG/NBD vs GBM comparison |
| `GET /api/clv/channels` | CLV by acquiring channel |
| `GET /api/clv/customers?limit=&offset=&segment=` | Paginated per-customer CLV |
| `GET /api/attribution/methods` | Per-method accuracy scores |
| `GET /api/attribution/credit?method=` | Per-channel credit vs truth (optionally one method) |
| `GET /api/attribution/budget` | Budget outcome per allocation method |

### Error handling

All errors share one body shape: `{"error": "<reason>", "detail": "<message>"}`.

- **404** — unknown `method` on `/api/attribution/credit`, or unknown `segment`
  on `/api/clv/customers` (the message lists the known values).
- **422** — invalid query params (e.g. `limit=0` or `limit>1000`), normalised
  from FastAPI's validation errors into the same shape.

```bash
curl -s "localhost:8000/api/attribution/credit?method=bogus"
# {"error":"not_found","detail":"Unknown attribution method 'bogus'. Known: ...shapley, truth"}
```

## Full-stack flow

```
 dbt / Python pipeline ── export_bi.py ──▶ CSVs (bi_export/, mirrored to
                                            frontend/public/data/)
                                                     │
                          ┌──────────────────────────┴───────────────────────┐
                          ▼                                                    ▼
                 api/ (FastAPI)                                 frontend/ (React) static mode
                 loads CSVs → typed                             reads the CSVs directly
                 Pydantic models → JSON                         (this is the GitHub Pages deploy)
                          │
                          ▼
                 frontend/ (React) API mode
                 VITE_API_BASE=http://localhost:8000
                 fetch() → same typed Dataset → charts
```

The React app has two interchangeable data sources (`frontend/src/data/source.ts`):

- **static** (default): reads the bundled CSVs — keeps the GitHub Pages build
  self-contained.
- **API**: set `VITE_API_BASE` and it fetches from this service instead, paging
  through `/api/clv/customers` to completion. CORS allows the Vite dev origins.

Run the whole stack locally:

```bash
# terminal 1 — the API
uvicorn api.main:app --port 8000

# terminal 2 — the dashboard, pointed at the API
cd frontend
VITE_API_BASE=http://localhost:8000 npm run dev
```

The dashboard then renders identically, but every figure arrives over HTTP from
the API. The response models here are also the contract the Java (Spring Boot)
port in Sprint 2 Task 4 reproduces.

## Tests

```bash
python -m pytest tests/test_api.py -q   # 15 tests
```

`tests/test_api.py` drives the app with `TestClient` over the committed data and
covers every endpoint, pagination, the segment/method filters, both error paths
(404 + 422), and that the OpenAPI schema is served — so it runs in CI without the
pipeline having been run.
