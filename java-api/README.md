# Customer Analytics API: Java (Spring Boot)

A **Spring Boot port of the FastAPI service in [`../api`](../api)**. It serves the
same `customer-analytics-suite` outputs (segmentation, CLV, attribution) over the
**same REST endpoints, the same JSON shapes, and the same error bodies**, so it
is a drop-in replacement for the Python service behind the React dashboard.

**Stack:** Java 17, Spring Boot 3.3 (spring-web), Jackson (via records), Maven
(wrapper committed), JUnit 5 + MockMvc for tests. No third-party CSV or JSON
library: the loader uses a small hand-written CSV reader mirroring the Python and
TypeScript ones.

## Run it

```bash
cd java-api
./mvnw spring-boot:run           # http://localhost:8000  (mvnw.cmd on Windows)
# or build a jar:
./mvnw -DskipTests package
java -jar target/customer-analytics-api-1.0.0.jar
```

Data-directory resolution matches the Python service: `$CAS_DATA_DIR`, then
`bi_export/`, then `frontend/public/data/` (each tried as `./` and `../` so it
works whether launched from the repo root or from `java-api/`), so it runs from a
fresh clone.

## Endpoints

Identical to [`../api/README.md`](../api/README.md):

| Method & path | Returns |
|---|---|
| `GET /health` | Data dir + per-table row counts |
| `GET /api/kpis` | Headline KPIs |
| `GET /api/segments` · `GET /api/segments/k-selection` | Segments; k-selection metrics |
| `GET /api/clv/summary` · `models` · `channels` | CLV fit, model comparison, per-channel |
| `GET /api/clv/customers?limit=&offset=&segment=` | Paginated per-customer CLV |
| `GET /api/attribution/methods` · `credit?method=` · `budget` | Scores; credit-vs-truth; budget cost |

**Errors** use the same `{"error", "detail"}` body: **404** for an unknown
`method`/`segment`, **422** for bad pagination (`limit` outside 1–1000, negative
`offset`) or a non-numeric param.

## Parity with the Python service

The response DTOs are Java `record`s
([`model/Models.java`](src/main/java/com/customeranalytics/api/model/Models.java))
whose field names match the CSV columns exactly, so Jackson emits byte-for-byte
the same JSON FastAPI's Pydantic models do. Verified live: both services return,
for example:

```
GET /api/kpis
{"total_customers":8000,"n_segments":5,"top20pct_value_share":0.766,
 "clv_rank_spearman":0.7366,"best_attribution_mae":0.0292,"n_channels":12}

GET /api/attribution/credit?method=bogus   -> 404
{"error":"not_found","detail":"Unknown attribution method 'bogus'. Known: ...shapley, truth"}

GET /api/clv/customers?limit=0             -> 422
{"error":"unprocessable_entity","detail":"limit: must be between 1 and 1000"}
```

Because the contract is identical, the React dashboard runs against either backend
unchanged; point it at this service and every figure loads over HTTP from Java:

```bash
# terminal 1
cd java-api && ./mvnw spring-boot:run          # :8000

# terminal 2
cd frontend && VITE_API_BASE=http://localhost:8000 npm run dev
```

(Confirmed in a browser: the dashboard renders identically, with all `/api/*`
requests served by the Spring Boot app.)

## Tests

```bash
./mvnw -B test    # 15 tests
```

[`AnalyticsApiTests`](src/test/java/com/customeranalytics/api/AnalyticsApiTests.java)
drives the full Spring context with MockMvc over the committed data and mirrors
`../tests/test_api.py` one-for-one, every endpoint, both filters, pagination, and
the 404/422 error paths, so both ports are held to the same contract. Runs in CI
via the Maven wrapper (`java-api` job in `.github/workflows/ci.yml`).

## Design notes

- **Records as the type layer.** Immutable DTOs, no getters/setters boilerplate,
  and Jackson serialises them directly, the closest Java analogue to Pydantic
  models / TypeScript interfaces.
- **Load once, serve from memory.** The API is read-only over a static export, so
  `DataRepository` loads every CSV into typed records at startup (a Spring
  singleton) and answers queries in memory.
- **Uniform errors via `@RestControllerAdvice`.** Domain exceptions map to
  404/422 with the shared body; a non-numeric query param (Spring's
  `MethodArgumentTypeMismatchException`) is remapped from the default 400 to 422 to
  match FastAPI's validation semantics.
- **Committed Maven wrapper** (script-only, no jar in the tree) so CI and a fresh
  clone need only a JDK.
