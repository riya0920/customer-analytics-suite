"""Tests for the REST API (api/). Uses FastAPI's TestClient over an app whose
repository is loaded from the committed data, so the suite runs on a fresh clone
and in CI without needing the pipeline to have been run."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.data import load_repository
from api.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    repo = load_repository()  # resolves bi_export/ or frontend/public/data/
    return TestClient(create_app(repo=repo))


def test_health_reports_row_counts(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["row_counts"]["segments"] == 5
    assert body["row_counts"]["clv_per_customer"] == 8000


def test_kpis_shape_and_values(client: TestClient) -> None:
    r = client.get("/api/kpis")
    assert r.status_code == 200
    body = r.json()
    assert body["total_customers"] == 8000
    assert body["n_channels"] == 12
    assert set(body) == {
        "total_customers",
        "n_segments",
        "top20pct_value_share",
        "clv_rank_spearman",
        "best_attribution_mae",
        "n_channels",
    }


def test_segments_typed(client: TestClient) -> None:
    r = client.get("/api/segments")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 5
    assert all(isinstance(row["customers"], int) for row in rows)
    total_share = sum(row["share_of_value"] for row in rows)
    assert total_share == pytest.approx(1.0, abs=0.01)


def test_k_selection_present(client: TestClient) -> None:
    r = client.get("/api/segments/k-selection")
    assert r.status_code == 200
    assert any(row["k"] == 5 for row in r.json())


def test_clv_endpoints(client: TestClient) -> None:
    assert client.get("/api/clv/summary").status_code == 200
    channels = client.get("/api/clv/channels").json()
    assert len(channels) == 12
    models = client.get("/api/clv/models").json()
    names = {m["model"] for m in models}
    assert "BG/NBD + Gamma-Gamma" in names


def test_clv_customers_pagination(client: TestClient) -> None:
    r = client.get("/api/clv/customers", params={"limit": 10, "offset": 0})
    assert r.status_code == 200
    page = r.json()
    assert page["total"] == 8000
    assert page["limit"] == 10
    assert len(page["items"]) == 10

    # A second page returns different customers.
    r2 = client.get("/api/clv/customers", params={"limit": 10, "offset": 10})
    ids1 = {c["customer_id"] for c in page["items"]}
    ids2 = {c["customer_id"] for c in r2.json()["items"]}
    assert ids1.isdisjoint(ids2)


def test_clv_customers_segment_filter(client: TestClient) -> None:
    r = client.get("/api/clv/customers", params={"segment": "Segment 4", "limit": 50})
    assert r.status_code == 200
    page = r.json()
    assert page["total"] > 0
    assert all(c["segment"] == "Segment 4" for c in page["items"])


def test_clv_customers_unknown_segment_404(client: TestClient) -> None:
    r = client.get("/api/clv/customers", params={"segment": "Segment 99"})
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "Segment 99" in body["detail"]


def test_clv_customers_bad_limit_422(client: TestClient) -> None:
    r = client.get("/api/clv/customers", params={"limit": 0})
    assert r.status_code == 422
    assert r.json()["error"] == "unprocessable_entity"

    r2 = client.get("/api/clv/customers", params={"limit": 99999})
    assert r2.status_code == 422


def test_attribution_methods(client: TestClient) -> None:
    r = client.get("/api/attribution/methods")
    assert r.status_code == 200
    methods = {m["method"] for m in r.json()}
    assert "shapley" in methods
    # Shapley is the lowest-MAE method in this export.
    best = min(r.json(), key=lambda m: m["mae"])
    assert best["method"] == "shapley"


def test_attribution_credit_filter(client: TestClient) -> None:
    r = client.get("/api/attribution/credit", params={"method": "shapley"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 12  # one per channel
    assert all(row["method"] == "shapley" for row in rows)


def test_attribution_credit_all_includes_truth(client: TestClient) -> None:
    rows = client.get("/api/attribution/credit").json()
    methods = {row["method"] for row in rows}
    assert "truth" in methods


def test_attribution_credit_unknown_method_404(client: TestClient) -> None:
    r = client.get("/api/attribution/credit", params={"method": "nope"})
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "nope" in body["detail"]


def test_attribution_budget(client: TestClient) -> None:
    rows = client.get("/api/attribution/budget").json()
    allocations = {row["allocation"] for row in rows}
    assert "shapley" in allocations
    assert "TRUTH" in allocations


def test_openapi_served(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/api/kpis" in r.json()["paths"]
