"""FastAPI application exposing the customer-analytics results as typed JSON.

Every route declares a ``response_model``, so the OpenAPI schema at /docs is
generated from the same Pydantic models the data layer produces. Errors are
returned in a uniform ``{"error", "detail"}`` body.

Run locally:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .data import Repository, load_repository
from .models import (
    AttributionCredit,
    BudgetOutcome,
    ClvByChannel,
    ClvModel,
    ClvPerCustomer,
    ClvSummary,
    ErrorResponse,
    Health,
    Kpi,
    KSelection,
    MethodScore,
    Page,
    Segment,
)

# Errors documented on every route so the contract (and the Java port in Task 4)
# is explicit about non-200 shapes.
ERROR_RESPONSES = {
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


def create_app(repo: Optional[Repository] = None) -> FastAPI:
    """Build the app. Pass a ``repo`` to inject data (used by the tests);
    otherwise it is loaded from the resolved data directory at startup."""
    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        # Load eagerly at startup unless a repo was injected, so a missing data
        # dir fails fast rather than on the first request.
        if app_.state.repo is None:
            app_.state.repo = load_repository()
        yield

    app = FastAPI(
        title="Customer Analytics API",
        version="1.0.0",
        description=(
            "Typed REST access to the customer-analytics-suite outputs: "
            "segmentation, CLV, and attribution."
        ),
        lifespan=lifespan,
    )

    # CORS so the React dashboard (a different origin) can call the API from the
    # browser. Origins are configurable; default is the local dev servers.
    origins = os.environ.get(
        "CAS_CORS_ORIGINS",
        "http://localhost:5173,http://localhost:5177",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.state.repo = repo

    def get_repo() -> Repository:
        if app.state.repo is None:
            app.state.repo = load_repository()
        return app.state.repo

    # --- uniform error bodies ---------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exc(_req: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": _reason(exc.status_code), "detail": str(exc.detail)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc(
        _req: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Collapse FastAPI's list of validation problems into the same shape as
        # every other error (e.g. limit=0 -> 422 unprocessable_entity).
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []))
        msg = first.get("msg", "invalid request")
        return JSONResponse(
            status_code=422,
            content={
                "error": "unprocessable_entity",
                "detail": f"{loc}: {msg}" if loc else msg,
            },
        )

    # --- health -----------------------------------------------------------
    @app.get("/health", response_model=Health, tags=["meta"])
    def health() -> Health:
        repo = get_repo()
        return Health(
            status="ok",
            data_dir=str(repo.data_dir),
            row_counts=repo.row_counts(),
        )

    # --- KPIs -------------------------------------------------------------
    @app.get("/api/kpis", response_model=Kpi, tags=["overview"])
    def kpis() -> Kpi:
        return get_repo().kpis

    # --- segmentation -----------------------------------------------------
    @app.get("/api/segments", response_model=List[Segment], tags=["segmentation"])
    def segments() -> List[Segment]:
        return get_repo().segments

    @app.get(
        "/api/segments/k-selection",
        response_model=List[KSelection],
        tags=["segmentation"],
    )
    def k_selection() -> List[KSelection]:
        return get_repo().k_selection

    # --- CLV --------------------------------------------------------------
    @app.get("/api/clv/summary", response_model=List[ClvSummary], tags=["clv"])
    def clv_summary() -> List[ClvSummary]:
        return get_repo().clv_summary

    @app.get("/api/clv/models", response_model=List[ClvModel], tags=["clv"])
    def clv_models() -> List[ClvModel]:
        return get_repo().clv_models

    @app.get("/api/clv/channels", response_model=List[ClvByChannel], tags=["clv"])
    def clv_channels() -> List[ClvByChannel]:
        return get_repo().clv_by_channel

    @app.get(
        "/api/clv/customers",
        response_model=Page[ClvPerCustomer],
        responses=ERROR_RESPONSES,
        tags=["clv"],
    )
    def clv_customers(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        segment: Optional[str] = Query(None),
    ) -> Page[ClvPerCustomer]:
        repo = get_repo()
        if segment is not None and not any(
            s.segment == segment for s in repo.segments
        ):
            raise HTTPException(404, f"Unknown segment '{segment}'")
        items, total = repo.customers(limit, offset, segment)
        return Page[ClvPerCustomer](
            items=items, total=total, limit=limit, offset=offset
        )

    # --- attribution ------------------------------------------------------
    @app.get(
        "/api/attribution/methods",
        response_model=List[MethodScore],
        tags=["attribution"],
    )
    def attribution_methods() -> List[MethodScore]:
        return get_repo().method_scores

    @app.get(
        "/api/attribution/credit",
        response_model=List[AttributionCredit],
        responses=ERROR_RESPONSES,
        tags=["attribution"],
    )
    def attribution_credit(
        method: Optional[str] = Query(
            None,
            description="Filter to one method (e.g. 'shapley'). "
            "Omit for all methods including 'truth'.",
        ),
    ) -> List[AttributionCredit]:
        repo = get_repo()
        if method is not None and not repo.has_method(method):
            known = repo.attribution_methods() + ["truth"]
            raise HTTPException(
                404,
                f"Unknown attribution method '{method}'. Known: {', '.join(known)}",
            )
        return repo.attribution_for(method)

    @app.get(
        "/api/attribution/budget",
        response_model=List[BudgetOutcome],
        tags=["attribution"],
    )
    def attribution_budget() -> List[BudgetOutcome]:
        return get_repo().budget_outcomes

    return app


def _reason(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        422: "unprocessable_entity",
        500: "internal_error",
    }.get(status_code, "error")


# Module-level app for `uvicorn api.main:app`.
app = create_app()
