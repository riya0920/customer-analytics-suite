"""Load the exported CSVs into typed models once, and answer queries over them.

The API is read-only over a static export, so everything is loaded into memory at
startup and served from there. The data directory is resolved in this order:

  1. $CAS_DATA_DIR, if set
  2. <repo>/bi_export            (the pipeline's own output, present after a run)
  3. <repo>/frontend/public/data (the committed copy the dashboard also ships)

so it works from a fresh clone (falls back to the committed copy) and from a
working tree that has just run the pipeline (uses the fresh export).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from .models import (
    AttributionCredit,
    BudgetOutcome,
    ClvByChannel,
    ClvModel,
    ClvPerCustomer,
    ClvSummary,
    Kpi,
    KSelection,
    MethodScore,
    Segment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

T = TypeVar("T")


def resolve_data_dir(explicit: Optional[str] = None) -> Path:
    candidates = []
    env = explicit or os.environ.get("CAS_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT / "bi_export")
    candidates.append(REPO_ROOT / "frontend" / "public" / "data")
    for c in candidates:
        if c.is_dir() and (c / "kpis.csv").exists():
            return c
    raise FileNotFoundError(
        "No data directory with kpis.csv found. Run `python export_bi.py` or set "
        f"CAS_DATA_DIR. Looked in: {', '.join(str(c) for c in candidates)}"
    )


def _read(path: Path, make: Callable[[dict[str, str]], T]) -> List[T]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [make(row) for row in csv.DictReader(f)]


def _f(row: dict[str, str], key: str) -> float:
    """Parse a float cell, treating an empty cell as NaN (e.g. a blank
    incremental_cac) rather than raising."""
    raw = row[key].strip()
    return float("nan") if raw == "" else float(raw)


def _i(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


class Repository:
    """In-memory store of every exported table, loaded once."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.segments = _read(
            data_dir / "segments.csv",
            lambda r: Segment(
                segment=r["segment"],
                customers=_i(r, "customers"),
                share_of_customers=_f(r, "share_of_customers"),
                cal_frequency=_f(r, "cal_frequency"),
                cal_monetary=_f(r, "cal_monetary"),
                recency_days=_f(r, "recency_days"),
                churn_rate=_f(r, "churn_rate"),
                holdout_orders=_f(r, "holdout_orders"),
                holdout_spend=_f(r, "holdout_spend"),
                clv_total=_f(r, "clv_total"),
                clv_mean=_f(r, "clv_mean"),
                share_of_value=_f(r, "share_of_value"),
            ),
        )
        self.k_selection = _read(
            data_dir / "k_selection.csv",
            lambda r: KSelection(
                k=_i(r, "k"),
                silhouette=_f(r, "silhouette"),
                calinski_harabasz=_f(r, "calinski_harabasz"),
                davies_bouldin=_f(r, "davies_bouldin"),
                inertia=_f(r, "inertia"),
                mean_ari=_f(r, "mean_ari"),
                adjusted_eta_squared=_f(r, "adjusted_eta_squared"),
                smallest_cluster_share=_f(r, "smallest_cluster_share"),
            ),
        )
        self.clv_by_channel = _read(
            data_dir / "clv_by_channel.csv",
            lambda r: ClvByChannel(
                channel=r["channel"],
                converters_touched=_i(r, "converters_touched"),
                mean_clv=_f(r, "mean_clv"),
                median_clv=_f(r, "median_clv"),
                total_clv=_f(r, "total_clv"),
                clv_index=_f(r, "clv_index"),
            ),
        )
        self.clv_summary = _read(
            data_dir / "clv_summary.csv",
            lambda r: ClvSummary(metric=r["metric"], value=_f(r, "value")),
        )
        self.clv_models = _read(
            data_dir / "clv_model_comparison.csv",
            lambda r: ClvModel(
                model=r["model"], spearman=_f(r, "spearman"), mae=_f(r, "mae")
            ),
        )
        self.clv_per_customer = _read(
            data_dir / "clv_per_customer.csv",
            lambda r: ClvPerCustomer(
                customer_id=_i(r, "customer_id"),
                segment=r["segment"],
                predicted_clv=_f(r, "predicted_clv"),
                holdout_spend=_f(r, "holdout_spend"),
                cal_frequency=_f(r, "cal_frequency"),
                recency_days=_f(r, "recency_days"),
            ),
        )
        self.attribution = _read(
            data_dir / "attribution_credit_long.csv",
            lambda r: AttributionCredit(
                method=r["method"],
                channel=r["channel"],
                credit=_f(r, "credit"),
                truth=_f(r, "truth"),
                error=_f(r, "error"),
            ),
        )
        self.method_scores = _read(
            data_dir / "method_scores.csv",
            lambda r: MethodScore(
                method=r["method"],
                mae=_f(r, "mae"),
                credit_to_zero_effect=_f(r, "credit_to_zero_effect"),
            ),
        )
        self.budget_outcomes = _read(
            data_dir / "budget_outcomes.csv",
            lambda r: BudgetOutcome(
                allocation=r["allocation"],
                conversions=_f(r, "conversions"),
                conversions_lost_vs_truth=_f(r, "conversions_lost_vs_truth"),
                pct_conversions_lost=_f(r, "pct_conversions_lost"),
                spend_on_zero_effect=_f(r, "spend_on_zero_effect"),
                customer_value=_f(r, "customer_value"),
                pct_value_lost=_f(r, "pct_value_lost"),
            ),
        )
        kpis = _read(
            data_dir / "kpis.csv",
            lambda r: Kpi(
                total_customers=_i(r, "total_customers"),
                n_segments=_i(r, "n_segments"),
                top20pct_value_share=_f(r, "top20pct_value_share"),
                clv_rank_spearman=_f(r, "clv_rank_spearman"),
                best_attribution_mae=_f(r, "best_attribution_mae"),
                n_channels=_i(r, "n_channels"),
            ),
        )
        if not kpis:
            raise ValueError("kpis.csv is empty")
        self.kpis = kpis[0]

    # --- queries -----------------------------------------------------------

    def attribution_methods(self) -> List[str]:
        seen: list[str] = []
        for row in self.attribution:
            if row.method != "truth" and row.method not in seen:
                seen.append(row.method)
        return seen

    def attribution_for(self, method: Optional[str]) -> List[AttributionCredit]:
        if method is None:
            return self.attribution
        rows = [r for r in self.attribution if r.method == method]
        return rows

    def has_method(self, method: str) -> bool:
        return any(r.method == method for r in self.attribution)

    def customers(
        self, limit: int, offset: int, segment: Optional[str]
    ) -> tuple[List[ClvPerCustomer], int]:
        rows = self.clv_per_customer
        if segment is not None:
            rows = [r for r in rows if r.segment == segment]
        return rows[offset : offset + limit], len(rows)

    def row_counts(self) -> dict[str, int]:
        return {
            "segments": len(self.segments),
            "k_selection": len(self.k_selection),
            "clv_by_channel": len(self.clv_by_channel),
            "clv_summary": len(self.clv_summary),
            "clv_models": len(self.clv_models),
            "clv_per_customer": len(self.clv_per_customer),
            "attribution": len(self.attribution),
            "method_scores": len(self.method_scores),
            "budget_outcomes": len(self.budget_outcomes),
        }


def load_repository(data_dir: Optional[str] = None) -> Repository:
    return Repository(resolve_data_dir(data_dir))
