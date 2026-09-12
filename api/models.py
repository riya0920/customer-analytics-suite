"""Pydantic response models. Field names match the exported CSV columns exactly
so the API contract, the CSVs, and the React types in frontend/src/data/types.ts
all line up. These models are the single source of truth for the JSON shapes and
for the OpenAPI schema FastAPI generates.
"""
from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Segment(BaseModel):
    segment: str
    customers: int
    share_of_customers: float
    cal_frequency: float
    cal_monetary: float
    recency_days: float
    churn_rate: float
    holdout_orders: float
    holdout_spend: float
    clv_total: float
    clv_mean: float
    share_of_value: float


class KSelection(BaseModel):
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    inertia: float
    mean_ari: float
    adjusted_eta_squared: float
    smallest_cluster_share: float


class ClvByChannel(BaseModel):
    channel: str
    converters_touched: int
    mean_clv: float
    median_clv: float
    total_clv: float
    clv_index: float


class ClvSummary(BaseModel):
    metric: str
    value: float


class ClvModel(BaseModel):
    model: str
    spearman: float
    mae: float


class ClvPerCustomer(BaseModel):
    customer_id: int
    segment: str
    predicted_clv: float
    holdout_spend: float
    cal_frequency: float
    recency_days: float


class AttributionCredit(BaseModel):
    method: str
    channel: str
    credit: float
    truth: float
    error: float


class MethodScore(BaseModel):
    method: str
    mae: float
    credit_to_zero_effect: float


class BudgetOutcome(BaseModel):
    allocation: str
    conversions: float
    conversions_lost_vs_truth: float
    pct_conversions_lost: float
    spend_on_zero_effect: float
    customer_value: float
    pct_value_lost: float


class Kpi(BaseModel):
    total_customers: int
    n_segments: int
    top20pct_value_share: float
    clv_rank_spearman: float
    best_attribution_mae: float
    n_channels: int


class Page(BaseModel, Generic[T]):
    """A slice of a larger collection, with the cursor echoed back so a client can
    page without guessing."""

    items: List[T]
    total: int
    limit: int
    offset: int


class Health(BaseModel):
    status: str
    data_dir: str
    row_counts: dict[str, int]


class ErrorResponse(BaseModel):
    """Uniform error body returned for 404/422/500 (documented in OpenAPI)."""

    error: str
    detail: Optional[str] = Field(default=None)
