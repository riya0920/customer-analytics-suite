// Alternate data source: assemble the same typed Dataset from the FastAPI REST
// API (api/) instead of the bundled CSVs. Selected at build/run time by
// VITE_API_BASE (see source.ts). The endpoints already return JSON in the row
// shapes declared in types.ts, so this maps mostly by fetching — except
// /api/clv/customers, which is paginated and is walked to completion here.

import type {
  AttributionCreditRow,
  BudgetOutcomeRow,
  ClvByChannelRow,
  ClvModelRow,
  ClvPerCustomerRow,
  ClvSummaryRow,
  Dataset,
  KpiRow,
  KSelectionRow,
  MethodScoreRow,
  SegmentRow,
} from "./types";

interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

async function getJson<T>(base: string, path: string): Promise<T> {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status */
    }
    throw new Error(`GET ${path} failed: ${detail}`);
  }
  return res.json() as Promise<T>;
}

/** Walk the paginated customers endpoint until every row is fetched. */
async function getAllCustomers(base: string): Promise<ClvPerCustomerRow[]> {
  const limit = 1000;
  const first = await getJson<Page<ClvPerCustomerRow>>(
    base,
    `/api/clv/customers?limit=${limit}&offset=0`,
  );
  const rows = [...first.items];
  for (let offset = limit; offset < first.total; offset += limit) {
    const page = await getJson<Page<ClvPerCustomerRow>>(
      base,
      `/api/clv/customers?limit=${limit}&offset=${offset}`,
    );
    rows.push(...page.items);
  }
  return rows;
}

export async function loadDatasetFromApi(base: string): Promise<Dataset> {
  const root = base.replace(/\/$/, "");
  const [
    kpis,
    segments,
    kSelection,
    clvSummary,
    clvModels,
    clvByChannel,
    attribution,
    methodScores,
    budgetOutcomes,
    clvPerCustomer,
  ] = await Promise.all([
    getJson<KpiRow>(root, "/api/kpis"),
    getJson<SegmentRow[]>(root, "/api/segments"),
    getJson<KSelectionRow[]>(root, "/api/segments/k-selection"),
    getJson<ClvSummaryRow[]>(root, "/api/clv/summary"),
    getJson<ClvModelRow[]>(root, "/api/clv/models"),
    getJson<ClvByChannelRow[]>(root, "/api/clv/channels"),
    getJson<AttributionCreditRow[]>(root, "/api/attribution/credit"),
    getJson<MethodScoreRow[]>(root, "/api/attribution/methods"),
    getJson<BudgetOutcomeRow[]>(root, "/api/attribution/budget"),
    getAllCustomers(root),
  ]);
  return {
    kpis,
    segments,
    kSelection,
    clvSummary,
    clvModels,
    clvByChannel,
    attribution,
    methodScores,
    budgetOutcomes,
    clvPerCustomer,
  };
}
