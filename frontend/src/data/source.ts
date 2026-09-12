// Chooses where the dashboard reads its data from:
//   - VITE_API_BASE set  -> the FastAPI REST API (full-stack mode)
//   - otherwise          -> the CSVs bundled under public/data (static mode)
// Static mode is the default so the GitHub Pages deploy is self-contained; point
// VITE_API_BASE at a running API (e.g. http://localhost:8000) to exercise the
// full stack. See ../../../api/README.md.

import { loadDatasetFromApi } from "./apiLoader";
import { loadDataset as loadDatasetFromCsv } from "./loaders";
import type { Dataset } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE as string | undefined;

export const dataSource: "api" | "static" = API_BASE ? "api" : "static";

export function loadDataset(): Promise<Dataset> {
  return API_BASE ? loadDatasetFromApi(API_BASE) : loadDatasetFromCsv();
}
