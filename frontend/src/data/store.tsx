// App-wide data store. The async CSV load is modeled as an explicit state
// machine (idle -> loading -> ready | error) driven by a reducer, exposed
// through context. Components read it with the useData() hook and never fetch
// themselves, so loading/error handling lives in exactly one place.

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type ReactNode,
} from "react";
import { loadDataset } from "./loaders";
import type { Dataset } from "./types";

export type DataState =
  | { status: "loading" }
  | { status: "ready"; data: Dataset }
  | { status: "error"; error: string };

type Action =
  | { type: "loading" }
  | { type: "ready"; data: Dataset }
  | { type: "error"; error: string };

export function dataReducer(_state: DataState, action: Action): DataState {
  switch (action.type) {
    case "loading":
      return { status: "loading" };
    case "ready":
      return { status: "ready", data: action.data };
    case "error":
      return { status: "error", error: action.error };
  }
}

const DataContext = createContext<DataState | undefined>(undefined);

export function DataProvider({
  children,
  loader = loadDataset,
}: {
  children: ReactNode;
  /** Injectable for tests so a component tree can be given data synchronously. */
  loader?: () => Promise<Dataset>;
}) {
  const [state, dispatch] = useReducer(dataReducer, { status: "loading" });

  useEffect(() => {
    let cancelled = false;
    dispatch({ type: "loading" });
    loader()
      .then((data) => {
        if (!cancelled) dispatch({ type: "ready", data });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          dispatch({
            type: "error",
            error: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [loader]);

  return <DataContext.Provider value={state}>{children}</DataContext.Provider>;
}

export function useData(): DataState {
  const ctx = useContext(DataContext);
  if (ctx === undefined) {
    throw new Error("useData must be used within a DataProvider");
  }
  return ctx;
}

/** Convenience for components that only run once data is ready. */
export function useDataset(): Dataset {
  const state = useData();
  if (state.status !== "ready") {
    throw new Error("useDataset called before data was ready");
  }
  // useMemo keeps referential stability across re-renders of the same state.
  return useMemo(() => state.data, [state]);
}
