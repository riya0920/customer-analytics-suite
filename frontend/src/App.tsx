import { useState } from "react";
import { StatCard } from "./components/primitives";
import { useData } from "./data/store";
import { formatGbp, formatPct } from "./data/transforms";
import { SegmentationView } from "./views/SegmentationView";
import { ClvView } from "./views/ClvView";
import { AttributionView } from "./views/AttributionView";

type Tab = "segmentation" | "clv" | "attribution";

const TABS: { id: Tab; label: string }[] = [
  { id: "segmentation", label: "Segmentation" },
  { id: "clv", label: "Customer Lifetime Value" },
  { id: "attribution", label: "Attribution" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("segmentation");
  const state = useData();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Customer Analytics - Segmentation · CLV · Attribution</h1>
        <p>
          Real transactions from a UK online gift retailer (UCI Online Retail II,
          Dec 2009 - Dec 2011, amounts in £). Segments and lifetime value come from
          that real data. The ad journeys are simulated, with channel effects we
          set, so each attribution method can be scored against a known truth.
          Every figure is read from the exported model results.
        </p>
      </header>

      {state.status === "loading" && (
        <div className="state">Loading model outputs…</div>
      )}

      {state.status === "error" && (
        <div className="state error">
          Could not load the data: {state.error}
        </div>
      )}

      {state.status === "ready" && (
        <>
          <KpiBar />
          <nav className="tabs" role="tablist" aria-label="Views">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                className={`tab ${tab === t.id ? "active" : ""}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>

          {tab === "segmentation" && <SegmentationView />}
          {tab === "clv" && <ClvView />}
          {tab === "attribution" && <AttributionView />}
        </>
      )}

      <footer className="app-footer">
        Data: exported results of the{" "}
        <a
          href="https://github.com/riya0920/customer-analytics-suite"
          target="_blank"
          rel="noreferrer"
        >
          customer-analytics-suite
        </a>{" "}
        dbt/Python pipeline on UCI Online Retail II (real segments and BG/NBD +
        Gamma-Gamma CLV; attribution over simulated ad journeys with known channel
        effects). This UI reads the CSVs verbatim and computes only
        presentation-layer aggregates.
      </footer>
    </div>
  );
}

function KpiBar() {
  const state = useData();
  if (state.status !== "ready") return null;
  const k = state.data.kpis;
  return (
    <div className="kpi-bar">
      <StatCard
        label="Customers"
        value={k.total_customers.toLocaleString()}
        sub={`${k.n_segments} segments`}
      />
      <StatCard
        label="Channels"
        value={k.n_channels}
        sub="multi-touch"
      />
      <StatCard
        label="Top-20% value share"
        value={formatPct(k.top20pct_value_share)}
        sub="value concentration"
      />
      <StatCard
        label="CLV rank agreement"
        value={k.clv_rank_spearman.toFixed(3)}
        sub="Spearman vs holdout"
      />
      <StatCard
        label="Best attribution MAE"
        value={k.best_attribution_mae.toFixed(4)}
        sub="vs known truth"
      />
      <StatCard
        label="Value at stake"
        value={formatGbp(
          state.data.segments.reduce((s, r) => s + r.clv_total, 0),
        )}
        sub="total predicted CLV, test period"
      />
    </div>
  );
}
