// Attribution view: how each method's per-channel credit compares to the known
// truth of the simulated ad journeys (the only simulated part of this project), how the methods rank on error, and what choosing the
// wrong one costs in conversions - including the planted zero-effect channel that
// every method over-credits.

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Callout, Card, DataTable, type Column } from "../components/primitives";
import { useDataset } from "../data/store";
import {
  attributionForMethod,
  attributionMethods,
  formatGbp,
  formatPct,
  zeroEffectChannel,
} from "../data/transforms";
import type { BudgetOutcomeRow } from "../data/types";

const METHOD_LABELS: Record<string, string> = {
  last_touch: "Last touch",
  first_touch: "First touch",
  linear: "Linear",
  time_decay: "Time decay",
  markov_removal: "Markov removal",
  shapley: "Shapley",
};

export function AttributionView() {
  const { attribution, methodScores, budgetOutcomes } = useDataset();
  const methods = attributionMethods(attribution);
  const zeroChannel = zeroEffectChannel(attribution);

  const bestMethod = [...methodScores].sort((a, b) => a.mae - b.mae)[0];
  const [method, setMethod] = useState<string>(bestMethod?.method ?? methods[0]);

  const perChannel = attributionForMethod(attribution, method).map((r) => ({
    ...r,
    label: r.channel === zeroChannel ? `${r.channel} *` : r.channel,
  }));

  const scoreRows = [...methodScores].sort((a, b) => a.mae - b.mae);

  const budgetCols: Column<BudgetOutcomeRow>[] = [
    {
      key: "alloc",
      header: "Allocation",
      render: (r) =>
        r.allocation === "TRUTH"
          ? "Truth (oracle)"
          : (METHOD_LABELS[r.allocation] ?? r.allocation),
    },
    {
      key: "conv",
      header: "Conversions",
      render: (r) => r.conversions.toFixed(0),
    },
    {
      key: "lost",
      header: "% conv. lost",
      render: (r) =>
        r.allocation === "TRUTH" ? "-" : formatPct(r.pct_conversions_lost / 100),
    },
    {
      key: "value",
      header: "Customer value",
      render: (r) => formatGbp(r.customer_value),
    },
    {
      key: "vlost",
      header: "% value lost",
      render: (r) =>
        r.allocation === "TRUTH" ? "-" : formatPct(r.pct_value_lost / 100),
    },
  ];

  const zeroRow = perChannel.find((r) => r.channel === zeroChannel);
  const creditedZero = methodScores.filter(
    (m) => m.credit_to_zero_effect > 0,
  ).length;

  const methodRows = budgetOutcomes.filter((r) => r.allocation !== "TRUTH");
  const bestBudget = methodRows.length
    ? methodRows.reduce((a, b) =>
        b.pct_conversions_lost < a.pct_conversions_lost ? b : a,
      )
    : undefined;
  const worstBudget = methodRows.length
    ? methodRows.reduce((a, b) =>
        b.pct_conversions_lost > a.pct_conversions_lost ? b : a,
      )
    : undefined;
  const lastTouch = methodRows.find((r) => r.allocation === "last_touch");
  const label = (m: string) => METHOD_LABELS[m] ?? m;

  return (
    <div className="grid">
      <Card
        title="Credit vs truth, by method"
        subtitle="Ad journeys are simulated for the real customers, with channel effects we set. For each method, the credit it gives each channel next to that known true effect. Channels ordered by true effect."
      >
        <div className="chip-row">
          {methods.map((m) => (
            <button
              key={m}
              className={`chip ${method === m ? "active" : ""}`}
              onClick={() => setMethod(m)}
            >
              {METHOD_LABELS[m] ?? m}
            </button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart
            data={perChannel}
            margin={{ top: 6, right: 12, bottom: 24, left: 0 }}
          >
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis
              dataKey="label"
              stroke="var(--muted)"
              tickLine={false}
              angle={-35}
              textAnchor="end"
              height={60}
              interval={0}
            />
            <YAxis
              stroke="var(--muted)"
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              tickLine={false}
            />
            <Tooltip
              formatter={(v: number, name: string) => [formatPct(v), name]}
              contentStyle={tooltipStyle}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar
              name="Assigned credit"
              dataKey="credit"
              fill="var(--accent)"
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              name="True effect"
              dataKey="truth"
              fill="var(--accent-2)"
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
        {zeroChannel && zeroRow && (
          <Callout>
            <strong>* {zeroChannel}</strong> is the planted incrementality
            control: its true effect is <strong>zero</strong>, yet{" "}
            {METHOD_LABELS[method] ?? method} still hands it{" "}
            <strong>{formatPct(zeroRow.credit)}</strong> of the credit.{" "}
            {creditedZero === methodScores.length
              ? "Every method gives"
              : `${creditedZero} of ${methodScores.length} methods give`}{" "}
            it some credit, because it is shown to people already about to
            buy. Only an experiment (switching it off in some regions) can catch
            that.
          </Callout>
        )}
      </Card>

      <div className="grid cols-2">
        <Card
          title="Method accuracy"
          subtitle="Mean absolute error against truth (x) vs credit wrongly given to the zero-effect channel (y). Bottom-left is best."
        >
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 10, right: 16, bottom: 20, left: 4 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis
                type="number"
                dataKey="mae"
                name="MAE"
                stroke="var(--muted)"
                tickLine={false}
                label={{
                  value: "MAE vs truth",
                  position: "insideBottom",
                  offset: -8,
                  fill: "var(--muted)",
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                dataKey="credit_to_zero_effect"
                name="Zero-effect credit"
                stroke="var(--muted)"
                tickLine={false}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              />
              <ZAxis range={[120, 120]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                formatter={(v: number, name: string) =>
                  name === "Zero-effect credit"
                    ? [formatPct(v), name]
                    : [v.toFixed(4), name]
                }
                contentStyle={tooltipStyle}
              />
              <Scatter data={scoreRows} isAnimationActive={false}>
                {scoreRows.map((s) => (
                  <Cell
                    key={s.method}
                    fill={
                      s.method === bestMethod.method
                        ? "var(--good)"
                        : "var(--accent)"
                    }
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <DataTable
            columns={[
              {
                key: "m",
                header: "Method",
                render: (r) => METHOD_LABELS[r.method] ?? r.method,
              },
              { key: "mae", header: "MAE", render: (r) => r.mae.toFixed(4) },
              {
                key: "z",
                header: "Zero-effect credit",
                render: (r) => formatPct(r.credit_to_zero_effect),
              },
            ]}
            rows={scoreRows}
            rowKey={(r) => r.method}
            isHighlighted={(r) => r.method === bestMethod.method}
          />
        </Card>

        <Card
          title="What the wrong method costs"
          subtitle="Split the budget by each method's credit, then count the sales it really produces under the known channel effects. Conversions and customer value lost versus budgeting on the truth."
        >
          <DataTable
            columns={budgetCols}
            rows={budgetOutcomes}
            rowKey={(r) => r.allocation}
            isHighlighted={(r) => r.allocation === "TRUTH"}
          />
          <Callout>
            {label(bestMethod.method)} is closest to the true credit (MAE{" "}
            {bestMethod.mae.toFixed(4)})
            {bestBudget && bestBudget.allocation !== bestMethod.method ? (
              <>
                , but <strong>{label(bestBudget.allocation)}</strong> gives the
                best budget, losing only{" "}
                {formatPct(bestBudget.pct_conversions_lost / 100)} of sales. The
                lowest error is not the same as the best decision.
              </>
            ) : (
              "."
            )}
            {worstBudget && (
              <>
                {" "}
                <strong>{label(worstBudget.allocation)}</strong> does worst: it
                loses {formatPct(worstBudget.pct_conversions_lost / 100)} of
                sales.
              </>
            )}
            {lastTouch && zeroChannel && (
              <>
                {" "}
                Last touch loses{" "}
                {formatPct(lastTouch.pct_conversions_lost / 100)} and puts{" "}
                {formatGbp(lastTouch.spend_on_zero_effect)} into {zeroChannel}.
              </>
            )}
          </Callout>
        </Card>
      </div>
    </div>
  );
}

const tooltipStyle = {
  background: "var(--panel-2)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--text)",
  fontSize: 13,
};
