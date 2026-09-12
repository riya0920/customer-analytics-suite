// CLV view: the model's calibration quality (BG/NBD + Gamma-Gamma vs a GBM
// challenger), the distribution of predicted value, and value by acquiring
// channel.

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Callout, Card, DataTable, type Column } from "../components/primitives";
import { StatCard } from "../components/primitives";
import { colorFor } from "../components/colors";
import { useDataset } from "../data/store";
import { clvHistogram, formatUsd } from "../data/transforms";
import type { ClvByChannelRow } from "../data/types";

type ChannelSort = "total_clv" | "mean_clv" | "converters_touched";

export function ClvView() {
  const { clvByChannel, clvModels, clvSummary, clvPerCustomer, kpis } =
    useDataset();
  const [sort, setSort] = useState<ChannelSort>("total_clv");

  const hist = clvHistogram(clvPerCustomer, 30, 600);
  const channels = [...clvByChannel].sort((a, b) => b[sort] - a[sort]);

  const summary = (metric: string) =>
    clvSummary.find((r) => r.metric === metric)?.value;

  const best = clvModels.reduce((a, b) => (a.mae <= b.mae ? a : b));

  const columns: Column<ClvByChannelRow>[] = [
    { key: "channel", header: "Channel", render: (r) => r.channel },
    {
      key: "conv",
      header: "Converters",
      render: (r) => r.converters_touched.toLocaleString(),
    },
    { key: "mean", header: "Mean CLV", render: (r) => formatUsd(r.mean_clv) },
    {
      key: "median",
      header: "Median CLV",
      render: (r) => formatUsd(r.median_clv),
    },
    { key: "total", header: "Total CLV", render: (r) => formatUsd(r.total_clv) },
    {
      key: "index",
      header: "CLV index",
      render: (r) => r.clv_index.toFixed(3),
    },
  ];

  return (
    <div className="grid">
      <div className="kpi-bar">
        <StatCard
          label="Rank agreement"
          value={(summary("Spearman (rank, holdout)") ?? 0).toFixed(3)}
          sub="Spearman vs holdout spend"
        />
        <StatCard
          label="Individual corr."
          value={(summary("Pearson (individual)") ?? 0).toFixed(3)}
          sub="Pearson, per customer"
        />
        <StatCard
          label="Top-20% value share"
          value={`${Math.round((summary("Top-20% share of value") ?? 0) * 100)}%`}
          sub="Concentration of predicted CLV"
        />
        <StatCard
          label="Best model MAE"
          value={formatUsd(best.mae)}
          sub={best.model}
        />
      </div>

      <div className="grid cols-2">
        <Card
          title="Predicted CLV distribution"
          subtitle={`All ${kpis.total_customers.toLocaleString()} customers, predicted CLV clipped at $600 (tail folded into the last bin). Most customers are worth little; a thin right tail carries the value.`}
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={hist} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis
                dataKey="mid"
                stroke="var(--muted)"
                tickFormatter={(v: number) => `$${Math.round(v)}`}
                tickLine={false}
                interval={4}
              />
              <YAxis stroke="var(--muted)" tickLine={false} />
              <Tooltip
                labelFormatter={(v: number) => `~$${Math.round(v)} CLV`}
                formatter={(v: number) => [v.toLocaleString(), "customers"]}
                contentStyle={tooltipStyle}
              />
              <Bar dataKey="count" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="Model calibration: BG/NBD vs GBM challenger"
          subtitle="Rank correlation against holdout spend, and mean absolute error in dollars. The probabilistic model is checked against a gradient-boosted challenger, not trusted on faith."
        >
          <DataTable
            columns={[
              { key: "model", header: "Model", render: (r) => r.model },
              {
                key: "spearman",
                header: "Spearman",
                render: (r) => r.spearman.toFixed(4),
              },
              { key: "mae", header: "MAE ($)", render: (r) => r.mae.toFixed(2) },
            ]}
            rows={clvModels}
            rowKey={(r) => r.model}
            isHighlighted={(r) => r.model === best.model}
          />
          <Callout>
            The BG/NBD + Gamma-Gamma model wins on both rank agreement and dollar
            error, so the interpretable probabilistic model is also the more
            accurate one here - no accuracy is being traded away for its closed
            form.
          </Callout>
        </Card>
      </div>

      <Card
        title="CLV by acquiring channel"
        subtitle="Mean/median/total predicted CLV of customers touched by each channel. The CLV index is a channel's mean relative to the overall mean (1.0 = average)."
      >
        <div className="chip-row">
          {(
            [
              ["total_clv", "Total CLV"],
              ["mean_clv", "Mean CLV"],
              ["converters_touched", "Converters"],
            ] as [ChannelSort, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              className={`chip ${sort === key ? "active" : ""}`}
              onClick={() => setSort(key)}
            >
              Sort: {label}
            </button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={channels}
            layout="vertical"
            margin={{ top: 4, right: 16, bottom: 4, left: 24 }}
          >
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" stroke="var(--muted)" tickLine={false} />
            <YAxis
              type="category"
              dataKey="channel"
              stroke="var(--muted)"
              width={92}
              tickLine={false}
            />
            <Tooltip
              formatter={(v: number) =>
                sort === "converters_touched"
                  ? v.toLocaleString()
                  : formatUsd(v)
              }
              contentStyle={tooltipStyle}
            />
            <Bar dataKey={sort} radius={[0, 4, 4, 0]}>
              {channels.map((c, i) => (
                <Cell key={c.channel} fill={colorFor(i)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <DataTable
          columns={columns}
          rows={channels}
          rowKey={(r) => r.channel}
        />
      </Card>
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
