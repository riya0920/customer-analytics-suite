// CLV view: the model's calibration quality (BG/NBD + Gamma-Gamma vs a GBM
// challenger), the distribution of predicted value, and value by the (simulated)
// channels that touched each customer.

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
import {
  clvHistogram,
  formatGbp,
  formatPct,
  zeroEffectChannel,
} from "../data/transforms";
import type { ClvByChannelRow } from "../data/types";

type ChannelSort = "total_clv" | "mean_clv" | "converters_touched";

// Histogram range: ~94% of real customers fall below the clip; the rest are
// folded into the last bin so a few giant accounts don't flatten the chart.
const HIST_BINS = 40;
const HIST_CLIP = 2000;

export function ClvView() {
  const {
    clvByChannel,
    clvModels,
    clvSummary,
    clvPerCustomer,
    kpis,
    attribution,
  } = useDataset();
  const [sort, setSort] = useState<ChannelSort>("total_clv");

  const hist = clvHistogram(clvPerCustomer, HIST_BINS, HIST_CLIP);
  const binWidth = HIST_CLIP / HIST_BINS;
  const shareAboveClip =
    clvPerCustomer.length > 0
      ? clvPerCustomer.filter((r) => r.predicted_clv >= HIST_CLIP).length /
        clvPerCustomer.length
      : 0;
  const sortedClv = clvPerCustomer
    .map((r) => r.predicted_clv)
    .sort((a, b) => a - b);
  const medianClv = sortedClv.length
    ? sortedClv[Math.floor(sortedClv.length / 2)]
    : 0;
  const channels = [...clvByChannel].sort((a, b) => b[sort] - a[sort]);
  const topIndex = clvByChannel.length
    ? clvByChannel.reduce((a, b) => (b.clv_index > a.clv_index ? b : a))
    : undefined;
  const zeroChannel = zeroEffectChannel(attribution);

  const summary = (metric: string) =>
    clvSummary.find((r) => r.metric === metric)?.value;

  const best = clvModels.reduce((a, b) => (a.mae <= b.mae ? a : b));
  const bestRank = clvModels.reduce((a, b) =>
    a.spearman >= b.spearman ? a : b,
  );
  const others = clvModels.filter((m) => m.model !== best.model);

  const columns: Column<ClvByChannelRow>[] = [
    { key: "channel", header: "Channel", render: (r) => r.channel },
    {
      key: "conv",
      header: "Converters",
      render: (r) => r.converters_touched.toLocaleString(),
    },
    { key: "mean", header: "Mean CLV", render: (r) => formatGbp(r.mean_clv) },
    {
      key: "median",
      header: "Median CLV",
      render: (r) => formatGbp(r.median_clv),
    },
    { key: "total", header: "Total CLV", render: (r) => formatGbp(r.total_clv) },
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
          sub="Pearson; inflated by a few huge accounts"
        />
        <StatCard
          label="Top-20% value share"
          value={`${Math.round((summary("Top-20% share of value") ?? 0) * 100)}%`}
          sub="Concentration of predicted CLV"
        />
        <StatCard
          label="Best model MAE"
          value={formatGbp(best.mae)}
          sub={best.model}
        />
      </div>

      <div className="grid cols-2">
        <Card
          title="Predicted CLV distribution"
          subtitle={`All ${kpis.total_customers.toLocaleString()} customers' predicted spend over the test period. Median ${formatGbp(medianClv)}; the ${formatPct(shareAboveClip)} above ${formatGbp(HIST_CLIP)} are folded into the last bin. A thin right tail carries much of the value.`}
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={hist} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis
                dataKey="x0"
                stroke="var(--muted)"
                tickFormatter={(v: number) => formatGbp(v)}
                tickLine={false}
                interval={7}
              />
              <YAxis stroke="var(--muted)" tickLine={false} />
              <Tooltip
                labelFormatter={(v: number) =>
                  v + binWidth >= HIST_CLIP
                    ? `${formatGbp(v)}+ CLV`
                    : `${formatGbp(v)} - ${formatGbp(v + binWidth)} CLV`
                }
                formatter={(v: number) => [v.toLocaleString(), "customers"]}
                contentStyle={tooltipStyle}
              />
              <Bar
                dataKey="count"
                fill="var(--accent)"
                radius={[3, 3, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="Model calibration: BG/NBD vs GBM challenger"
          subtitle="Rank correlation against real spend in the test period, and mean absolute error in pounds, on the same held-out customers. The probabilistic model is checked against a gradient-boosted challenger, not trusted on faith."
        >
          <DataTable
            columns={[
              { key: "model", header: "Model", render: (r) => r.model },
              {
                key: "spearman",
                header: "Spearman",
                render: (r) => r.spearman.toFixed(4),
              },
              { key: "mae", header: "MAE (£)", render: (r) => r.mae.toFixed(2) },
            ]}
            rows={clvModels}
            rowKey={(r) => r.model}
            isHighlighted={(r) => r.model === best.model}
          />
          <Callout>
            {bestRank.model === best.model ? (
              <>
                <strong>{best.model}</strong> wins on both rank agreement (
                {[best, ...others].map((m) => m.spearman.toFixed(2)).join(" vs ")})
                and error (
                {[best, ...others].map((m) => formatGbp(m.mae)).join(" vs ")}), so
                the interpretable model is also the more accurate one here.
              </>
            ) : (
              <>
                <strong>{bestRank.model}</strong> ranks customers better (
                {bestRank.spearman.toFixed(2)}), but{" "}
                <strong>{best.model}</strong> has the lower error (
                {formatGbp(best.mae)}).
              </>
            )}{" "}
            Use it to rank and size groups, not to price one customer.
          </Callout>
        </Card>
      </div>

      <Card
        title="CLV by channel touched"
        subtitle="Mean/median/total predicted CLV of converting customers each simulated channel touched. The CLV index is a channel's mean relative to the overall mean (1.0 = average). It shows who a channel reaches, not what it causes."
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
                  : formatGbp(v)
              }
              contentStyle={tooltipStyle}
            />
            <Bar dataKey={sort} radius={[0, 4, 4, 0]} isAnimationActive={false}>
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
        {topIndex && (
          <Callout>
            <strong>{topIndex.channel}</strong> has the highest CLV index (
            {topIndex.clv_index.toFixed(2)}x the average).
            {topIndex.channel === zeroChannel
              ? " It is the channel with zero real effect: it is shown to people already about to buy, and those are the best buyers. Weighting budget by this index would reward it twice for the same bias."
              : " The index is correlational: a high value can mean the channel reaches customers who were already valuable."}
          </Callout>
        )}
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
