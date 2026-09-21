// Segmentation view: who the five clusters are, how concentrated value is across
// them, and why k=5 was chosen rather than asserted. All narrative numbers are
// computed from the loaded CSVs.

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Callout, Card, DataTable, type Column } from "../components/primitives";
import { colorFor } from "../components/colors";
import { useDataset } from "../data/store";
import {
  formatGbp,
  formatPct,
  kSelectionSummary,
  segmentHighlights,
  valueConcentrationCurve,
} from "../data/transforms";
import type { SegmentRow } from "../data/types";

/** The k the pipeline uses (see README, "Use 5 segments"). */
const CHOSEN_K = 5;

export function SegmentationView() {
  const { segments, kSelection } = useDataset();
  const hl = segmentHighlights(segments);
  const ks = kSelectionSummary(kSelection, CHOSEN_K);

  const byValue = [...segments].sort((a, b) => b.share_of_value - a.share_of_value);
  const curve = valueConcentrationCurve(segments);

  const columns: Column<SegmentRow>[] = [
    { key: "segment", header: "Segment", render: (r) => r.segment },
    {
      key: "customers",
      header: "Customers",
      render: (r) => r.customers.toLocaleString(),
    },
    {
      key: "share",
      header: "% of base",
      render: (r) => formatPct(r.share_of_customers),
    },
    { key: "freq", header: "Cal. freq", render: (r) => r.cal_frequency.toFixed(1) },
    {
      key: "recency",
      header: "Recency (d)",
      render: (r) => r.recency_days.toFixed(0),
    },
    {
      key: "churn",
      header: "No buy next 6 mo",
      render: (r) => formatPct(r.churn_rate),
    },
    {
      key: "clvmean",
      header: "CLV / cust",
      render: (r) => formatGbp(r.clv_mean),
    },
    {
      key: "valshare",
      header: "% of value",
      render: (r) => formatPct(r.share_of_value),
    },
  ];

  return (
    <div className="grid">
      <div className="grid cols-2">
        <Card
          title="Value share by segment"
          subtitle="Each cluster's share of total predicted CLV. Colour is kept per segment across the app."
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={byValue}
              margin={{ top: 6, right: 12, bottom: 4, left: 0 }}
            >
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis
                dataKey="segment"
                stroke="var(--muted)"
                tickLine={false}
              />
              <YAxis
                stroke="var(--muted)"
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                tickLine={false}
              />
              <Tooltip
                formatter={(v: number) => formatPct(v)}
                contentStyle={tooltipStyle}
              />
              <Bar
                dataKey="share_of_value"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              >
                {byValue.map((s) => (
                  <Cell key={s.segment} fill={colorFor(segmentIndex(s.segment))} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="Value concentration"
          subtitle="Segments ordered by value per customer, accumulated. The dashed line is perfectly even value; the gap above it is concentration."
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart
              data={curve}
              margin={{ top: 6, right: 12, bottom: 4, left: 0 }}
            >
              <CartesianGrid stroke="var(--grid)" />
              <XAxis
                dataKey="cumCustomers"
                type="number"
                domain={[0, 1]}
                stroke="var(--muted)"
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                tickLine={false}
              />
              <YAxis
                dataKey="cumValue"
                type="number"
                domain={[0, 1]}
                stroke="var(--muted)"
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                tickLine={false}
              />
              <Tooltip
                formatter={(v: number) => formatPct(v)}
                labelFormatter={() => ""}
                contentStyle={tooltipStyle}
              />
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ]}
                stroke="var(--muted)"
                strokeDasharray="5 5"
              />
              <Line
                dataKey="cumValue"
                stroke="var(--accent-2)"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card
        title="The five segments"
        subtitle="k-means on recency, frequency, spend, products per order and return rate over the first 18 months, then checked against what each segment did in the next 6."
      >
        <DataTable
          columns={columns}
          rows={byValue}
          rowKey={(r) => r.segment}
          isHighlighted={(r) => r.segment === hl?.top.segment}
        />
        {hl && (
          <Callout>
            <strong>{hl.top.segment}</strong> holds{" "}
            <strong>{formatPct(hl.top.share_of_value)}</strong> of predicted
            value on <strong>{formatPct(hl.top.share_of_customers)}</strong> of
            customers.{" "}
            {hl.richest.segment !== hl.top.segment && (
              <>
                <strong>{hl.richest.segment}</strong> is only{" "}
                {hl.richest.customers.toLocaleString()} customers (
                {formatPct(hl.richest.share_of_customers)}) averaging{" "}
                {hl.richest.cal_frequency.toFixed(0)} orders each, yet holds{" "}
                <strong>{formatPct(hl.richest.share_of_value)}</strong> of value -
                wholesale-sized accounts that skew every average.{" "}
              </>
            )}
            At the other end, <strong>{hl.poorest.segment}</strong> is{" "}
            {formatPct(hl.poorest.share_of_customers)} of customers and{" "}
            {formatPct(hl.poorest.share_of_value)} of value, and{" "}
            {formatPct(hl.poorest.churn_rate)} of them did not buy again in the
            next 6 months.
          </Callout>
        )}
      </Card>

      <Card
        title="Why k = 5"
        subtitle="Chosen by forward separation: how well segments formed on the first 18 months separate spend in the next 6 (adjusted η², left axis). Stability and smallest-cluster size are shown for reference (right axis)."
      >
        <ResponsiveContainer width="100%" height={260}>
          <LineChart
            data={kSelection}
            margin={{ top: 6, right: 4, bottom: 4, left: 0 }}
          >
            <CartesianGrid stroke="var(--grid)" />
            <XAxis dataKey="k" stroke="var(--muted)" tickLine={false} />
            <YAxis
              yAxisId="eta"
              stroke="var(--muted)"
              domain={[0, "auto"]}
              tickFormatter={(v: number) => v.toFixed(2)}
              tickLine={false}
            />
            <YAxis
              yAxisId="unit"
              orientation="right"
              stroke="var(--muted)"
              domain={[0, 1]}
              tickLine={false}
            />
            <Tooltip
              formatter={(v: number) => v.toFixed(3)}
              labelFormatter={(k: number) => `k = ${k}`}
              contentStyle={tooltipStyle}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine
              yAxisId="eta"
              x={CHOSEN_K}
              stroke="var(--accent)"
              strokeDasharray="4 4"
            />
            <Line
              yAxisId="eta"
              name="adjusted η² (forward separation)"
              dataKey="adjusted_eta_squared"
              stroke="var(--accent-2)"
              strokeWidth={2.5}
              dot={{ r: 3 }}
              isAnimationActive={false}
            />
            <Line
              yAxisId="unit"
              name="mean ARI (stability)"
              dataKey="mean_ari"
              stroke="var(--accent)"
              strokeWidth={1.5}
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
            <Line
              yAxisId="unit"
              name="smallest cluster share"
              dataKey="smallest_cluster_share"
              stroke="var(--warn)"
              strokeWidth={1.5}
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        {ks && (
          <Callout>
            Adjusted η² is{" "}
            {ks.previous && (
              <>
                {ks.previous.adjusted_eta_squared.toFixed(3)} at k=
                {ks.previous.k},{" "}
              </>
            )}
            <strong>
              {ks.chosen.adjusted_eta_squared.toFixed(3)} at k={ks.chosen.k}
            </strong>
            {ks.bestForward.k !== ks.chosen.k && (
              <>
                {" "}
                and peaks at {ks.bestForward.adjusted_eta_squared.toFixed(3)} at
                k={ks.bestForward.k}. k={ks.chosen.k} gets{" "}
                {formatPct(ks.shareOfBestForward, 0)} of that with fewer groups to
                act on
              </>
            )}
            . Silhouette picks k={ks.silhouetteK}, Davies-Bouldin k=
            {ks.daviesBouldinK} and stability k={ks.stabilityK}; those score how
            tidy the clusters look, not whether they predict the future.{" "}
            {ks.moreStableKs.length > 0 && (
              <>
                Stability is not the reason for k={ks.chosen.k}: its mean ARI is{" "}
                {ks.chosen.mean_ari.toFixed(2)},{" "}
                {ks.moreStableKs.length === kSelection.length - 1
                  ? "the lowest of any k tested"
                  : `below k=${ks.moreStableKs.join(", ")}`}
                .
              </>
            )}
          </Callout>
        )}
      </Card>
    </div>
  );
}

/** Stable colour index from a "Segment N" label. */
function segmentIndex(label: string): number {
  const m = label.match(/(\d+)/);
  return m ? Number(m[1]) : 0;
}

const tooltipStyle = {
  background: "var(--panel-2)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--text)",
  fontSize: 13,
};
