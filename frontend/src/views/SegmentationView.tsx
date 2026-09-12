// Segmentation view: who the five clusters are, how concentrated value is across
// them, and why k=5 was chosen rather than asserted.

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { formatPct, formatUsd, valueConcentrationCurve } from "../data/transforms";
import type { SegmentRow } from "../data/types";

export function SegmentationView() {
  const { segments, kSelection } = useDataset();

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
    { key: "churn", header: "Churn", render: (r) => formatPct(r.churn_rate) },
    {
      key: "clvmean",
      header: "CLV / cust",
      render: (r) => formatUsd(r.clv_mean),
    },
    {
      key: "valshare",
      header: "% of value",
      render: (r) => formatPct(r.share_of_value),
    },
  ];

  const topSegment = byValue[0];

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
              <Bar dataKey="share_of_value" radius={[4, 4, 0, 0]}>
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
        subtitle="RFM + behavioural clustering on the calibration window, with the forward-tested holdout outcome per segment."
      >
        <DataTable
          columns={columns}
          rows={byValue}
          rowKey={(r) => r.segment}
          isHighlighted={(r) => r.segment === topSegment.segment}
        />
        <Callout>
          <strong>{topSegment.segment}</strong> holds{" "}
          <strong>{formatPct(topSegment.share_of_value)}</strong> of predicted
          value on{" "}
          <strong>{formatPct(topSegment.share_of_customers)}</strong> of
          customers. The two large high-recency/high-churn clusters (Segments 1 &
          3) are ~45% of the base but under 4% of value - the map a flat
          "everyone gets the same email" plan ignores.
        </Callout>
      </Card>

      <Card
        title="Why k = 5"
        subtitle="k was selected, not assumed: bootstrap cluster stability (mean ARI) stays high at k=5 while the smallest cluster is still a usable size, and the elbow in inertia has flattened."
      >
        <ResponsiveContainer width="100%" height={240}>
          <LineChart
            data={kSelection}
            margin={{ top: 6, right: 16, bottom: 4, left: 0 }}
          >
            <CartesianGrid stroke="var(--grid)" />
            <XAxis dataKey="k" stroke="var(--muted)" tickLine={false} />
            <YAxis
              stroke="var(--muted)"
              domain={[0, 1]}
              tickLine={false}
            />
            <Tooltip contentStyle={tooltipStyle} />
            <ReferenceLine x={5} stroke="var(--accent)" strokeDasharray="4 4" />
            <Line
              name="mean ARI (stability)"
              dataKey="mean_ari"
              stroke="var(--accent)"
              strokeWidth={2}
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
            <Line
              name="smallest cluster share"
              dataKey="smallest_cluster_share"
              stroke="var(--warn)"
              strokeWidth={2}
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        <Callout>
          At k=5 bootstrap stability (mean ARI) is{" "}
          <strong>
            {kSelection.find((r) => r.k === 5)?.mean_ari.toFixed(2) ?? "-"}
          </strong>{" "}
          and the smallest cluster still holds{" "}
          <strong>
            {formatPct(
              kSelection.find((r) => r.k === 5)?.smallest_cluster_share ?? 0,
            )}
          </strong>{" "}
          of customers. Push to k=6+ and stability drops and clusters splinter.
        </Callout>
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
