"""A Plotly view of the attribution result: credited share vs the planted truth.

    python -m src.plotly_view        # writes out/attribution_plotly.html

The repo already computes, in ``run_analytics.py``, the credited share of
conversions each attribution method assigns to each channel, scored against the
**true** incremental effect share from ``TRUTH.json``. This renders that same
table as an interactive grouped bar chart: for every channel, the true effect
share next to what each method credits it. The gap between a method's bar and
the TRUTH bar is the method's error; the planted **zero-effect channel**
(``retargeting``) is annotated, because every observational method over-credits
it. Attribution is not incrementality.

Self-contained HTML (Plotly bundled inline), so it opens offline with no server.
Requires the generated data (``python -m src.generate``).
"""
from __future__ import annotations

import json
import os

import numpy as np
import plotly.graph_objects as go

from src import attribution as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")


def _load():
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)
    with open(os.path.join(DATA, "TRUTH.json")) as f:
        truth = json.load(f)
    return jd, truth


def credited_shares(jd: dict, truth: dict) -> tuple[list[str], dict]:
    """Credited share by channel for each method, plus the TRUTH row."""
    journeys, conversions = jd["journeys"], jd["conversions"]
    touch_days = jd.get("touch_days")
    channels = list(truth["channel_effects"])
    series = {"TRUTH": {ch: truth["true_effect_share"][ch] for ch in channels}}
    for name, fn in A.METHODS.items():
        if name == "time_decay":
            credit = fn(journeys, conversions, channels, touch_days=touch_days)
        else:
            credit = fn(journeys, conversions, channels)
        series[name] = {ch: credit[ch] for ch in channels}
    return channels, series


def figure(channels: list[str], series: dict, zero_channel: str) -> go.Figure:
    fig = go.Figure()
    for name, shares in series.items():
        is_truth = name == "TRUTH"
        fig.add_bar(
            name=name,
            x=channels,
            y=[shares[ch] for ch in channels],
            marker_line_width=2 if is_truth else 0,
            opacity=1.0 if is_truth else 0.85,
        )
    # MAE subtitle per method (how far each sits from truth, averaged over channels)
    truth = series["TRUTH"]
    maes = {
        name: float(np.mean([abs(shares[ch] - truth[ch]) for ch in channels]))
        for name, shares in series.items() if name != "TRUTH"
    }
    ranked = ", ".join(f"{m} {v:.3f}" for m, v in sorted(maes.items(), key=lambda kv: kv[1]))
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Attribution credited share vs TRUE effect share, by channel"
            f"<br><sub>mean abs. error vs truth (lower is better): {ranked}</sub>"
        ),
        xaxis_title="channel",
        yaxis_title="share of conversions credited",
        legend_title="method",
        template="plotly_white",
        bargap=0.25,
    )
    # Annotate the planted zero-effect channel.
    if zero_channel in channels:
        ymax = max(max(s.values()) for s in series.values())
        fig.add_annotation(
            x=zero_channel, y=ymax,
            text=f"{zero_channel}: TRUE effect = 0<br>(every method over-credits it)",
            showarrow=True, arrowhead=2, yshift=10, font=dict(size=11),
        )
    return fig


def build(out_path: str | None = None) -> str:
    jd, truth = _load()
    channels, series = credited_shares(jd, truth)
    fig = figure(channels, series, truth.get("zero_effect_channel", ""))
    os.makedirs(OUT, exist_ok=True)
    out_path = out_path or os.path.join(OUT, "attribution_plotly.html")
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return out_path


def main() -> int:
    if not os.path.exists(os.path.join(DATA, "journeys.json")):
        print("No data found. Run `python -m src.generate` first.")
        return 1
    path = build()
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
