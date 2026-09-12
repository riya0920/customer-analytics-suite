"""Guards on the Plotly attribution view.

The figure must carry a TRUTH series plus one trace per attribution method, and
the build path must emit a self-contained HTML when the generated data exists.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import plotly_view as PV  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def test_figure_has_truth_and_one_trace_per_method():
    channels = ["email", "retargeting"]
    series = {
        "TRUTH": {"email": 0.6, "retargeting": 0.0},
        "last_touch": {"email": 0.3, "retargeting": 0.7},
        "shapley": {"email": 0.55, "retargeting": 0.05},
    }
    fig = PV.figure(channels, series, zero_channel="retargeting")
    names = [t.name for t in fig.data]
    assert "TRUTH" in names
    assert len(fig.data) == len(series)
    assert all(t.type == "bar" for t in fig.data)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "journeys.json")),
                    reason="generated data absent; run python -m src.generate")
def test_build_writes_self_contained_html(tmp_path):
    out = tmp_path / "attr.html"
    PV.build(out_path=str(out))
    assert out.exists() and out.stat().st_size > 1000
    head = out.read_text(encoding="utf-8")[:2000].lower()
    assert "plotly" in head  # bundled inline, opens offline
