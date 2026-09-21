"""Tests for the real-data path: the cleaning rules and the BG/NBD a < 1 case."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import clv as CLV                       # noqa: E402
from src.clean_retail import clean               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _line(inv, code, qty, when, cust=1.0, price=2.0):
    return dict(Invoice=inv, StockCode=code, Description="x", Quantity=qty,
                InvoiceDate=pd.Timestamp(when), Price=price, **{"Customer ID": cust},
                Country="United Kingdom")


def test_cleaning_rules_on_a_small_frame():
    df = pd.DataFrame([
        _line("1", "10001", 10, "2010-01-05 10:00"),
        _line("1", "10001", 10, "2010-01-05 10:00"),              # exact duplicate
        _line("2", "10002", 5, "2010-01-05 15:00"),               # same day -> same occasion
        _line("3", "POST", 1, "2010-01-06 09:00"),                # not a product
        _line("4", "10003", 4, "2010-01-07 09:00", cust=np.nan),  # no customer
        _line("5", "10004", 3, "2010-01-08 09:00", price=0.0),    # free line
        _line("C6", "10001", -4, "2010-01-09 09:00"),             # partial return
        _line("7", "10005", 2, "2010-02-01 09:00"),
        _line("C8", "10005", -2, "2010-02-02 09:00"),             # full return
    ])
    occ, rep = clean(df)
    assert rep["dropped_exact_duplicates"] == 1
    assert rep["dropped_non_product_codes"] == 1
    assert rep["dropped_missing_customer"] == 1
    assert rep["dropped_price_le_zero"] == 1
    assert rep["dropped_fully_returned_occasions"] == 1
    assert len(occ) == 1                                  # one surviving purchase day
    row = occ.iloc[0]
    assert row.order_value == pytest.approx((10 - 4) * 2.0 + 5 * 2.0)
    assert row.n_products == 2
    assert bool(row.had_return)


def test_bgnbd_prediction_is_right_when_a_is_below_one():
    """Real Online Retail II fits a ~ 0.15. The first version replaced (a - 1)
    with 1e-6 whenever a <= 1 and returned huge negative predictions. Reference
    values are the closed form evaluated at 50 digits with mpmath."""
    m = CLV.BGNBD()
    m.r, m.alpha, m.a, m.b = 0.6890, 67.4035, 0.1468, 3.1536
    cases = [((0, 0, 546), 0.21329695855986683),
             ((5, 300, 400), 2.1156183339504717),
             ((150, 545, 546), 46.146872066192564)]
    for (x, tx, T), ref in cases:
        got = m.expected_purchases(192, np.array([x]), np.array([tx]), np.array([T]))[0]
        assert got == pytest.approx(ref, rel=1e-9)


def test_built_cohort_is_real_and_inside_the_window():
    path = os.path.join(HERE, "data", "transactions.npy")
    if not os.path.exists(path):
        pytest.skip("run `python src/build_data.py` first")
    txn = np.load(path)
    first = pd.Series(txn[:, 1]).groupby(txn[:, 0]).min()
    assert first.max() <= 546                 # every customer seen in calibration
    assert txn[:, 1].max() <= 738
    assert (txn[:, 2] > 0).all()
    assert set(np.unique(txn[:, 4])) <= {0.0, 1.0}
