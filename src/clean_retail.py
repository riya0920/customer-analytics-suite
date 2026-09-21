"""Clean UCI Online Retail II into one row per customer per purchase day.

Source: UCI Machine Learning Repository, "Online Retail II" (Chen, 2019), also
mirrored on Kaggle. A UK online gift retailer, 1 Dec 2009 - 9 Dec 2011, about
1.07M invoice lines. Many customers are small wholesalers.

Every rule below is counted, and the counts are written to
data/cleaning_report.json so the README can say exactly what was removed.

Rules, in order:
  1. drop exact duplicate lines
  2. drop lines with no Customer ID (cannot be tied to a customer)
  3. drop 'A' invoices (bad-debt adjustments, not sales)
  4. drop non-product stock codes (postage, manual, fees, samples, vouchers)
  5. drop lines with price <= 0
  6. net cancellations ('C' invoices) against the same customer's earlier
     purchases of the same product, newest first. A purchase that was later
     cancelled in full disappears; a partial return reduces the quantity.
     The order it came from is flagged `had_return`.
  7. aggregate to customer x calendar day. Several invoices on one day are one
     purchase occasion (the standard grain for BG/NBD), otherwise a customer who
     splits a basket across two invoices looks twice as loyal.
  8. drop occasions whose net value is <= 0 after returns.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_XLSX = os.path.join(HERE, "data", "raw", "online_retail_II.xlsx")
RAW_PKL = os.path.join(HERE, "data", "raw", "online_retail_ii.pkl")
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

START = pd.Timestamp("2009-12-01")
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"


def load_raw() -> pd.DataFrame:
    """Both workbook sheets as one frame. Parsing the xlsx takes ~4 minutes, so
    the result is cached next to it."""
    if os.path.exists(RAW_PKL):
        return pd.read_pickle(RAW_PKL)
    if not os.path.exists(RAW_XLSX):
        raise FileNotFoundError(
            f"{RAW_XLSX} not found. Download {DATA_URL} and unzip it into data/raw/")
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_pickle(RAW_PKL)
    return df


def clean(df: pd.DataFrame):
    rep = {"raw_lines": int(len(df))}

    def step(name, keep):
        nonlocal df
        rep[name] = int((~keep).sum())
        df = df[keep]

    step("dropped_exact_duplicates", ~df.duplicated())
    step("dropped_missing_customer", df["Customer ID"].notna())
    step("dropped_adjustment_invoices", ~df["Invoice"].str.startswith("A"))
    step("dropped_non_product_codes", df["StockCode"].str.match(PRODUCT_CODE))
    step("dropped_price_le_zero", df["Price"] > 0)

    df = df.assign(customer=df["Customer ID"].astype(int),
                   day=(df["InvoiceDate"].dt.normalize() - START).dt.days)
    is_cancel = df["Invoice"].str.startswith("C")
    buys = df[~is_cancel & (df["Quantity"] > 0)].copy()
    cancels = df[is_cancel & (df["Quantity"] < 0)]
    rep["dropped_other_nonpositive_qty"] = int(len(df) - len(buys) - len(cancels))
    rep["cancellation_lines"] = int(len(cancels))

    # 6. net cancellations, newest matching purchase first
    buys = buys.sort_values("InvoiceDate", kind="stable")
    qty = buys["Quantity"].to_numpy().astype(float)
    returned = np.zeros(len(buys), dtype=bool)
    index = defaultdict(list)
    for pos, key in enumerate(zip(buys["customer"], buys["StockCode"])):
        index[key].append(pos)
    times = buys["InvoiceDate"].to_numpy()
    unmatched = 0
    for c, s, q, t in zip(cancels["customer"], cancels["StockCode"],
                          -cancels["Quantity"].to_numpy(),
                          cancels["InvoiceDate"].to_numpy()):
        left = float(q)
        for pos in reversed(index.get((c, s), [])):
            if left <= 0:
                break
            if times[pos] > t or qty[pos] <= 0:
                continue
            take = min(qty[pos], left)
            qty[pos] -= take
            left -= take
            returned[pos] = True
        unmatched += left > 0
    buys["net_qty"] = qty
    buys["returned"] = returned
    rep["cancellations_without_a_matching_purchase"] = int(unmatched)
    rep["purchase_lines_fully_cancelled"] = int((qty == 0).sum())

    # 7. customer x day occasions
    buys["value"] = buys["net_qty"] * buys["Price"]
    live = buys[buys["net_qty"] > 0]
    occ = buys.groupby(["customer", "day"]).agg(
        order_value=("value", "sum"),
        had_return=("returned", "any"),
        country=("Country", "first")).reset_index()
    prods = live.groupby(["customer", "day"])["StockCode"].nunique()
    occ = occ.join(prods.rename("n_products"), on=["customer", "day"])
    occ["n_products"] = occ["n_products"].fillna(0).astype(int)
    rep["invoices_kept"] = int(buys["Invoice"].nunique())

    # 8. occasions wiped out by returns
    keep = occ["order_value"] > 0
    rep["dropped_fully_returned_occasions"] = int((~keep).sum())
    occ = occ[keep].reset_index(drop=True)

    rep["clean_lines"] = int(len(live))
    rep["purchase_occasions"] = int(len(occ))
    rep["customers"] = int(occ["customer"].nunique())
    rep["first_day"] = str(START.date())
    rep["last_day"] = str((START + pd.Timedelta(days=int(occ["day"].max()))).date())
    return occ, rep


if __name__ == "__main__":
    occ, rep = clean(load_raw())
    print(json.dumps(rep, indent=2))
    print(occ.describe())
