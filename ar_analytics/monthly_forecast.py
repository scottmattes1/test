"""Monthly AR deposit forecast, made on the 1st of each month.

Target: total cash deposits (sum of TOTALPAID by WHENPAID) in the month.
Constraint: only ~22 usable months of history (Aug/Sep 2024 are system ramp-up),
so the model must stay simple. The leverage is not the short monthly history but
the invoice pipeline that is already known on the 1st: invoices on the books with
a due date this month, plus the overdue-but-open backlog.

Models compared (expanding-window backtest over the last 12 months, each forecast
made as-of the 1st with only prior-month information):

  seasonal_naive   deposits(M) = deposits(M-12)
  trailing_3m      mean of the last 3 months
  pipeline_ratio   r * due_this_month              (r = collection rate, learned)
  pipeline_regress Ridge on [due_this_month, overdue_open, business_days]

Deps: numpy, pandas, scikit-learn, matplotlib.  Run:
    python ar_analytics/monthly_forecast.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "out"

INVOICE_FILES = ["ar_invoices1_1.csv", "ar_invoices1_2.csv",
                 "ar_invoices2_1.csv", "ar_invoices2_2.csv"]
FIRST_MONTH = "2024-10-01"   # drop Aug/Sep 2024 ramp-up
TEST_MONTHS = 12             # size of the backtest window


def d(v: float) -> str:  # literal $ (escaped so matplotlib doesn't parse mathtext)
    return f"\\${v:,.0f}"


def load_invoices() -> pd.DataFrame:
    cols = ["WHENPOSTED", "WHENDUE", "WHENPAID", "TOTALENTERED", "TOTALPAID"]
    inv = pd.concat([pd.read_csv(RAW / f, usecols=cols, low_memory=False)
                     for f in INVOICE_FILES], ignore_index=True)
    for c in ["WHENPOSTED", "WHENDUE", "WHENPAID"]:
        inv[c] = pd.to_datetime(inv[c], format="%m/%d/%Y", errors="coerce")
    for c in ["TOTALENTERED", "TOTALPAID"]:
        inv[c] = pd.to_numeric(inv[c], errors="coerce").fillna(0.0)
    return inv


def business_days_by_month() -> pd.Series:
    df = pd.read_csv(RAW / "calendar_holidays.csv",
                     usecols=["CalendarDay", "is_fed_business_day"])
    df["CalendarDay"] = pd.to_datetime(df["CalendarDay"])
    df["m"] = df["CalendarDay"].values.astype("datetime64[M]")
    return df.groupby("m")["is_fed_business_day"].sum().rename("bdays")


def build_monthly(inv: pd.DataFrame) -> pd.DataFrame:
    paid = inv[(inv.WHENPAID.notna()) & (inv.TOTALPAID != 0)].copy()
    paid["m"] = paid.WHENPAID.values.astype("datetime64[M]")
    monthly = paid.groupby("m")["TOTALPAID"].sum().rename("deposits").to_frame()

    rows = []
    for p in pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"):
        start = p.to_timestamp()
        nxt = (p + 1).to_timestamp()
        booked = inv[inv.WHENPOSTED < start]                       # known on the 1st
        openi = booked[booked.WHENPAID.isna() | (booked.WHENPAID >= start)]
        due_this = openi.loc[(openi.WHENDUE >= start) & (openi.WHENDUE < nxt),
                             "TOTALENTERED"].sum()
        overdue = openi.loc[openi.WHENDUE < start, "TOTALENTERED"].sum()
        rows.append({"m": start, "due_this_month": due_this, "overdue_open": overdue})
    pipe = pd.DataFrame(rows).set_index("m")

    out = monthly.join(pipe).join(business_days_by_month())
    return out.loc[FIRST_MONTH:]


# --- models (each: train_df -> prediction for the target month) --------------


def m_seasonal_naive(train: pd.DataFrame, row: pd.Series, hist: pd.DataFrame) -> float:
    m12 = row.name - pd.DateOffset(months=12)
    return float(hist.loc[m12, "deposits"]) if m12 in hist.index else float(
        train["deposits"].iloc[-1])


def m_trailing_3m(train: pd.DataFrame, row: pd.Series, hist: pd.DataFrame) -> float:
    return float(train["deposits"].tail(3).mean())


def m_pipeline_ratio(train: pd.DataFrame, row: pd.Series, hist: pd.DataFrame) -> float:
    r = (train["deposits"] / train["due_this_month"]).replace([np.inf, -np.inf],
                                                               np.nan).dropna().mean()
    return float(r * row["due_this_month"])


def m_pipeline_regress(train: pd.DataFrame, row: pd.Series, hist: pd.DataFrame) -> float:
    feats = ["due_this_month", "overdue_open", "bdays"]
    Xtr, ytr = train[feats].values, train["deposits"].values
    sc = StandardScaler().fit(Xtr)
    model = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    return float(model.predict(sc.transform(row[feats].values.reshape(1, -1)))[0])


MODELS = {
    "seasonal_naive": m_seasonal_naive,
    "trailing_3m": m_trailing_3m,
    "pipeline_ratio": m_pipeline_ratio,
    "pipeline_regress": m_pipeline_regress,
}


def backtest(monthly: pd.DataFrame) -> pd.DataFrame:
    test_idx = monthly.index[-TEST_MONTHS:]
    recs = []
    for m in test_idx:
        train = monthly.loc[monthly.index < m]
        row = monthly.loc[m]
        rec = {"m": m, "actual": row["deposits"]}
        for name, fn in MODELS.items():
            rec[name] = fn(train, row, monthly)
        recs.append(rec)
    return pd.DataFrame(recs).set_index("m")


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    a = bt["actual"].values
    out = []
    for name in MODELS:
        p = bt[name].values
        ape = np.abs(a - p) / a
        out.append({
            "model": name,
            "MAPE_%": np.mean(ape) * 100,
            "RMSE": np.sqrt(np.mean((a - p) ** 2)),
            "within_10%": np.mean(ape <= 0.10) * 100,
            "within_20%": np.mean(ape <= 0.20) * 100,
        })
    return pd.DataFrame(out).sort_values("MAPE_%").reset_index(drop=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = load_invoices()
    monthly = build_monthly(inv)
    corr = monthly[["due_this_month", "overdue_open", "bdays"]].corrwith(
        monthly["deposits"])
    bt = backtest(monthly)
    mt = metrics(bt)
    best = mt.iloc[0]["model"]

    bt.to_csv(OUT / "monthly_backtest.csv")
    mt.to_csv(OUT / "monthly_metrics.csv", index=False)

    # --- chart: actuals + regression model only ---
    reg = mt.loc[mt["model"] == "pipeline_regress"].iloc[0]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(monthly.index, monthly["deposits"] / 1e6, color="#111827", lw=2,
            marker="o", ms=5, label="Actual monthly deposits")
    ax.plot(bt.index, bt["pipeline_regress"] / 1e6, color="#dc2626", lw=2.6,
            marker="s", ms=5, label=f"Pipeline regression forecast  (MAPE "
            f"{reg['MAPE_%']:.1f}%, {reg['within_20%']:.0f}% within 20%)")
    ax.axvspan(bt.index.min(), bt.index.max(), color="#f59e0b", alpha=0.06)
    ax.axvline(bt.index.min(), color="#9ca3af", ls=":", lw=1)
    ax.text(bt.index.min(), ax.get_ylim()[1] * 0.97, "  12-month backtest →",
            fontsize=9, va="top", color="#374151")
    ax.set_title("Monthly AR deposit forecast (made on the 1st)  ·  "
                 "expanding-window backtest", fontsize=13, fontweight="bold")
    ax.set_ylabel("Monthly deposits (\\$M)")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.96)
    ax.text(0.995, 0.03, f"MAPE {reg['MAPE_%']:.1f}%   ·   RMSE {d(reg['RMSE'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            fontweight="bold", family="monospace",
            bbox=dict(boxstyle="round", fc="#fffbeb", ec="#f59e0b"))
    fig.tight_layout()
    fig.savefig(OUT / "monthly_forecast_chart.png", dpi=140)
    plt.close(fig)

    pd.options.display.float_format = lambda x: f"{x:,.2f}"
    print(f"Usable months: {len(monthly)} ({monthly.index.min().date()} .. "
          f"{monthly.index.max().date()})")
    print("\nCorrelation of as-of-1st signals with monthly deposits:")
    print(corr.round(3).to_string())
    print(f"\nBacktest metrics (last {TEST_MONTHS} months):")
    print(mt.to_string(index=False))
    print(f"\nBest model: {best}")
    print("Wrote monthly_forecast_chart.png, monthly_backtest.csv, monthly_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
