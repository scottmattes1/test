"""Spike-imputation experiment for the 30-day AR deposit forecast.

Question (from the task): does forecasting performance improve if we impute
"spike" deposit days to a typical (average) value instead of leaving the raw
capped spikes in the target?

Approach
--------
Two Ridge models, *identical* architecture, differing only in the target:

  Model A (raw)           target = daily deposits, capped at $1M (current
                          production convention).
  Model B (spike-imputed) target = same series, but each spike day (deposit
                          > $1M) replaced by the mean deposit of *non-spike*
                          days sharing its weekday.

Both are direct 30-day-ahead forecasts: features known as-of the origin day
t-30 (trailing deposit cadence) plus calendar/holiday features of the target
day t (known in advance). Evaluated on the last 143 days (matching the prior
ablation's holdout). RMSE is reported for each model against its own target and
against the raw actuals.

A third line — the provided ``vw_cash_receipts_forecast`` (Expected/p50) — is
overlaid as a forward projection. Its horizon (2026-07-23 onward) has no
realized actuals, so it cannot be scored; it is shown for context only.

Deps: numpy, pandas, scikit-learn, matplotlib.  Run:
    python ar_analytics/forecast_spike_experiment.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "out"

HORIZON = 30           # forecast this many days ahead
HOLDOUT = 143          # size of the test window (matches prior ablation)
CAP = 1_000_000.0      # existing model caps daily deposits at $1M
SPIKE_THRESHOLD = 1_000_000.0  # a "spike day" clears the cap

INVOICE_FILES = [
    "ar_invoices1_1.csv", "ar_invoices1_2.csv",
    "ar_invoices2_1.csv", "ar_invoices2_2.csv",
]


# --- data loading ------------------------------------------------------------


def load_daily_deposits() -> pd.Series:
    """Daily deposits = sum(TOTALPAID) by WHENPAID, over a gap-filled calendar."""
    frames = []
    for name in INVOICE_FILES:
        df = pd.read_csv(RAW / name, usecols=["WHENPAID", "TOTALPAID"],
                         dtype={"TOTALPAID": "float64"}, low_memory=False)
        frames.append(df)
    inv = pd.concat(frames, ignore_index=True)
    inv["WHENPAID"] = pd.to_datetime(inv["WHENPAID"], format="%m/%d/%Y", errors="coerce")
    inv = inv.dropna(subset=["WHENPAID"])
    inv = inv[inv["TOTALPAID"] != 0]
    daily = inv.groupby(inv["WHENPAID"].dt.normalize())["TOTALPAID"].sum()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full, fill_value=0.0).rename("deposits")


def load_holiday_features() -> pd.DataFrame:
    """Business-day / holiday features per calendar day (feature group H)."""
    cols = ["CalendarDay", "is_fed_holiday", "is_federal_observed_hol", "is_weekend",
            "is_fed_business_day", "is_month_end_business_day",
            "is_month_start_business_day", "bdays_to_month_end", "bdays_from_month_start"]
    df = pd.read_csv(RAW / "calendar_holidays.csv", usecols=cols)
    df["CalendarDay"] = pd.to_datetime(df["CalendarDay"])
    return df.set_index("CalendarDay").astype("float64")


def load_provided_forecast() -> pd.Series:
    """Expected (p50) daily forecast from vw_cash_receipts_forecast (headerless)."""
    day, amt, rtype, scen = [], [], [], []
    with (RAW / "vw_cash_receipts_forecast.csv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 24:
                continue
            day.append(row[0]); rtype.append(row[19]); scen.append(row[20]); amt.append(row[23])
    df = pd.DataFrame({"day": day, "rtype": rtype, "scen": scen, "amt": amt})
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df = df.dropna(subset=["day", "amt"])
    exp = df[(df["rtype"] == "Forecast") & (df["scen"] == "Expected")]
    return exp.groupby("day")["amt"].sum().rename("provided_expected")


# --- target construction -----------------------------------------------------


def impute_spikes(deposits: pd.Series) -> pd.Series:
    """Replace spike days (>threshold) with the non-spike mean for their weekday."""
    s = deposits.copy()
    is_spike = s > SPIKE_THRESHOLD
    wd = s.index.weekday
    weekday_mean = {d: s[(~is_spike) & (wd == d)].mean() for d in range(7)}
    out = s.copy()
    for i, (ts, spike) in enumerate(zip(s.index, is_spike)):
        if spike:
            out.iloc[i] = weekday_mean[ts.weekday()]
    return out


# --- feature engineering -----------------------------------------------------


def build_design(target: pd.Series, holidays: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Direct 30-day-ahead design matrix for a given (capped) target series.

    Cadence features are computed on the *origin* day (t-HORIZON) so they are
    strictly known at prediction time; holiday/calendar features describe the
    target day t (known in advance).
    """
    dep = target.copy()
    origin = dep.shift(HORIZON)  # value/rolling as-of the origin day

    feat = pd.DataFrame(index=dep.index)
    # Cadence (as-of origin) — base group.
    feat["lag_h"] = origin
    for w in (7, 14, 30, 60):
        feat[f"roll{w}_mean"] = dep.shift(HORIZON).rolling(w).mean()
        feat[f"roll{w}_sum"] = dep.shift(HORIZON).rolling(w).sum()
    feat["roll7_std"] = dep.shift(HORIZON).rolling(7).std()

    # Calendar of the target day — base group.
    idx = feat.index
    feat["dow"] = idx.weekday
    for d in range(6):  # one-hot Mon..Sat (Sun as baseline)
        feat[f"dow_{d}"] = (idx.weekday == d).astype(float)
    feat["dom"] = idx.day
    feat["month"] = idx.month
    feat = feat.drop(columns=["dow"])

    # Holiday group H — target day.
    h = holidays.reindex(idx)
    for c in h.columns:
        feat[c] = h[c].values

    data = feat.join(dep.rename("y")).dropna()
    return data.drop(columns=["y"]), data["y"]


# --- fit / evaluate ----------------------------------------------------------


def fit_eval(X: pd.DataFrame, y: pd.Series) -> dict:
    split = len(y) - HOLDOUT
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=10.0).fit(scaler.transform(Xtr), ytr)
    pred = pd.Series(model.predict(scaler.transform(Xte)), index=yte.index).clip(lower=0)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    return {"pred": pred, "y_test": yte, "rmse": rmse, "test_index": yte.index}


def rmse_vs(pred: pd.Series, truth: pd.Series) -> float:
    j = pred.index.intersection(truth.index)
    return float(np.sqrt(mean_squared_error(truth.loc[j], pred.loc[j])))


# --- main --------------------------------------------------------------------


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    deposits = load_daily_deposits()
    holidays = load_holiday_features()

    raw_capped = deposits.clip(upper=CAP)
    imputed = impute_spikes(deposits).clip(upper=CAP)
    actuals_capped = raw_capped  # ground truth line on the chart

    n_spikes = int((deposits > SPIKE_THRESHOLD).sum())

    XA, yA = build_design(raw_capped, holidays)
    XB, yB = build_design(imputed, holidays)
    A = fit_eval(XA, yA)
    B = fit_eval(XB, yB)

    # Fair comparison: each model's predictions vs the RAW actuals too.
    a_vs_actual = rmse_vs(A["pred"], actuals_capped)
    b_vs_actual = rmse_vs(B["pred"], actuals_capped)

    provided = load_provided_forecast()

    # --- metrics table ---
    metrics = pd.DataFrame([
        {"model": "A_raw_capped", "rmse_vs_own_target": A["rmse"],
         "rmse_vs_raw_actuals": a_vs_actual},
        {"model": "B_spike_imputed", "rmse_vs_own_target": B["rmse"],
         "rmse_vs_raw_actuals": b_vs_actual},
    ])
    metrics.to_csv(OUT / "spike_experiment_metrics.csv", index=False)

    def d(v: float) -> str:  # literal $ (escaped so matplotlib doesn't parse mathtext)
        return f"\\${v:,.0f}"

    verdict_target = "lowers" if B["rmse"] < A["rmse"] else "does NOT lower"
    verdict_real = "improves" if b_vs_actual < a_vs_actual else "does NOT improve"

    # --- chart: two panels ---
    test_idx = A["test_index"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 2]})

    # Panel 1 — the scoreable comparison over the 143-day holdout.
    ax1.plot(actuals_capped.loc[test_idx].index, actuals_capped.loc[test_idx].values / 1e6,
             color="#111827", lw=1.7, label="Actual deposits (capped \\$1M)", zorder=3)
    ax1.plot(A["pred"].index, A["pred"].values / 1e6, color="#2563eb", lw=1.9,
             label=f"Model A · raw target  —  RMSE vs actuals {d(a_vs_actual)}")
    ax1.plot(B["pred"].index, B["pred"].values / 1e6, color="#dc2626", lw=1.9, ls="--",
             label=f"Model B · spike-imputed  —  RMSE vs actuals {d(b_vs_actual)}")
    sp = deposits.loc[test_idx][deposits.loc[test_idx] > SPIKE_THRESHOLD]
    if len(sp):
        ax1.scatter(sp.index, np.full(len(sp), CAP) / 1e6, marker="^", color="#f59e0b",
                    s=70, zorder=4, edgecolor="#7c2d12",
                    label=f"Spike day in holdout (>\\${CAP/1e6:.0f}M), n={len(sp)}")
    ax1.set_title("AR deposit forecast · 30-day horizon · raw vs. spike-imputed target "
                  "(143-day holdout)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Daily deposits (\\$M)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.96)
    ax1.text(
        0.995, 0.955,
        "RMSE on each model's OWN target:\n"
        f"   A (raw):          {d(A['rmse'])}\n"
        f"   B (imputed):    {d(B['rmse'])}\n"
        f"Imputation {verdict_target} target-RMSE, but\n"
        f"{verdict_real} accuracy vs real actuals.",
        transform=ax1.transAxes, ha="right", va="top", fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round", fc="#fffbeb", ec="#f59e0b"))

    # Panel 2 — full history + the provided forecast as a forward projection.
    ax2.plot(actuals_capped.index, actuals_capped.values / 1e6, color="#111827", lw=1.0,
             label="Actual deposits (capped \\$1M)")
    if len(provided):
        ax2.plot(provided.index, provided.values / 1e6, color="#059669", lw=1.2, alpha=0.9,
                 label="Provided vw_cash_receipts_forecast (Expected/p50)")
    cut = actuals_capped.index.max()
    ax2.axvspan(cut, provided.index.max() if len(provided) else cut,
                color="#059669", alpha=0.06)
    ax2.axvline(cut, color="#9ca3af", ls=":", lw=1)
    ax2.text(cut, ax2.get_ylim()[1] * 0.92, "  actuals end / forecast begins\n"
             "  (no overlap → provided model not scoreable)",
             fontsize=8.5, va="top", color="#374151")
    ax2.set_title("Context: full deposit history and the provided forward forecast",
                  fontsize=12)
    ax2.set_ylabel("Daily deposits (\\$M)")
    ax2.set_xlabel("Target date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "spike_experiment_chart.png", dpi=140)

    print(f"Daily deposit series: {deposits.index.min().date()} .. "
          f"{deposits.index.max().date()} ({len(deposits)} days, {n_spikes} spikes)")
    print(f"Model A  target-RMSE ${A['rmse']:,.0f} | vs raw actuals ${a_vs_actual:,.0f}")
    print(f"Model B  target-RMSE ${B['rmse']:,.0f} | vs raw actuals ${b_vs_actual:,.0f}")
    print(f"Verdict: spike-imputation {verdict_target} target-RMSE but "
          f"{verdict_real} accuracy vs real actuals.")
    print(f"Wrote {OUT/'spike_experiment_chart.png'} and spike_experiment_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
