"""Spike-imputation evaluation for the 30-day AR deposit forecast.

No capping anywhere. One forecasting model, evaluated against two ground truths:

  Graph 1 — model performance vs. FULL UNALTERED ACTUALS (spikes included).
  Graph 2 — model performance vs. MEAN-IMPUTED ACTUALS (spike days replaced by
            the average deposit for that weekday).

The model is a single 30-day-ahead Ridge forecaster trained on the mean-imputed
series (so a handful of outlier days don't distort the linear fit). The *same*
holdout predictions are then scored both ways — only the ground truth (and thus
the RMSE) differs between the two graphs. Nothing is drawn past the last day of
actuals.

Deps: numpy, pandas, scikit-learn, matplotlib.  Run:
    python ar_analytics/forecast_spike_experiment.py
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "out"

HORIZON = 30       # forecast this many days ahead
HOLDOUT = 143      # size of the test window (matches the prior ablation)
SPIKE_K = 2.0      # a "spike day" exceeds mean + SPIKE_K * std of the series

INVOICE_FILES = [
    "ar_invoices1_1.csv", "ar_invoices1_2.csv",
    "ar_invoices2_1.csv", "ar_invoices2_2.csv",
]


# --- data --------------------------------------------------------------------


def load_daily_deposits() -> pd.Series:
    """Daily deposits = sum(TOTALPAID) by WHENPAID, gap-filled. No cap."""
    frames = [
        pd.read_csv(RAW / n, usecols=["WHENPAID", "TOTALPAID"],
                    dtype={"TOTALPAID": "float64"}, low_memory=False)
        for n in INVOICE_FILES
    ]
    inv = pd.concat(frames, ignore_index=True)
    inv["WHENPAID"] = pd.to_datetime(inv["WHENPAID"], format="%m/%d/%Y", errors="coerce")
    inv = inv.dropna(subset=["WHENPAID"])
    inv = inv[inv["TOTALPAID"] != 0]
    daily = inv.groupby(inv["WHENPAID"].dt.normalize())["TOTALPAID"].sum()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full, fill_value=0.0).rename("deposits")


def load_holiday_features() -> pd.DataFrame:
    cols = ["CalendarDay", "is_fed_holiday", "is_federal_observed_hol", "is_weekend",
            "is_fed_business_day", "is_month_end_business_day",
            "is_month_start_business_day", "bdays_to_month_end", "bdays_from_month_start"]
    df = pd.read_csv(RAW / "calendar_holidays.csv", usecols=cols)
    df["CalendarDay"] = pd.to_datetime(df["CalendarDay"])
    return df.set_index("CalendarDay").astype("float64")


def spike_mask(deposits: pd.Series) -> tuple[pd.Series, float]:
    thr = deposits.mean() + SPIKE_K * deposits.std()
    return deposits > thr, float(thr)


def mean_impute(deposits: pd.Series, is_spike: pd.Series) -> pd.Series:
    """Replace each spike day with the mean of non-spike days on its weekday."""
    wd = deposits.index.weekday
    wmean = {d: deposits[(~is_spike) & (wd == d)].mean() for d in range(7)}
    out = deposits.copy()
    for i, ts in enumerate(deposits.index):
        if is_spike.iloc[i]:
            out.iloc[i] = wmean[ts.weekday()]
    return out


# --- model -------------------------------------------------------------------


def build_design(target: pd.Series, holidays: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Direct 30-day-ahead design matrix (cadence as-of origin t-HORIZON)."""
    dep = target
    feat = pd.DataFrame(index=dep.index)
    feat["lag_h"] = dep.shift(HORIZON)
    for w in (7, 14, 30, 60):
        feat[f"roll{w}_mean"] = dep.shift(HORIZON).rolling(w).mean()
        feat[f"roll{w}_sum"] = dep.shift(HORIZON).rolling(w).sum()
    feat["roll7_std"] = dep.shift(HORIZON).rolling(7).std()

    idx = feat.index
    for d in range(6):  # day-of-week one-hot, Sun as baseline
        feat[f"dow_{d}"] = (idx.weekday == d).astype(float)
    feat["dom"] = idx.day
    feat["month"] = idx.month

    h = holidays.reindex(idx)
    for c in h.columns:
        feat[c] = h[c].values

    data = feat.join(dep.rename("y")).dropna()
    return data.drop(columns=["y"]), data["y"]


def fit_predict(train_target: pd.Series, holidays: pd.DataFrame) -> pd.Series:
    """Fit on all but the last HOLDOUT days; return holdout predictions."""
    X, y = build_design(train_target, holidays)
    split = len(y) - HOLDOUT
    scaler = StandardScaler().fit(X.iloc[:split])
    model = Ridge(alpha=10.0).fit(scaler.transform(X.iloc[:split]), y.iloc[:split])
    pred = model.predict(scaler.transform(X.iloc[split:]))
    return pd.Series(pred, index=y.index[split:]).clip(lower=0)


def rmse(pred: pd.Series, truth: pd.Series) -> float:
    j = pred.index.intersection(truth.index)
    return float(np.sqrt(mean_squared_error(truth.loc[j], pred.loc[j])))


# --- plotting ----------------------------------------------------------------


def d(v: float) -> str:  # literal $ (escaped so matplotlib doesn't parse mathtext)
    return f"\\${v:,.0f}"


def make_graph(path: Path, title: str, actuals: pd.Series, pred: pd.Series,
               spikes: pd.Series, err: float, actual_label: str, color: str) -> None:
    idx = pred.index
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.plot(actuals.loc[idx].index, actuals.loc[idx].values / 1e6, color="#111827",
            lw=1.7, label=actual_label, zorder=3)
    ax.plot(pred.index, pred.values / 1e6, color=color, lw=1.9,
            label=f"30-day Ridge forecast  —  RMSE {d(err)}")
    sp = spikes.loc[idx][spikes.loc[idx]]
    if len(sp):
        ax.scatter(sp.index, actuals.loc[sp.index].values / 1e6, marker="^",
                   color="#f59e0b", edgecolor="#7c2d12", s=75, zorder=4,
                   label=f"Spike day (>mean+{SPIKE_K:g}sd), n={len(sp)}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Daily deposits (\\$M)")
    ax.set_xlabel("Target date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.96)
    ax.text(0.995, 0.955, f"RMSE = {d(err)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold", family="monospace",
            bbox=dict(boxstyle="round", fc="#fffbeb", ec="#f59e0b"))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- main --------------------------------------------------------------------


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    deposits = load_daily_deposits()            # full, unaltered, uncapped
    holidays = load_holiday_features()
    is_spike, thr = spike_mask(deposits)
    imputed = mean_impute(deposits, is_spike)   # spike days -> weekday mean

    # One model, trained on the mean-imputed series; same predictions scored twice.
    pred = fit_predict(imputed, holidays)
    rmse_unaltered = rmse(pred, deposits)
    rmse_imputed = rmse(pred, imputed)

    n_hold_spikes = int(is_spike.loc[pred.index].sum())
    make_graph(
        OUT / "graph1_vs_unaltered_actuals.png",
        "AR deposit forecast (30-day) vs. FULL UNALTERED ACTUALS  ·  no cap  ·  "
        "143-day holdout",
        deposits, pred, is_spike, rmse_unaltered,
        "Actual deposits (unaltered, uncapped)", "#2563eb")
    make_graph(
        OUT / "graph2_vs_mean_imputed_actuals.png",
        "AR deposit forecast (30-day) vs. MEAN-IMPUTED ACTUALS  ·  no cap  ·  "
        "143-day holdout",
        imputed, pred, is_spike, rmse_imputed,
        "Actual deposits (spike days mean-imputed)", "#dc2626")

    pd.DataFrame([
        {"ground_truth": "full_unaltered_actuals", "rmse": rmse_unaltered},
        {"ground_truth": "mean_imputed_actuals", "rmse": rmse_imputed},
    ]).to_csv(OUT / "spike_experiment_metrics.csv", index=False)

    print(f"Daily deposits: {deposits.index.min().date()} .. {deposits.index.max().date()} "
          f"({len(deposits)} days), no cap")
    print(f"Spike threshold (mean+{SPIKE_K:g}sd): ${thr:,.0f} -> "
          f"{int(is_spike.sum())} spike days total, {n_hold_spikes} in holdout")
    print(f"RMSE vs FULL UNALTERED actuals: ${rmse_unaltered:,.0f}")
    print(f"RMSE vs MEAN-IMPUTED actuals:   ${rmse_imputed:,.0f}")
    print(f"Wrote graph1_vs_unaltered_actuals.png, graph2_vs_mean_imputed_actuals.png, "
          f"spike_experiment_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
