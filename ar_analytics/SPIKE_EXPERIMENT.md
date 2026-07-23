# Spike-imputation evaluation — findings

**Question:** Does the 30-day AR deposit forecast perform better when spike days
are mean-imputed instead of left in the actuals? **No cap is applied anywhere.**

**Answer:** The RMSE drops by ~67%, but that improvement is a *measurement*
effect, not a better forecast. The forecast is identical in both graphs — only
the ground truth it is scored against changes.

## Setup

- **Actuals:** daily deposits = `SUM(TOTALPAID)` by `WHENPAID` from the batch-2
  invoice exports, gap-filled to a daily calendar. 2024-08-06 → 2026-07-23,
  **717 days. No $1M cap — removed entirely.**
- **Spike day:** deposits > mean + 2·std = **> $1,264,611** → **18 days** total
  (11 fall inside the holdout).
- **One model:** a 30-day-ahead Ridge (α=10, standardized), trained on the
  **mean-imputed** series so a few outliers don't distort the linear fit.
  Features: trailing deposit cadence as-of the origin day (t−30: lag,
  7/14/30/60-day rolling mean/sum, 7-day std) + calendar/holiday features of the
  target day (`calendar_holidays` group **H**). Last **143 days** held out.
- The **same holdout predictions** are scored against two ground truths.
- Nothing is plotted past the last actuals day (the provided
  `vw_cash_receipts_forecast` is entirely future, so it is dropped).

## Results

| Ground truth (same forecast) | RMSE |
|---|---|
| **Graph 1 — full unaltered actuals** | **$728,673** |
| **Graph 2 — mean-imputed actuals** | **$240,726** |

Mean-imputing the spike days cuts RMSE from **$728,673 → $240,726** (−67%).
Because the two graphs share one identical forecast, that entire gap comes from
the ~11 spike days: they contribute √(728,673² − 240,726²) ≈ **$688k** of RMSE
on their own. The model tracks the ~$0.5M weekday cadence well (Graph 2) but
never approaches the spikes, which reach **$6.5M** (Graph 1).

## Interpretation

"Improvement" here means we stopped grading the model on the days it cannot
predict — not that the forecast got better. Spikes are a **different process** (a
few large invoices settling on one day, e.g. 2026-06-07: $6.6M across 3,203
invoices), invisible from 30-day-old cadence. Mean-imputation is a reasonable
way to measure normal-day accuracy in isolation, but for cash planning the spike
days are exactly the ones that matter, so they should be modeled separately
(spike-timing classification / invoice-level settlement) rather than smoothed
away.

## Forecast horizon sweep (7 / 14 / 30 days)

The prior thread proposed shortening the horizon to make spike timing more
knowable. It does not help — RMSE is flat across horizons (all differences are
well inside the ~$49k holdout SE):

| Horizon | RMSE vs unaltered actuals | RMSE vs mean-imputed actuals |
|---|---|---|
| 7 days | $734,612 | $247,170 |
| 14 days | $737,364 | $250,562 |
| 30 days | **$728,673** | **$240,726** |

Shortening the horizon buys nothing here: normal-day deposits are weekly-seasonal
and about equally predictable 7 or 30 days out, and the spike days — which own
the unaltered-actuals error — are driven by large invoices settling, timing a
daily-aggregate cadence model can't see at *any* of these horizons. The lever is
not horizon length; it is modeling spikes at the invoice level (due/discount
dates, per-customer settlement behavior), where knowing which large invoices are
due in the next 7 days is genuinely informative.

## Note on scale

Uncapped RMSE vs. unaltered actuals (**$728,673**) is close to the prior thread's
reported ~$822k, and far above the earlier capped reconstruction (~$263k) —
confirming RMSE on this series is dominated by how spike days are handled.

## Reproduce

```bash
python ar_analytics/forecast_spike_experiment.py
# -> data/out/graph1_vs_unaltered_actuals.png
#    data/out/graph2_vs_mean_imputed_actuals.png
#    data/out/spike_experiment_metrics.csv
```
