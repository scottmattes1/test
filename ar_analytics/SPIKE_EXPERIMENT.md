# Spike-imputation experiment — findings

**Question:** Does the 30-day AR deposit forecast improve if we impute "spike"
days to a typical (average) value instead of leaving the raw capped spikes in
the target?

**Answer: No.** Imputation makes the target *look* more predictable but does not
improve real-world accuracy — it slightly worsens it.

## Setup

- **Actuals:** daily deposits = `SUM(TOTALPAID)` by `WHENPAID` from the batch-2
  invoice exports, gap-filled to a daily calendar. 2024-08-06 → 2026-07-23,
  **717 days**, capped at **$1M/day** (the existing model's convention).
- **Spike day:** a day whose deposits exceed the $1M cap — **33 days** total
  (4.6% of days, 21% of all deposit dollars).
- **Two models, identical architecture** (Ridge, α=10, standardized), differing
  only in the target:
  - **A — raw:** target = capped daily deposits.
  - **B — spike-imputed:** each spike day replaced by the mean deposit of
    *non-spike* days sharing its weekday.
- **Features:** trailing deposit cadence as-of the origin day (t−30: lag,
  7/14/30/60-day rolling mean/sum, 7-day std) + calendar/holiday features of the
  target day (day-of-week, day-of-month, month, and the `calendar_holidays`
  business-day/holiday flags — group **H**).
- **Direct 30-day-ahead** forecast; last **143 days** held out.

## Results

| Model | RMSE vs its **own** target | RMSE vs **real** actuals |
|---|---|---|
| A — raw | **$263,129** | **$263,129** |
| B — spike-imputed | $191,553 | **$272,694** |

**Reading it:** imputation drops RMSE from $263k → $192k *against the de-spiked
target* — but that lower number is measured against an easier, edited ground
truth. Judged against what actually happened, model B is **worse** ($263k →
$273k), because it is trained to under-predict exactly the spike days that drive
real error. Spike-imputation relabels the hard days as easy ones rather than
forecasting them better.

## Why the model misses spikes

Both models track the weekday deposit cadence well (~$0.5M working-day rhythm,
near-zero weekends) but flatten out around $0.55–0.65M and never reach the $1M
spike level. Spikes are a **different process** — a handful of large invoices
settling on one day (e.g. 2026-06-07: $6.6M across 3,203 invoices) — not a
scaled-up normal day. A daily-total regression cannot see them coming from
30-day-old cadence. This matches the prior thread's own next-step suggestion:
shorten the horizon and/or model spike *timing* separately (classification /
invoice-level), rather than adding more static daily features.

## Caveats

- **Reconstruction, not the original pipeline.** The prior thread's model
  reported RMSE ≈ $822k; this reconstruction sits far lower (~$263k), mostly
  because the target is capped at $1M — a single uncapped $6.6M miss over a
  143-day window adds ~$550k to RMSE by itself. **RMSE here is dominated by the
  capping choice**, so treat the absolute figures as internal to this
  experiment; the valid takeaway is the **A-vs-B contrast**, which is
  apples-to-apples. Drop in the original `ar_deposits_holiday_features.py` and
  the numbers can be matched exactly.
- **Provided `vw_cash_receipts_forecast` is not scored.** Its Expected/p50
  scenario runs 2026-07-23 → 2027-10, beginning exactly where actuals end —
  **zero overlap**, so no RMSE against actuals is possible. It is drawn in the
  lower panel as a forward projection only. A backtested version (predictions on
  dates that already have actuals) would let it join the comparison.

## Reproduce

```bash
python ar_analytics/forecast_spike_experiment.py
# -> data/out/spike_experiment_chart.png, spike_experiment_metrics.csv
```
