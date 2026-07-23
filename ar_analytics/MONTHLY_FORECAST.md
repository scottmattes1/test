# Monthly deposit forecast (made on the 1st) — findings

**Question:** On the 1st of each month, how accurately can we predict the month's
total AR deposits from this data?

**Answer: about ±15% in a typical month** (MAPE 15.4%), with **two-thirds of
months landing within 20%**. The accuracy comes almost entirely from the invoice
pipeline that is already known on the 1st — not from the short monthly history.

## What makes it work

On the 1st we already know every invoice on the books and its due date, so two
signals are available before the month starts:

| As-of-1st signal | Correlation with the month's deposits |
|---|---|
| **overdue_open** (open invoices past due, by face value) | **0.84** |
| **due_this_month** (open invoices due this month) | **0.78** |
| business days in the month | 0.19 |

Deposits ≈ a portion of (what's due this month) + (catch-up on the overdue
backlog). That relationship is stable enough to forecast from.

## Backtest (expanding window, last 12 months, forecast made as-of the 1st)

| Model | MAPE | RMSE | within 10% | within 20% |
|---|---|---|---|---|
| **pipeline_regress** (Ridge on due + overdue + business days) | **15.4%** | $3.09M | 42% | 67% |
| **pipeline_ratio** (collection-rate × due_this_month) | 15.6% | **$2.86M** | **58%** | 58% |
| trailing_3m (last-3-month average) | 21.0% | $4.12M | 33% | 67% |
| seasonal_naive (same month last year) | 25.8% | $5.48M | 25% | 50% |

The two pipeline models are effectively tied and both clearly beat the
history-only baselines. Seasonal-naive is weak because the book is **growing fast**
(monthly deposits roughly doubled, ~$7M → ~$13M, with a $23M surge in Jun-2026),
so "same month last year" is stale; there also isn't enough history for stable
seasonality.

## Honest limits

- **Only 22 usable months** (Aug/Sep 2024 are system ramp-up and were dropped).
  The 12-month backtest is a small sample, so treat ~15% as indicative, not
  precise — the metrics themselves carry meaningful uncertainty.
- **Outlier months dominate RMSE.** The Jun-2026 surge to $23.2M (which contains
  the $6.6M single-day settlement seen earlier) and the Jul-2026 drop to $11.3M
  are the two big misses; RMSE (~$2.9–3.1M) is inflated by them even though the
  median month is close.
- The model **lags sharp turns** — it anchors to the pipeline plus recent
  run-rate, so it under-shoots surges and over-shoots the month after.

## How to push accuracy further

1. **Age-weight the overdue backlog** — a 30-day-overdue invoice collects at a
   very different rate than a 180-day one; a single "overdue_open" total blurs
   that.
2. **Per-customer collection timing** — a handful of large customers (Vertiv,
   Masonite, Armstrong, Dow) drive the totals; their individual payment lag is
   learnable and would sharpen the pipeline estimate.
3. **More history** as it accrues — even 12 more months would materially firm up
   the seasonal and ratio estimates.

## Note on "how many"

This forecasts the **dollar total** of monthly deposits (the project's through-line).
The same pipeline predicts the **count** of receipts too (`n_receipts` is built
alongside `deposits`) — say the word and I'll switch the target.

## Reproduce

```bash
python ar_analytics/monthly_forecast.py
# -> data/out/monthly_forecast_chart.png, monthly_backtest.csv, monthly_metrics.csv
```
