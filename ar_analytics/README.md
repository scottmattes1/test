# AR Analytics

Consolidation and validation for the accounts-receivable (AR) dataset — a
logistics/freight company's invoice ledger exported from **McLeod** (TMS) and
**Sage Intacct** (customer master), joined to a reusable date dimension.

> **Note:** this folder is self-contained and unrelated to the rest of this
> repository. It was added for the AR CSV project and can be lifted into its
> own repo at any time.

## The data model (star schema)

```
                 ┌─────────────────────────┐
                 │  calendar_holidays.csv  │   date dimension
                 │  (grain: one day)       │   2022-01-01 .. 2030-12-31
                 └────────────┬────────────┘
                              │ CalendarDay = due_date
                              │
   ┌─────────────┐     ┌──────┴───────────────────┐
   │customers.csv│     │  ar_aging1 + ar_aging2   │   AR fact
   │ (customer   ├─────┤  (grain: one invoice)    │   (two halves of one table)
   │  master)    │     └──────────────────────────┘
   └─────────────┘  CUSTOMERID = CUSTOMERID
```

| Table | Grain | Source | Key |
|---|---|---|---|
| `ar_invoices` (union of `ar_aging1` + `ar_aging2`) | one invoice line | McLeod | `RECORDNO` |
| `customers` | one customer | Sage Intacct | `CUSTOMERID` |
| `calendar_holidays` | one calendar day | generated | `CalendarDay` |

### Key facts established from the first batch

- **`ar_aging1.csv` and `ar_aging2.csv` are two halves of one table**, not two
  snapshots: same schema, same period, and disjoint records. They are
  **unioned** (stacked) into `ar_invoices` — **110,674** invoice lines,
  invoice dates **Jul 2024 → Jul 2026**.
- **`calendar.csv` is dropped** — it is a strict column-subset of
  `calendar_holidays.csv` (same 3,287 days, minus 10 holiday/business-day
  columns). The enriched file is the single date dimension.
- Every AR `CUSTOMERID` resolves to the customer master (**0 orphans**), and
  every AR `due_date` exists in the calendar (**0 gaps**).
- Open AR reconciles to the customer master: **$34.66M** (`total_due`) vs.
  **$34.65M** (`TOTALDUE`) — **0.03%** drift, confirming the two AR files
  together are the complete ledger.
- Total invoiced over the two years (`total_entered`): **~$287.9M**.

## Usage

Dependency-free — standard-library Python 3.11+, no `pip install` needed.

```bash
python ar_analytics/build.py
```

Outputs land in `ar_analytics/data/out/`:

- `ar_invoices.csv` — the two AR files unioned into one consolidated fact table.
- `validation_report.md` / `.json` — join checks, reconciliation, and totals.

Override the locations with `--raw` / `--out`.

## Adding the next batch

Files are classified **by their columns, not their filenames**, so hash-named
uploads and future exports need no renaming:

1. Drop the new CSV(s) into `ar_analytics/data/raw/`.
2. Re-run `python ar_analytics/build.py`.
3. Read `data/out/validation_report.md`.

`build.py` recognizes four layouts: AR fact (`invoice_state` + `aging_bucket`),
enriched calendar (`is_fed_holiday`), plain calendar (`CalendarDay`), and the
customer master (`CUSTOMERID` + `CREDITLIMIT` + `TOTALDUE`). Anything else is
reported under "dropped" rather than silently ignored.

## Data privacy

`ar_analytics/data/` is **git-ignored**. The raw exports and derived outputs
contain customer names, outstanding balances, and credit limits, so they are
never committed to version control. Only code and documentation are tracked.

## Open items for the next batches / next steps

- **39 rows** in `ar_aging2` have a blank `invoice_state`, `aging_bucket`, and
  `RECORDNO` — likely export artifacts; confirm whether to drop them.
- `Reversal` / `Reversed` states (142 each) net to zero AR but inflate
  `total_entered`; decide whether reporting should exclude them.
- Aging buckets arrive pre-computed (`Current`, `1-5`, `6-30`, `31-44`,
  `45-60`, `61-90`, `91-120`, `121+`) with a `days_past_due` alongside — usable
  as-is or recomputable as-of any date from `due_date` + the calendar.
