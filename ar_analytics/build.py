"""Consolidate and validate a batch of AR-aging CSV exports.

This is a dependency-free (standard-library only) pipeline for the accounts-
receivable dataset. It classifies each raw CSV by its columns — not its
filename — so hash-named uploads and future batches slot in without renaming,
then it:

  1. unions the invoice-level AR files into one fact table,
  2. keeps the enriched ``calendar_holidays`` file as the date dimension
     (the plain ``calendar`` file is a strict subset and is dropped),
  3. loads the customer master as a dimension,
  4. validates every join and reconciles open AR against the customer master,
  5. writes a consolidated ``ar_invoices.csv`` plus a validation report
     (Markdown + JSON) to ``data/out/``.

Run it with::

    python ar_analytics/build.py

Point it elsewhere with ``--raw`` / ``--out``. Drop the next batch's CSVs into
the raw directory and re-run; classification is by header, so nothing else
needs to change.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# csv fields in the AR files (long consignee_ref / batch strings) can exceed
# the default 128 KiB field cap; lift it so parsing never silently truncates.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# --- file classification -----------------------------------------------------

AR_MARKERS = {"invoice_state", "aging_bucket", "total_due", "due_date"}
HOLIDAY_MARKERS = {"is_fed_holiday", "is_fed_business_day"}
CALENDAR_MARKERS = {"CalendarDay", "CalendarYear"}
CUSTOMER_MARKERS = {"CUSTOMERID", "CREDITLIMIT", "TOTALDUE"}


def classify(header: list[str]) -> str:
    """Return the logical kind of a CSV from its column names."""
    cols = set(header)
    if AR_MARKERS <= cols:
        return "ar"
    if HOLIDAY_MARKERS <= cols and CALENDAR_MARKERS <= cols:
        return "calendar_holidays"
    if CUSTOMER_MARKERS <= cols:
        return "customers"
    if CALENDAR_MARKERS <= cols:
        return "calendar"
    return "unknown"


@dataclass
class Loaded:
    ar_rows: list[dict] = field(default_factory=list)
    ar_headers: list[str] = field(default_factory=list)
    ar_files: list[str] = field(default_factory=list)
    customers: list[dict] = field(default_factory=list)
    calendar_days: set[str] = field(default_factory=set)
    calendar_kind: str | None = None  # which calendar file supplied the dim
    dropped: list[str] = field(default_factory=list)  # redundant/unknown files


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return (reader.fieldnames or []), rows


def load_batch(raw_dir: Path) -> Loaded:
    """Read every ``*.csv`` in ``raw_dir`` and route it by column layout."""
    loaded = Loaded()
    # Prefer the enriched calendar; only fall back to the plain one if that is
    # all the batch provides.
    calendar_candidates: dict[str, set[str]] = {}

    for path in sorted(raw_dir.glob("*.csv")):
        header, rows = read_csv(path)
        kind = classify(header)
        if kind == "ar":
            if not loaded.ar_headers:
                loaded.ar_headers = header
            loaded.ar_rows.extend(rows)
            loaded.ar_files.append(path.name)
        elif kind == "customers":
            loaded.customers.extend(rows)
        elif kind in ("calendar_holidays", "calendar"):
            calendar_candidates[kind] = {r["CalendarDay"] for r in rows}
        else:
            loaded.dropped.append(f"{path.name} (unrecognized columns)")

    if "calendar_holidays" in calendar_candidates:
        loaded.calendar_kind = "calendar_holidays"
        loaded.calendar_days = calendar_candidates["calendar_holidays"]
        if "calendar" in calendar_candidates:
            loaded.dropped.append("calendar.csv (subset of calendar_holidays.csv)")
    elif "calendar" in calendar_candidates:
        loaded.calendar_kind = "calendar"
        loaded.calendar_days = calendar_candidates["calendar"]

    return loaded


# --- helpers -----------------------------------------------------------------


def to_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except ValueError:
        return 0.0


def money(value: float) -> str:
    return f"${value:,.2f}"


# --- validation --------------------------------------------------------------


@dataclass
class Report:
    ok: bool
    lines: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def add(self, passed: bool, text: str) -> None:
        mark = "PASS" if passed else "FAIL"
        self.lines.append(f"- [{mark}] {text}")
        if not passed:
            self.ok = False


def validate(loaded: Loaded) -> Report:
    rep = Report(ok=True)
    ar = loaded.ar_rows
    customers = loaded.customers
    cust_ids = {c["CUSTOMERID"] for c in customers}

    # Coverage / cardinality.
    ar_cust_ids = {r["CUSTOMERID"] for r in ar if r.get("CUSTOMERID")}
    orphans = ar_cust_ids - cust_ids
    rep.add(not orphans, f"AR customers resolve to master: {len(ar_cust_ids)} distinct, "
            f"{len(orphans)} orphaned")

    due_dates = {r["due_date"] for r in ar if r.get("due_date")}
    missing_days = due_dates - loaded.calendar_days if loaded.calendar_days else due_dates
    rep.add(bool(loaded.calendar_days) and not missing_days,
            f"AR due_dates covered by calendar: {len(missing_days)} missing")

    dup_cust = [cid for cid, n in Counter(c["CUSTOMERID"] for c in customers).items() if n > 1]
    rep.add(not dup_cust, f"CUSTOMERID unique in master: {len(dup_cust)} duplicates")

    # Reconciliation: open AR vs. customer-master balances (allow small drift).
    ar_open = sum(to_float(r.get("total_due")) for r in ar)
    master_due = sum(to_float(c.get("TOTALDUE")) for c in customers)
    drift = ar_open - master_due
    tol = max(1.0, abs(master_due) * 0.01)  # 1% tolerance
    rep.add(abs(drift) <= tol,
            f"Open AR reconciles to master TOTALDUE: {money(ar_open)} vs {money(master_due)} "
            f"(drift {money(drift)})")

    # Descriptive stats (not pass/fail).
    states = Counter(r.get("invoice_state") or "(blank)" for r in ar)
    buckets = Counter(r.get("aging_bucket") or "(blank)" for r in ar)
    status = Counter(c.get("STATUS") or "(blank)" for c in customers)
    invoiced = sum(to_float(r.get("total_entered")) for r in ar)

    rep.data = {
        "batch_files": {
            "ar_fact": loaded.ar_files,
            "date_dimension": loaded.calendar_kind,
            "dropped": loaded.dropped,
        },
        "ar_invoice_lines": len(ar),
        "customers": {"total": len(customers), "by_status": dict(status)},
        "distinct_ar_customers": len(ar_cust_ids),
        "totals": {
            "total_entered": round(invoiced, 2),
            "open_total_due": round(ar_open, 2),
            "master_total_due": round(master_due, 2),
            "reconciliation_drift": round(drift, 2),
        },
        "invoice_state_counts": dict(states),
        "aging_bucket_counts": dict(buckets),
    }
    return rep


# --- outputs -----------------------------------------------------------------


def write_consolidated_ar(loaded: Loaded, out_dir: Path) -> Path:
    dest = out_dir / "ar_invoices.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=loaded.ar_headers)
        writer.writeheader()
        writer.writerows(loaded.ar_rows)
    return dest


def write_reports(rep: Report, out_dir: Path) -> tuple[Path, Path]:
    json_path = out_dir / "validation_report.json"
    json_path.write_text(
        json.dumps({"ok": rep.ok, **rep.data}, indent=2), encoding="utf-8"
    )

    d = rep.data
    md = out_dir / "validation_report.md"
    lines = [
        "# AR batch validation report",
        "",
        f"**Overall: {'PASS' if rep.ok else 'FAIL'}**",
        "",
        "## Checks",
        *rep.lines,
        "",
        "## Batch composition",
        f"- AR fact files unioned: {', '.join(d['batch_files']['ar_fact']) or '(none)'}",
        f"- Date dimension: {d['batch_files']['date_dimension']}",
        f"- Dropped: {', '.join(d['batch_files']['dropped']) or '(none)'}",
        "",
        "## Totals",
        f"- AR invoice lines: {d['ar_invoice_lines']:,}",
        f"- Distinct AR customers: {d['distinct_ar_customers']:,}",
        f"- Customers in master: {d['customers']['total']:,} {d['customers']['by_status']}",
        f"- Total invoiced (total_entered): {money(d['totals']['total_entered'])}",
        f"- Open AR (total_due): {money(d['totals']['open_total_due'])}",
        f"- Master TOTALDUE: {money(d['totals']['master_total_due'])}",
        f"- Reconciliation drift: {money(d['totals']['reconciliation_drift'])}",
        "",
        "## Invoice state",
        *[f"- {k}: {v:,}" for k, v in sorted(d["invoice_state_counts"].items())],
        "",
        "## Aging buckets",
        *[f"- {k}: {v:,}" for k, v in sorted(d["aging_bucket_counts"].items())],
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md, json_path


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=here / "data" / "raw",
                        help="directory of raw batch CSVs")
    parser.add_argument("--out", type=Path, default=here / "data" / "out",
                        help="directory for consolidated outputs + reports")
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"error: raw directory not found: {args.raw}", file=sys.stderr)
        print("       drop the batch CSVs there and re-run.", file=sys.stderr)
        return 2

    loaded = load_batch(args.raw)
    if not loaded.ar_rows:
        print(f"error: no AR-aging files found in {args.raw}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    rep = validate(loaded)
    ar_path = write_consolidated_ar(loaded, args.out)
    md_path, json_path = write_reports(rep, args.out)

    print(f"Unioned {len(loaded.ar_rows):,} AR lines from {loaded.ar_files} "
          f"-> {ar_path.name}")
    print(f"Date dimension: {loaded.calendar_kind}; dropped: {loaded.dropped or '(none)'}")
    print(f"Reports: {md_path.name}, {json_path.name}")
    print(f"Validation: {'PASS' if rep.ok else 'FAIL'}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
