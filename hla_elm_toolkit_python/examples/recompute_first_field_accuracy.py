# -----------------------------------------------------------------------------
# This file is part of hla_elm_toolkit, open-source software developed to
# accompany:
#   Kepentzis, S.; Chatzistamatiou, T.; Digalakis, J.; Petropoulou, O.;
#   Matsopoulos, G.K.; Koutsouris, D. "Development of an Extreme Learning
#   Machine approach to upgrade low/mid to high resolution HLA data
#   improving the usability of donor data from registries." Genes (MDPI).
#
# Software authors: Stavros Kepentzis (skepenjis@biomed.ntua.gr)
#                    Jason Digalakis  (jdigalakis@biomed.ntua.gr)
# Affiliation: Biomedical Engineering Laboratory, School of Electrical and
#              Computer Engineering, National Technical University of
#              Athens (NTUA), Athens, Greece
#
# Released as open-source software.
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
"""
recompute_first_field_accuracy.py
===================================

Addresses article gap **Gamma.1**: GRIMM, hlaR ImputeHaplo, and the
in-house EM baseline (Sections 3.8-3.11) are scored at first-field
resolution, while the ELM framework's headline accuracy figures are
reported at two-field resolution -- so Tables 16-19 are not currently a
like-for-like comparison. This script closes that gap **without any new
model runs**: it takes the ELM's *already-computed* two-field
predictions and confirmatory truth, truncates both to first-field, and
recomputes call rate / per-locus accuracy / joint accuracy with 95%
Wilson score confidence intervals in exactly the format used throughout
the article's tables.

-----------------------------------------------------------------------
INPUT FILE FORMAT
-----------------------------------------------------------------------
A CSV file with one row per (donor, locus) prediction, columns:

    donor_id, cohort, locus, called, pred_allele_1, pred_allele_2,
    true_allele_1, true_allele_2

  - donor_id       : any string/int identifying the donor
  - cohort         : e.g. "HTO", "ORAM", "GRPT" (free text; used only
                      for grouping in the output)
  - locus          : one of A, B, C, DRB1, DQB1 (or your own locus set)
  - called         : "1"/"0"/"true"/"false" -- whether the ELM returned
                      a call at/above its confidence threshold for this
                      donor/locus (a row with called=0 is still counted
                      in the call-rate denominator but excluded from the
                      accuracy numerator/denominator, matching Section 2.3)
  - pred_allele_1/2: the ELM's two-field call, e.g. "A*02:01" (leave
                      blank if called=0)
  - true_allele_1/2: confirmatory two-field truth, e.g. "A*02:05"

Adjust ``REQUIRED_COLUMNS`` / ``read_rows()`` below if your working
prediction log uses different column names -- the truncation and
scoring logic itself does not need to change.

-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------
    python recompute_first_field_accuracy.py predictions.csv \\
        --cohort ORAM --out oram_first_field_summary.csv

    python recompute_first_field_accuracy.py predictions.csv \\
        --joint    # additionally report all-loci-simultaneously accuracy

Run once per cohort (HTO / ORAM / GRPT) to get the three rows needed to
extend the "ELM (reported, overall)" rows of Tables 16-19 with a
first-field figure alongside the existing two-field one.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REQUIRED_COLUMNS = [
    "donor_id", "cohort", "locus", "called",
    "pred_allele_1", "pred_allele_2", "true_allele_1", "true_allele_2",
]


def truncate_first_field(allele: str) -> str:
    """
    "A*02:01:01" -> "A*02" ; "A*02:01" -> "A*02" ; "A*02" -> "A*02".
    Returns "" unchanged for an empty/missing allele string.
    """
    if not allele:
        return allele
    if "*" not in allele:
        return allele  # not in expected LOCUS*FIELD:FIELD... format; leave as-is
    locus, _, fields = allele.partition("*")
    first_field = fields.split(":")[0]
    return f"{locus}*{first_field}"


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y")


@dataclass
class Row:
    donor_id: str
    cohort: str
    locus: str
    called: bool
    pred: Optional[Tuple[str, str]]
    true: Optional[Tuple[str, str]]


def read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Input file is missing required column(s): {sorted(missing)}\n"
                f"Found columns: {reader.fieldnames}\n"
                f"See the module docstring for the expected format, or adapt "
                f"read_rows() to your own column names."
            )
        for r in reader:
            called = _to_bool(r["called"])
            pred = (r["pred_allele_1"], r["pred_allele_2"]) if called else None
            true = (
                (r["true_allele_1"], r["true_allele_2"])
                if r["true_allele_1"] and r["true_allele_2"]
                else None
            )
            rows.append(
                Row(
                    donor_id=r["donor_id"],
                    cohort=r["cohort"],
                    locus=r["locus"],
                    called=called,
                    pred=pred,
                    true=true,
                )
            )
    return rows


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Same Wilson score method used throughout hla_elm_toolkit.metrics and the article (Section 2.8)."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = {0.90: 1.644853627, 0.95: 1.959963985, 0.99: 2.575829304}[confidence]
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def first_field_match(pred: Tuple[str, str], true: Tuple[str, str]) -> bool:
    """Unordered-pair match at first-field resolution (matches the article's
    'first-field top-1 concordance' definition used for GRIMM/hlaR/EM baselines,
    Sections 3.8-3.11)."""
    pred_ff = frozenset(truncate_first_field(a) for a in pred)
    true_ff = frozenset(truncate_first_field(a) for a in true)
    return pred_ff == true_ff


def summarize_locus(rows: List[Row]) -> Dict[str, object]:
    n_total = len(rows)
    n_called = sum(1 for r in rows if r.called)
    correct = sum(
        1 for r in rows
        if r.called and r.pred is not None and r.true is not None
        and first_field_match(r.pred, r.true)
    )
    n_scoreable = sum(
        1 for r in rows if r.called and r.pred is not None and r.true is not None
    )

    out: Dict[str, object] = {"n_total": n_total, "n_called": n_called}
    if n_total > 0:
        lo, hi = wilson_score_interval(n_called, n_total)
        out["call_rate_pct"] = round(100 * n_called / n_total, 1)
        out["call_rate_ci"] = (round(lo * 100, 1), round(hi * 100, 1))
    if n_scoreable > 0:
        lo, hi = wilson_score_interval(correct, n_scoreable)
        out["accuracy_pct"] = round(100 * correct / n_scoreable, 1)
        out["accuracy_ci"] = (round(lo * 100, 1), round(hi * 100, 1))
        out["n_scoreable"] = n_scoreable
        out["n_correct"] = correct
    return out


def summarize_joint(rows: List[Row]) -> Dict[str, object]:
    """All-loci-simultaneously-correct accuracy at first-field resolution (Section 2.3 'overall accuracy')."""
    by_donor: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_donor[r.donor_id].append(r)

    n_total = len(by_donor)
    n_called = 0
    n_scoreable = 0
    n_correct = 0
    for donor_id, donor_rows in by_donor.items():
        if not all(r.called for r in donor_rows):
            continue
        n_called += 1
        if not all(r.pred is not None and r.true is not None for r in donor_rows):
            continue
        n_scoreable += 1
        if all(first_field_match(r.pred, r.true) for r in donor_rows):
            n_correct += 1

    out: Dict[str, object] = {"n_total": n_total, "n_called": n_called}
    if n_total > 0:
        lo, hi = wilson_score_interval(n_called, n_total)
        out["call_rate_pct"] = round(100 * n_called / n_total, 1)
        out["call_rate_ci"] = (round(lo * 100, 1), round(hi * 100, 1))
    if n_scoreable > 0:
        lo, hi = wilson_score_interval(n_correct, n_scoreable)
        out["accuracy_pct"] = round(100 * n_correct / n_scoreable, 1)
        out["accuracy_ci"] = (round(lo * 100, 1), round(hi * 100, 1))
        out["n_scoreable"] = n_scoreable
        out["n_correct"] = n_correct
    return out


def format_row(label: str, s: Dict[str, object]) -> str:
    parts = [label]
    if "call_rate_pct" in s:
        lo, hi = s["call_rate_ci"]
        parts.append(f"call rate {s['call_rate_pct']}% ({s['n_called']}/{s['n_total']}) [{lo}-{hi}%]")
    if "accuracy_pct" in s:
        lo, hi = s["accuracy_ci"]
        parts.append(f"accuracy {s['accuracy_pct']}% ({s['n_correct']}/{s['n_scoreable']}) [{lo}-{hi}%]")
    else:
        parts.append("accuracy n/a (no scoreable donors)")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, help="Path to the ELM two-field prediction log (see docstring for format).")
    parser.add_argument("--cohort", type=str, default=None, help="Restrict to one cohort (e.g. HTO, ORAM, GRPT). Default: all rows.")
    parser.add_argument("--joint", action="store_true", help="Also report all-five-loci joint accuracy.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write a summary CSV.")
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if args.cohort:
        rows = [r for r in rows if r.cohort == args.cohort]
        if not rows:
            raise SystemExit(f"No rows found for cohort={args.cohort!r}. "
                              f"Cohorts present: {sorted({r.cohort for r in read_rows(args.input_csv)})}")

    by_locus: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_locus[r.locus].append(r)

    print(f"ELM first-field re-scoring ({args.cohort or 'all cohorts'}, n donor-locus rows = {len(rows)})")
    print("-" * 78)

    summary_rows = []
    for locus in sorted(by_locus):
        s = summarize_locus(by_locus[locus])
        print(format_row(f"  {locus:6s}", s))
        summary_rows.append({"locus": locus, **s})

    if args.joint:
        s = summarize_joint(rows)
        print(format_row("  JOINT (all loci)", s))
        summary_rows.append({"locus": "JOINT", **s})

    if args.out:
        fieldnames = sorted({k for row in summary_rows for k in row})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nWrote summary to {args.out}")

    print(
        "\nNext step: paste these first-field figures into the 'ELM (reported, "
        "overall)' row of the relevant table (16=ORAM/GRPT via GRIMM, 17=HTO, "
        "18=hlaR ImputeHaplo, 19=in-house EM baseline), alongside the existing "
        "two-field figure, with both explicitly labelled by resolution."
    )


if __name__ == "__main__":
    main()
