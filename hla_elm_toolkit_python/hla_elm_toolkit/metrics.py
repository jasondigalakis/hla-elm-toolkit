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

"""
hla_elm_toolkit.metrics
=========================

Evaluation metrics as defined in article Section 2.3 (per-locus accuracy,
overall/multilocus accuracy, call rate, posterior probability, top-k
concordance) and the confidence-interval methodology of Section 2.8
(Wilson score intervals, chosen over a naive percentile bootstrap because
several of the article's small-sample comparator results sit at or near
the 0%/100% boundary where a naive bootstrap is uninformative).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Diplotype, Genotype, Haplotype, ReferencePopulation


@dataclass
class PredictionResult:
    """One donor's prediction outcome at one locus (or jointly across loci)."""

    donor_id: str
    called: bool               # True if a call was returned at/above threshold
    correct: Optional[bool]    # True/False if truth is known and a call was made; else None
    post_p: Optional[float] = None


def wilson_score_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    Preferred over a naive percentile bootstrap or normal-approximation
    (Wald) interval in this toolkit because several comparator results in
    the article sit at n as small as 2-11 and/or at 0% or 100% observed
    accuracy (Sections 3.10-3.11); a bootstrap resampled directly from a
    fully homogeneous 0%/100% sample is degenerate (zero width) regardless
    of n, whereas the Wilson interval remains well behaved (Section 2.8).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")

    z = {0.90: 1.644853627, 0.95: 1.959963985, 0.99: 2.575829304}.get(confidence)
    if z is None:
        raise ValueError("confidence must be one of 0.90, 0.95, 0.99")

    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return lo, hi


def call_rate(results: Iterable[PredictionResult]) -> Tuple[float, int, int]:
    """Proportion of submitted genotypes for which a call was returned (Section 2.3)."""
    results = list(results)
    n = len(results)
    k = sum(1 for r in results if r.called)
    return (k / n if n else 0.0), k, n


def accuracy_among_called(results: Iterable[PredictionResult]) -> Tuple[float, int, int]:
    """
    Accuracy among the donors for whom a call was actually returned
    (per-locus or overall/multilocus accuracy, Section 2.3, depending on
    what ``results`` represents).
    """
    called = [r for r in results if r.called and r.correct is not None]
    n = len(called)
    k = sum(1 for r in called if r.correct)
    return (k / n if n else 0.0), k, n


def top_k_concordance(
    ranked_predictions: Sequence[Tuple[Haplotype, Haplotype]],
    truth: Tuple[Haplotype, Haplotype],
    k: int,
) -> bool:
    """
    Whether the confirmatory diplotype is among the top-k highest-posterior
    predictions (Section 2.3), comparing diplotypes as unordered
    haplotype pairs.
    """
    truth_set = frozenset(truth)
    for pred in ranked_predictions[:k]:
        if frozenset(pred) == truth_set:
            return True
    return False


def summarize(
    results: Iterable[PredictionResult],
    digits: int = 1,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """
    Convenience summary combining call rate and accuracy-among-called with
    Wilson score 95% CIs, in the compact "value [lo-hi%]" style used
    throughout the article's tables (e.g. Tables 16-19).
    """
    results = list(results)
    cr, k_call, n_call = call_rate(results)
    acc, k_acc, n_acc = accuracy_among_called(results)

    out: Dict[str, object] = {
        "n_total": n_call,
        "call_rate": cr,
        "call_rate_count": f"{k_call}/{n_call}",
    }
    if n_call > 0:
        lo, hi = wilson_score_interval(k_call, n_call, confidence)
        out["call_rate_ci"] = (round(lo * 100, digits), round(hi * 100, digits))
    if n_acc > 0:
        out["accuracy"] = acc
        out["accuracy_count"] = f"{k_acc}/{n_acc}"
        lo, hi = wilson_score_interval(k_acc, n_acc, confidence)
        out["accuracy_ci"] = (round(lo * 100, digits), round(hi * 100, digits))
    else:
        out["accuracy"] = None
    return out


def format_summary(name: str, s: Dict[str, object]) -> str:
    """Render a `summarize()` dict as one human-readable line."""
    parts = [f"{name}:"]
    parts.append(
        f"call rate {s['call_rate']*100:.1f}% ({s['call_rate_count']})"
        + (f" [{s['call_rate_ci'][0]}-{s['call_rate_ci'][1]}%]" if "call_rate_ci" in s else "")
    )
    if s.get("accuracy") is not None:
        parts.append(
            f"accuracy {s['accuracy']*100:.1f}% ({s['accuracy_count']})"
            + (f" [{s['accuracy_ci'][0]}-{s['accuracy_ci'][1]}%]" if "accuracy_ci" in s else "")
        )
    else:
        parts.append("accuracy n/a (no called donors with known truth)")
    return " ".join(parts)
