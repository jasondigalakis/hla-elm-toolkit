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
hla_elm_toolkit.data
=====================

Genotype / haplotype / diplotype representations used throughout this
toolkit, and a synthetic reference-population generator used by the
examples and tests (the real HTO/ORAM/GRPT registry data used in the
article is not distributable -- see the article's Data Availability
Statement -- so a synthetic population with realistic allele-frequency
and linkage-disequilibrium structure is provided instead).

Corresponds to article Sections 2.1-2.2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# The five loci considered throughout the article (Section 2.1).
DEFAULT_LOCI: Tuple[str, ...] = ("A", "B", "C", "DRB1", "DQB1")

Allele = str
Haplotype = Tuple[Allele, ...]          # one allele per locus, in DEFAULT_LOCI order
Diplotype = Tuple[Haplotype, Haplotype]  # an unordered pair of haplotypes


@dataclass
class Genotype:
    """
    A single donor's typing record.

    ``calls`` maps each locus to the set of alleles the low/mid-resolution
    typing is compatible with (Section 2.2): a single allele for an
    unambiguous high-resolution call, several alleles for an ambiguous
    low/intermediate-resolution call, or ``None`` if the locus was not
    typed at all (missing-locus imputation task, Section 2.2 / 2.4).
    """

    donor_id: str
    calls: Dict[str, Optional[FrozenSet[Allele]]]
    truth: Optional[Dict[str, Haplotype]] = None  # confirmatory high-res truth, if known (2 alleles/locus)

    def is_typed(self, locus: str) -> bool:
        return self.calls.get(locus) is not None

    def is_ambiguous(self, locus: str) -> bool:
        s = self.calls.get(locus)
        return s is not None and len(s) > 1


@dataclass
class ReferencePopulation:
    """
    A reference haplotype-frequency table plus the donor records used to
    estimate it, following the offline/online split described for
    HaploStats, GRIMM, and Hapl-o-Mat in Section 2.7.1 (Algorithms 3-5).
    """

    loci: Tuple[str, ...]
    haplotype_freq: Dict[Haplotype, float] = field(default_factory=dict)
    donors: List[Genotype] = field(default_factory=list)

    def alleles_at(self, locus: str) -> List[Allele]:
        idx = self.loci.index(locus)
        return sorted({h[idx] for h in self.haplotype_freq})


def compatible_diplotypes(
    genotype: Genotype,
    ref: ReferencePopulation,
    loci: Optional[Sequence[str]] = None,
) -> List[Diplotype]:
    """
    Enumerate the diplotypes in ``ref.haplotype_freq`` that are compatible
    with ``genotype`` at the requested loci (Section 2.2/2.4: for a typed,
    possibly ambiguous, locus the haplotype's allele must be one of the
    observed compatible alleles; for an untyped locus, every reference
    allele at that locus is allowed -- missing-locus imputation).

    This is a small reference implementation intended for demonstration
    and unit testing on modestly sized populations; it enumerates
    candidate haplotype pairs directly rather than using an indexed graph
    structure, and is therefore not intended for registry-scale (10^5+
    donor) throughput without further optimization (see README).
    """
    loci = tuple(loci) if loci is not None else ref.loci
    idxs = [ref.loci.index(l) for l in loci]

    def allele_ok(hap: Haplotype) -> bool:
        for locus, idx in zip(loci, idxs):
            allowed = genotype.calls.get(locus)
            if allowed is None:
                continue  # untyped locus: any reference allele is allowed
            if hap[idx] not in allowed:
                return False
        return True

    candidate_haps = [h for h in ref.haplotype_freq if allele_ok(h)]
    diplotypes: List[Diplotype] = []
    n = len(candidate_haps)
    for i in range(n):
        for j in range(i, n):
            diplotypes.append((candidate_haps[i], candidate_haps[j]))
    return diplotypes


def hardy_weinberg_likelihood(dt: Diplotype, ref: ReferencePopulation) -> float:
    """Likelihood of a diplotype under Hardy-Weinberg proportions (Section 2.3)."""
    h1, h2 = dt
    f1 = ref.haplotype_freq.get(h1, 0.0)
    f2 = ref.haplotype_freq.get(h2, 0.0)
    if h1 == h2:
        return f1 * f1
    return 2.0 * f1 * f2


def posterior_probabilities(
    diplotypes: List[Diplotype], ref: ReferencePopulation
) -> List[Tuple[Diplotype, float]]:
    """
    Normalize Hardy-Weinberg likelihoods into posterior probabilities over
    a compatible-diplotype set (Section 2.3): Post-P(dt) = L(dt) / sum_i L(d_i).
    Returned sorted by posterior probability, descending.
    """
    scored = [(dt, hardy_weinberg_likelihood(dt, ref)) for dt in diplotypes]
    total = sum(l for _, l in scored)
    if total <= 0:
        return []
    post = [(dt, l / total) for dt, l in scored]
    post.sort(key=lambda x: x[1], reverse=True)
    return post


# --------------------------------------------------------------------------- #
# Synthetic reference-population generator (for examples/tests only)
# --------------------------------------------------------------------------- #

def _zipf_like_frequencies(n: int, rng: random.Random, s: float = 1.3) -> List[float]:
    """Rough Zipf-like allele-frequency spectrum: a few common alleles, many rare ones."""
    raw = [1.0 / ((i + 1) ** s) for i in range(n)]
    jitter = [r * (0.85 + 0.3 * rng.random()) for r in raw]
    total = sum(jitter)
    return [j / total for j in jitter]


def make_synthetic_population(
    n_donors: int = 2000,
    loci: Sequence[str] = DEFAULT_LOCI,
    n_alleles_per_locus: int = 12,
    n_haplotypes: int = 60,
    seed: int = 0,
) -> ReferencePopulation:
    """
    Build a small synthetic reference population with a Zipf-like
    haplotype-frequency spectrum (rare-allele-dominated tail, mirroring
    the pattern quantified in article Table 5), for use in the examples
    and unit tests. This is NOT the HTO/ORAM/GRPT registry data used in
    the article, which is not distributable (Data Availability Statement).
    """
    rng = random.Random(seed)
    loci = tuple(loci)
    alleles_by_locus = {
        locus: [f"{locus}*{i:02d}:01" for i in range(1, n_alleles_per_locus + 1)]
        for locus in loci
    }

    haplotypes: List[Haplotype] = []
    seen = set()
    while len(haplotypes) < n_haplotypes:
        hap = tuple(rng.choice(alleles_by_locus[locus]) for locus in loci)
        if hap not in seen:
            seen.add(hap)
            haplotypes.append(hap)

    freqs = _zipf_like_frequencies(len(haplotypes), rng)
    haplotype_freq = dict(zip(haplotypes, freqs))

    ref = ReferencePopulation(loci=loci, haplotype_freq=haplotype_freq)

    weights = [haplotype_freq[h] for h in haplotypes]
    for i in range(n_donors):
        h1 = rng.choices(haplotypes, weights=weights, k=1)[0]
        h2 = rng.choices(haplotypes, weights=weights, k=1)[0]
        truth = {locus: (h1[idx], h2[idx]) for idx, locus in enumerate(loci)}

        calls: Dict[str, Optional[FrozenSet[Allele]]] = {}
        for idx, locus in enumerate(loci):
            r = rng.random()
            true_alleles = frozenset({h1[idx], h2[idx]})
            if r < 0.15:
                calls[locus] = None  # untyped (missing-locus imputation)
            elif r < 0.45:
                # ambiguous low/intermediate-resolution call: true allele(s)
                # plus 1-2 confusable alternatives from the same locus
                pool = [a for a in alleles_by_locus[locus] if a not in true_alleles]
                extra = rng.sample(pool, k=min(2, len(pool)))
                calls[locus] = frozenset(true_alleles | set(extra))
            else:
                calls[locus] = true_alleles  # unambiguous high-resolution call

        ref.donors.append(
            Genotype(donor_id=f"D{i:06d}", calls=calls, truth=truth)
        )

    return ref
