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
hla_elm_toolkit.baselines.haplo_em
=====================================

Expectation-maximization (EM) haplotype-frequency estimation and
imputation, following Algorithm 5 of the article ("Hapl-o-Mat: open-source
EM haplotype-frequency estimation and imputation", Section 2.7.1).

This is the same core algorithm used for:
  * the two-locus (HLA-A~HLA-B) baseline run directly on the registry
    data in article Section 3.7, and
  * the five-locus "in-house Hapl-o-Mat-style" baseline of Section 3.11,

by simply changing which loci are passed in. It is explicitly an
**approximation** of the published Hapl-o-Mat tool's core EM logic, not
a reproduction of Hapl-o-Mat itself: population-weighting options,
specific convergence criteria, and ambiguity-resolution details of the
published tool are not reproduced here (article Sections 2.7.1, 3.11,
and 5 make the same distinction for the original implementation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..data import Diplotype, Genotype, Haplotype, ReferencePopulation, compatible_diplotypes


@dataclass
class HaploEM:
    """Genotype-based EM estimator for multi-locus haplotype frequencies."""

    loci: Tuple[str, ...]
    max_iter: int = 100
    tol: float = 1e-6

    haplotype_freq: Dict[Haplotype, float] = field(default_factory=dict)
    n_iterations_run: int = 0

    def fit(self, genotypes: Sequence[Genotype], ref_for_enumeration: ReferencePopulation) -> "HaploEM":
        """
        Phase 1 (offline) of Algorithm 5: estimate haplotype frequencies
        from a reference panel of unphased, possibly-ambiguous genotypes.

        ``ref_for_enumeration`` supplies the candidate haplotype universe
        used to enumerate phase-compatible diplotypes for each genotype
        (in the article, this is built directly from the observed allele
        combinations in the training data; see Sections 3.7 and 3.11 for
        the exact candidate-pool sizes on the real registry).
        """
        # Per-genotype compatible diplotype lists (built once, reused every EM iteration).
        per_genotype_dts: List[List[Diplotype]] = []
        for g in genotypes:
            dts = compatible_diplotypes(g, ref_for_enumeration, loci=self.loci)
            if dts:
                per_genotype_dts.append(dts)

        if not per_genotype_dts:
            raise ValueError("No genotype had any phase-compatible diplotype for the given loci.")

        # Initialize F uniformly over every haplotype appearing in any compatible diplotype.
        all_haps = {h for dts in per_genotype_dts for (h1, h2) in dts for h in (h1, h2)}
        F: Dict[Haplotype, float] = {h: 1.0 / len(all_haps) for h in all_haps}

        for iteration in range(self.max_iter):
            # E-step: expected haplotype counts under current F.
            counts: Dict[Haplotype, float] = {h: 0.0 for h in all_haps}
            for dts in per_genotype_dts:
                weights = []
                for (h1, h2) in dts:
                    like = F[h1] * F[h1] if h1 == h2 else F[h1] * F[h2]
                    weights.append(like)
                total = sum(weights)
                if total <= 0:
                    continue
                for (h1, h2), w in zip(dts, weights):
                    post = w / total
                    counts[h1] += post
                    counts[h2] += post

            # M-step: renormalize.
            total_count = sum(counts.values())
            if total_count <= 0:
                break
            F_new = {h: c / total_count for h, c in counts.items()}

            delta = max(abs(F_new[h] - F[h]) for h in all_haps)
            F = F_new
            self.n_iterations_run = iteration + 1
            if delta < self.tol:
                break

        self.haplotype_freq = F
        return self

    def as_reference_population(self) -> ReferencePopulation:
        """Wrap the fitted frequencies as a ``ReferencePopulation`` for scoring/inference."""
        return ReferencePopulation(loci=self.loci, haplotype_freq=dict(self.haplotype_freq))

    def score_genotype(
        self, genotype: Genotype, theta: float = 0.0
    ) -> Tuple[Optional[Diplotype], Optional[float]]:
        """
        Phase 2 (online) of Algorithm 5: rank diplotypes compatible with a
        query genotype by posterior probability and return the top-1 call
        if it clears the confidence threshold ``theta``.
        """
        ref = self.as_reference_population()
        dts = compatible_diplotypes(genotype, ref, loci=self.loci)
        if not dts:
            return None, None
        scored = []
        for (h1, h2) in dts:
            like = self.haplotype_freq.get(h1, 0.0) * (
                self.haplotype_freq.get(h1, 0.0) if h1 == h2 else self.haplotype_freq.get(h2, 0.0)
            )
            scored.append(((h1, h2), like))
        total = sum(l for _, l in scored)
        if total <= 0:
            return None, None
        scored = [(dt, l / total) for dt, l in scored]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_dt, top_p = scored[0]
        if top_p >= theta:
            return top_dt, top_p
        return None, top_p
