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
hla_elm_toolkit.baselines.grimm_style
========================================

GRIMM-style graph-based five-locus imputation, following Algorithm 3 of
the article ("GRIMM: graph-based five-locus imputation, simplified, based
on the published method [10]", Section 2.7.1).

The reference haplotype graph is represented here as edge weights between
consecutive-locus allele pairs derived from a haplotype-frequency table
(a simplification documented in the article: "these pseudocode summaries
omit implementation-specific details ... such as GRIMM's exact graph
construction and traversal optimizations", Section 2.7.1). Candidate
paths through the graph are scored as the product of edge weights, and
the highest-likelihood path is returned as the top-1 call, or
``no_match_in_reference_panel`` if no path is consistent with the query
genotype -- matching the two-outcome status the article reports GRIMM
returning (Section 3.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data import Genotype, Haplotype, ReferencePopulation


@dataclass
class GrimmStyleGraph:
    """A simplified reference haplotype graph built from a haplotype-frequency table."""

    loci: Tuple[str, ...]
    edge_weight: Dict[Tuple[int, str, str], float] = field(default_factory=dict)  # (locus_idx, allele_i, allele_i+1) -> freq
    haplotypes: List[Haplotype] = field(default_factory=list)

    @classmethod
    def from_reference(cls, ref: ReferencePopulation) -> "GrimmStyleGraph":
        graph = cls(loci=ref.loci)
        graph.haplotypes = list(ref.haplotype_freq.keys())
        edge_counts: Dict[Tuple[int, str, str], float] = {}
        for hap, freq in ref.haplotype_freq.items():
            for i in range(len(hap) - 1):
                key = (i, hap[i], hap[i + 1])
                edge_counts[key] = edge_counts.get(key, 0.0) + freq
        graph.edge_weight = edge_counts
        return graph

    def path_likelihood(self, hap: Haplotype) -> float:
        """Product of edge weights along a haplotype's locus-to-locus path (Algorithm 3, line 6)."""
        like = 1.0
        for i in range(len(hap) - 1):
            like *= self.edge_weight.get((i, hap[i], hap[i + 1]), 0.0)
        return like

    def compatible_haplotypes(self, genotype: Genotype) -> List[Haplotype]:
        out = []
        for hap in self.haplotypes:
            ok = True
            for idx, locus in enumerate(self.loci):
                allowed = genotype.calls.get(locus)
                if allowed is not None and hap[idx] not in allowed:
                    ok = False
                    break
            if ok:
                out.append(hap)
        return out


GRIMM_IMPUTED = "imputed"
GRIMM_NO_MATCH = "no_match_in_reference_panel"


@dataclass
class GrimmStyleImputer:
    graph: GrimmStyleGraph

    def impute(self, genotype: Genotype) -> Tuple[str, Optional[Tuple[Haplotype, Haplotype]], Optional[float]]:
        """
        Algorithm 3: traverse the reference graph restricted to alleles
        consistent with the query genotype at each typed locus, rank
        candidate diplotypes (pairs of compatible haplotype paths) by the
        product of their path likelihoods, and return the top-1 call with
        its (unnormalized) likelihood, or ``no_match_in_reference_panel``
        if no candidate path exists.
        """
        candidates = self.graph.compatible_haplotypes(genotype)
        if not candidates:
            return GRIMM_NO_MATCH, None, None

        best_dt: Optional[Tuple[Haplotype, Haplotype]] = None
        best_like = -1.0
        n = len(candidates)
        for i in range(n):
            li = self.graph.path_likelihood(candidates[i])
            for j in range(i, n):
                lj = li if i == j else self.graph.path_likelihood(candidates[j])
                like = li * lj
                if like > best_like:
                    best_like = like
                    best_dt = (candidates[i], candidates[j])

        if best_dt is None or best_like <= 0:
            return GRIMM_NO_MATCH, None, None
        return GRIMM_IMPUTED, best_dt, best_like
