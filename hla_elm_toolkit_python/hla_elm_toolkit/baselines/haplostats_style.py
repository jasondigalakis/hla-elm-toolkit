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
hla_elm_toolkit.baselines.haplostats_style
=============================================

HaploStats-style haplotype-frequency imputation, following Algorithm 4 of
the article ("HaploStats-style haplotype-frequency imputation, simplified,
based on the published NMDP method [17]", Section 2.7.1).

Unlike ``haplo_em.HaploEM`` (which *estimates* haplotype frequencies from
genotype data via EM), this module assumes a haplotype-frequency table is
already available (e.g. pre-estimated offline from a large multi-ethnic
reference panel, as the real HaploStats service does) and performs only
the online diplotype-enumeration-and-scoring step. This mirrors the
article's own approach in Section 3.10, where the published NMDP
HaploStats web service could not be batch-queried, so an open-source,
HaploStats-inspired baseline was substituted (article Section 3.10, 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..data import (
    Diplotype,
    Genotype,
    ReferencePopulation,
    compatible_diplotypes,
    posterior_probabilities,
)


@dataclass
class HaploStatsStyleImputer:
    """Direct posterior-probability scoring against a pre-estimated haplotype-frequency table."""

    reference: ReferencePopulation

    def impute(
        self, genotype: Genotype, theta: float = 0.0, top_k: int = 1
    ) -> List[Tuple[Diplotype, float]]:
        """
        Algorithm 4, lines 1-15: enumerate compatible diplotypes, score by
        normalized Hardy-Weinberg likelihood, and return the top-k ranked
        list (empty list if no compatible diplotype exists in the
        reference table -- the "no_match" case, article Algorithm 4 line 3).
        """
        dts = compatible_diplotypes(genotype, self.reference)
        ranked = posterior_probabilities(dts, self.reference)
        return ranked[:top_k]

    def top1_call(
        self, genotype: Genotype, theta: float = 0.0
    ) -> Tuple[Optional[Diplotype], Optional[float]]:
        ranked = self.impute(genotype, top_k=1)
        if not ranked:
            return None, None
        dt, p = ranked[0]
        return (dt, p) if p >= theta else (None, p)
