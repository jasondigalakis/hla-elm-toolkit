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
hla_elm_toolkit.elm.ensemble_elm
==================================

Ensemble ELM, following a strategy comparable to Zhang et al. (article
ref. [30]) as described in article Section 2.6: multiple independently
initialized base learners (ELM or WELM) are trained on bootstrap
resamples of the training partition and combined by weighted majority
voting over the discrete diplotype output, each base learner's vote
weighted by its own posterior probability for that call. Reduces the
random-initialization-driven run-to-run prediction variance of a single
base ELM (article Table 14) without materially increasing training time,
since every base learner remains a single closed-form fit and members can
be trained in parallel.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data import Diplotype, Genotype, ReferencePopulation
from .base_elm import BaseELM
from .welm import WELM


@dataclass
class EnsembleELM:
    n_members: int = 10
    hidden_size: int = 200
    use_welm: bool = False
    reg_lambda: float = 1.0

    members: List[object] = field(default_factory=list)

    def fit(
        self,
        genotypes: Sequence[Genotype],
        ref: ReferencePopulation,
        max_diplotypes: Optional[int] = 4000,
        rng_seed: int = 0,
    ) -> "EnsembleELM":
        rng = np.random.default_rng(rng_seed)
        n = len(genotypes)
        self.members = []
        for m in range(self.n_members):
            boot_idx = rng.integers(0, n, size=n)  # bootstrap resample of the training partition
            boot_sample = [genotypes[i] for i in boot_idx]
            if self.use_welm:
                model = WELM(hidden_size=self.hidden_size, reg_c=self.reg_lambda, seed=m)
            else:
                model = BaseELM(hidden_size=self.hidden_size, reg_lambda=self.reg_lambda, seed=m)
            model.fit(boot_sample, ref, max_diplotypes=max_diplotypes, rng_seed=rng_seed + m)
            self.members.append(model)
        return self

    def predict_ranked(
        self, genotype: Genotype, ref: ReferencePopulation, top_k: int = 5
    ) -> List[Tuple[Diplotype, float]]:
        """
        Weighted majority vote: each member casts its top-1 vote for this
        genotype, weighted by that member's own posterior probability for
        the call; votes are summed per candidate diplotype and normalized.
        """
        votes: dict = defaultdict(float)
        for model in self.members:
            ranked = model.predict_ranked(genotype, ref, top_k=1)
            if not ranked:
                continue
            dt, p = ranked[0]
            votes[dt] += p
        if not votes:
            return []
        total = sum(votes.values())
        ranked = sorted(((dt, v / total) for dt, v in votes.items()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def predict_top1(
        self, genotype: Genotype, ref: ReferencePopulation, theta: float = 0.0
    ) -> Tuple[Optional[Diplotype], Optional[float]]:
        ranked = self.predict_ranked(genotype, ref, top_k=1)
        if not ranked:
            return None, None
        dt, p = ranked[0]
        return (dt, p) if p >= theta else (None, p)
