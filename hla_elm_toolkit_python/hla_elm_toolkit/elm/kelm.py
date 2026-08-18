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
hla_elm_toolkit.elm.kelm
==========================

Kernel ELM (KELM), following Huang et al. (article ref. [28]) as
described in article Section 2.6: the explicit random hidden layer is
replaced by an implicit Mercer kernel matrix Omega, with predictions
f(x) = [K(x,x_1),...,K(x,x_n)] (Omega + I/C)^-1 T. Removes the need to
choose a hidden-layer size, at the cost of an NxN kernel matrix that
scales less favorably to large training cohorts than the base ELM's
random-feature closed-form solution (Figure 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data import Diplotype, Genotype, ReferencePopulation, compatible_diplotypes
from .base_elm import InputEncoder, OutputCatalog


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian (RBF) kernel matrix K(x_i, y_j) = exp(-gamma * ||x_i - y_j||^2)."""
    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
    sq_dists = X_sq + Y_sq - 2 * X @ Y.T
    np.clip(sq_dists, 0, None, out=sq_dists)
    return np.exp(-gamma * sq_dists)


@dataclass
class KELM:
    C: float = 1.0
    gamma: float = 0.1

    encoder: Optional[InputEncoder] = None
    catalog: Optional[OutputCatalog] = None
    X_train: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None  # (Omega + I/C)^-1 T, precomputed

    def fit(
        self,
        genotypes: Sequence[Genotype],
        ref: ReferencePopulation,
        max_diplotypes: Optional[int] = 4000,
        rng_seed: int = 0,
    ) -> "KELM":
        self.encoder = InputEncoder.fit(ref)
        self.catalog = OutputCatalog.from_reference(ref, max_diplotypes=max_diplotypes)
        rng = np.random.default_rng(rng_seed)

        X_rows: List[np.ndarray] = []
        T_rows: List[np.ndarray] = []
        for g in genotypes:
            candidates = [dt for dt in compatible_diplotypes(g, ref) if dt in self.catalog.index]
            if not candidates:
                continue
            target = candidates[rng.integers(0, len(candidates))]
            x = self.encoder.encode(g, ref)
            t = np.zeros(self.catalog.n_outputs, dtype=np.float64)
            t[self.catalog.index[target]] = 1.0
            X_rows.append(x)
            T_rows.append(t)

        if not X_rows:
            raise ValueError("No training genotype had a compatible diplotype in the output catalog.")

        self.X_train = np.vstack(X_rows)
        T = np.vstack(T_rows)
        Omega = rbf_kernel(self.X_train, self.X_train, self.gamma)
        I = np.eye(Omega.shape[0])
        self.alpha = np.linalg.solve(Omega + I / self.C, T)
        return self

    def predict_scores(self, genotype: Genotype, ref: ReferencePopulation) -> np.ndarray:
        x = self.encoder.encode(genotype, ref).reshape(1, -1)
        k = rbf_kernel(x, self.X_train, self.gamma)
        return (k @ self.alpha).ravel()

    def predict_ranked(
        self, genotype: Genotype, ref: ReferencePopulation, top_k: int = 5
    ) -> List[Tuple[Diplotype, float]]:
        candidates = [dt for dt in compatible_diplotypes(genotype, ref) if dt in self.catalog.index]
        if not candidates:
            return []
        scores_all = self.predict_scores(genotype, ref)
        idxs = [self.catalog.index[dt] for dt in candidates]
        raw = scores_all[idxs]
        raw = raw - raw.max()
        probs = np.exp(raw)
        probs = probs / probs.sum()
        ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
