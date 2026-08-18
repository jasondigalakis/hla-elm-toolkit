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
hla_elm_toolkit.elm.welm
==========================

Weighted ELM (WELM), following Zong et al. (article ref. [29]) as
described in article Section 2.6: a diagonal sample-weight matrix W is
introduced into the ELM least-squares objective,
beta = (H^T W H + I/C)^-1 H^T W T, with each training sample weighted
inversely to the observed frequency of its target high-resolution allele
class. Addresses the allele-frequency imbalance intrinsic to HLA data
(Figure 3; article Tables 11-12 show common diplotypes upgraded far more
reliably than rare ones under the unweighted base ELM).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data import Diplotype, Genotype, ReferencePopulation, compatible_diplotypes
from .base_elm import InputEncoder, OutputCatalog, _sigmoid


@dataclass
class WELM:
    hidden_size: int
    reg_c: float = 1.0
    seed: int = 0

    encoder: Optional[InputEncoder] = None
    catalog: Optional[OutputCatalog] = None
    W_hidden: Optional[np.ndarray] = None
    b_hidden: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None

    def _init_hidden_layer(self, n_inputs: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W_hidden = rng.uniform(-1, 1, size=(n_inputs, self.hidden_size))
        self.b_hidden = rng.uniform(-1, 1, size=(self.hidden_size,))

    def _hidden_activation(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.W_hidden + self.b_hidden)

    def fit(
        self,
        genotypes: Sequence[Genotype],
        ref: ReferencePopulation,
        max_diplotypes: Optional[int] = 4000,
        rng_seed: Optional[int] = None,
    ) -> "WELM":
        self.encoder = InputEncoder.fit(ref)
        self.catalog = OutputCatalog.from_reference(ref, max_diplotypes=max_diplotypes)
        self._init_hidden_layer(self.encoder.n_inputs)
        rng = np.random.default_rng(rng_seed if rng_seed is not None else self.seed)

        X_rows: List[np.ndarray] = []
        T_rows: List[np.ndarray] = []
        targets: List[Diplotype] = []
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
            targets.append(target)

        if not X_rows:
            raise ValueError("No training genotype had a compatible diplotype in the output catalog.")

        X = np.vstack(X_rows)
        T = np.vstack(T_rows)
        H = self._hidden_activation(X)

        # Sample weights inversely proportional to target-diplotype frequency.
        freq = Counter(targets)
        n = len(targets)
        w = np.array([n / (len(freq) * freq[t]) for t in targets], dtype=np.float64)
        w = w / w.mean()  # normalize so the effective sample count is unchanged
        W = np.diag(w)

        I = np.eye(H.shape[1])
        self.beta = np.linalg.solve(H.T @ W @ H + I / self.reg_c, H.T @ W @ T)
        return self

    def predict_scores(self, genotype: Genotype, ref: ReferencePopulation) -> np.ndarray:
        x = self.encoder.encode(genotype, ref)
        h = self._hidden_activation(x.reshape(1, -1))
        return (h @ self.beta).ravel()

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
