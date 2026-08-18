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
hla_elm_toolkit.elm.base_elm
==============================

Base Extreme Learning Machine (ELM) for HLA resolution upgrading and
missing-locus imputation, following Algorithms 1 and 2 of the article
(Sections 2.4-2.5).

Implementation note (read before using for anything beyond demonstration
or teaching): the article's network uses a *domain-structured* topology
(input nodes -> HL-1 three-locus partial haplotypes -> HL-2 five-locus
partial haplotypes -> HL-3 complete haplotypes -> output diplotypes),
with connectivity fixed in advance by which partial haplotypes are
supersets of which (Section 2.4). Faithfully reproducing that exact
multi-stage topology is out of scope for this reference package. This
module instead implements the same *core ELM mathematics* the article
describes -- randomly generated, untrained input-to-hidden weights and a
closed-form, Moore-Penrose-pseudo-inverse solution for the hidden-to-
output weights (Figure 1; Algorithm 1, lines 7 and 18) -- over a single
random hidden layer applied to a flat multi-hot locus/allele input
encoding, rather than the exact HL-1/HL-2/HL-3 hierarchy. This preserves
the algorithm's defining property (no iterative back-propagation) and is
sufficient for benchmarking against the other methods in this package,
but is not a byte-for-byte reproduction of the original implementation
(which remains withheld pending the associated doctoral thesis defense;
see the article's Data Availability Statement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data import Diplotype, Genotype, Haplotype, ReferencePopulation, compatible_diplotypes


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


@dataclass
class InputEncoder:
    """
    Multi-hot encoding of a genotype into the ELM input layer: one input
    node per (locus, allele) pair observed anywhere in the reference
    population (Section 2.4: "input-layer nodes correspond to observed
    alleles/genotype fragments"). A missing locus activates every
    reference allele at that locus (missing-locus imputation, Section 2.2).
    """

    loci: Tuple[str, ...]
    allele_index: Dict[Tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def fit(cls, ref: ReferencePopulation) -> "InputEncoder":
        enc = cls(loci=ref.loci)
        idx = 0
        for locus in ref.loci:
            for allele in ref.alleles_at(locus):
                enc.allele_index[(locus, allele)] = idx
                idx += 1
        return enc

    @property
    def n_inputs(self) -> int:
        return len(self.allele_index)

    def encode(self, genotype: Genotype, ref: ReferencePopulation) -> np.ndarray:
        x = np.zeros(self.n_inputs, dtype=np.float64)
        for locus in self.loci:
            alleles = genotype.calls.get(locus)
            if alleles is None:
                alleles = ref.alleles_at(locus)  # untyped -> all reference alleles active
            for a in alleles:
                key = (locus, a)
                if key in self.allele_index:
                    x[self.allele_index[key]] = 1.0
        return x


@dataclass
class OutputCatalog:
    """Maps candidate diplotypes (built from the reference haplotype table) to output-node indices."""

    diplotypes: List[Diplotype]
    index: Dict[Diplotype, int] = field(default_factory=dict)

    @classmethod
    def from_reference(cls, ref: ReferencePopulation, max_diplotypes: Optional[int] = None) -> "OutputCatalog":
        haps = list(ref.haplotype_freq.keys())
        dips: List[Diplotype] = []
        n = len(haps)
        for i in range(n):
            for j in range(i, n):
                dips.append((haps[i], haps[j]))
                if max_diplotypes is not None and len(dips) >= max_diplotypes:
                    break
            if max_diplotypes is not None and len(dips) >= max_diplotypes:
                break
        cat = cls(diplotypes=dips)
        cat.index = {dt: i for i, dt in enumerate(dips)}
        return cat

    @property
    def n_outputs(self) -> int:
        return len(self.diplotypes)


@dataclass
class BaseELM:
    """
    Base Extreme Learning Machine: random, untrained input-to-hidden
    weights/biases; closed-form hidden-to-output weights via the
    (ridge-regularized) Moore-Penrose pseudo-inverse (Algorithm 1, line 18;
    Figure 1).
    """

    hidden_size: int
    reg_lambda: float = 1.0
    seed: int = 0

    encoder: Optional[InputEncoder] = None
    catalog: Optional[OutputCatalog] = None
    W: Optional[np.ndarray] = None     # input-to-hidden weights (random, fixed)
    b: Optional[np.ndarray] = None     # hidden biases (random, fixed)
    beta: Optional[np.ndarray] = None  # hidden-to-output weights (solved analytically)

    def _init_hidden_layer(self, n_inputs: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W = rng.uniform(-1, 1, size=(n_inputs, self.hidden_size))
        self.b = rng.uniform(-1, 1, size=(self.hidden_size,))

    def _hidden_activation(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.W + self.b)

    def fit(
        self,
        genotypes: Sequence[Genotype],
        ref: ReferencePopulation,
        max_diplotypes: Optional[int] = 4000,
        rng_seed: Optional[int] = None,
    ) -> "BaseELM":
        """
        Train following Algorithm 1: for each training genotype, sample a
        target diplotype uniformly at random from the diplotypes
        compatible with that (possibly ambiguous) genotype (line 13; the
        uniform-prior assumption is discussed as an open methodological
        question in article Section 2.5/Section 5), accumulate the
        hidden-layer output matrix H and one-hot target matrix, then solve
        beta in closed form.
        """
        self.encoder = InputEncoder.fit(ref)
        self.catalog = OutputCatalog.from_reference(ref, max_diplotypes=max_diplotypes)
        self._init_hidden_layer(self.encoder.n_inputs)

        rng = np.random.default_rng(rng_seed if rng_seed is not None else self.seed)

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

        X = np.vstack(X_rows)
        T = np.vstack(T_rows)
        H = self._hidden_activation(X)

        # beta = (H^T H + I/lambda)^-1 H^T T  (Algorithm 1, line 18)
        I = np.eye(H.shape[1])
        self.beta = np.linalg.solve(H.T @ H + I / self.reg_lambda, H.T @ T)
        return self

    def predict_scores(self, genotype: Genotype, ref: ReferencePopulation) -> np.ndarray:
        """Raw output-layer scores for every candidate diplotype (Algorithm 2, line 6)."""
        x = self.encoder.encode(genotype, ref)
        h = self._hidden_activation(x.reshape(1, -1))
        return (h @ self.beta).ravel()

    def predict_ranked(
        self, genotype: Genotype, ref: ReferencePopulation, top_k: int = 5
    ) -> List[Tuple[Diplotype, float]]:
        """
        Rank candidate diplotypes by output score restricted to those
        actually compatible with this genotype's activated input nodes
        (Algorithm 2, lines 4-11), and convert scores to a probability-like
        distribution via a softmax over the compatible subset for
        readability (the article reports Post-P from the Hardy-Weinberg
        likelihood for the EM-style comparators; the base ELM's own
        analog is the relative rank / normalized output score used here).
        """
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

    def predict_top1(
        self, genotype: Genotype, ref: ReferencePopulation, theta: float = 0.0
    ) -> Tuple[Optional[Diplotype], Optional[float]]:
        """Top-1 call with a Post-P-style confidence threshold (Algorithm 2, lines 12-15)."""
        ranked = self.predict_ranked(genotype, ref, top_k=1)
        if not ranked:
            return None, None
        dt, p = ranked[0]
        if p >= theta:
            return dt, p
        return None, p
