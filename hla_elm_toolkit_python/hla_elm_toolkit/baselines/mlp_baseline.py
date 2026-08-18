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
hla_elm_toolkit.baselines.mlp_baseline
=========================================

Multilayer perceptron (MLP) comparator, trained by back-propagation with
early stopping, as described in article Section 2.7: "a multilayer
perceptron (MLP) with one hidden layer of equivalent size trained by
back-propagation with early stopping [32]". Used in the framework-level
benchmarking of Sections 3.2-3.4 (Table 7).

Requires scikit-learn (see requirements.txt / README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data import Diplotype, Genotype, ReferencePopulation, compatible_diplotypes
from ..elm.base_elm import InputEncoder, OutputCatalog

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MLPBaseline requires scikit-learn. Install with `pip install scikit-learn`."
    ) from exc


@dataclass
class MLPBaseline:
    """
    Per-locus MLP: for a single target locus, predicts the most likely
    high-resolution allele class from the multi-hot input encoding used
    by the ELM family, allowing a like-for-like comparison (Table 7).
    """

    hidden_size: int = 200
    max_iter: int = 500
    seed: int = 0

    encoder: Optional[InputEncoder] = None
    target_locus: Optional[str] = None
    classes_: List[str] = field(default_factory=list)
    model: Optional[MLPClassifier] = None

    def fit(
        self,
        genotypes: Sequence[Genotype],
        ref: ReferencePopulation,
        target_locus: str,
    ) -> "MLPBaseline":
        self.encoder = InputEncoder.fit(ref)
        self.target_locus = target_locus

        X_rows: List[np.ndarray] = []
        y_rows: List[str] = []
        for g in genotypes:
            truth = (g.truth or {}).get(target_locus)
            if truth is None:
                continue
            # Use the first of the pair as the single-label training target
            # (a simplification; a production system would train one
            # multi-label or two-allele-slot model instead).
            X_rows.append(self.encoder.encode(g, ref))
            y_rows.append(truth[0])

        if not X_rows:
            raise ValueError(f"No genotype had confirmatory truth at locus {target_locus}.")

        X = np.vstack(X_rows)
        # Label-encode to integers: some scikit-learn/numpy version combinations
        # raise a TypeError inside early_stopping's internal validation-score
        # check when class labels are plain Python strings; integer-encoding
        # avoids that entirely and is otherwise equivalent.
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_rows)
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_size,),
            early_stopping=True,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model.fit(X, y_encoded)
        self.classes_ = list(le.inverse_transform(self.model.classes_))
        return self

    def predict_top1(self, genotype: Genotype, ref: ReferencePopulation) -> Tuple[Optional[str], Optional[float]]:
        x = self.encoder.encode(genotype, ref).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return self.classes_[idx], float(probs[idx])
