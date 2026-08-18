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
hla_elm_toolkit.baselines.gbt_baseline
=========================================

Gradient-boosted tree ensemble comparator, following article Section 2.7:
"a gradient-boosted tree ensemble [33,34], as a representative of the
tree-ensemble family [35], trained per locus using the same input
representation as the ELM models." Used in the framework-level
benchmarking of Sections 3.2-3.4 (Table 7) and computational-efficiency
comparison (Section 3.4, Table 8).

Requires scikit-learn (see requirements.txt / README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data import Genotype, ReferencePopulation
from ..elm.base_elm import InputEncoder

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "GradientBoostedTreesBaseline requires scikit-learn. Install with `pip install scikit-learn`."
    ) from exc


@dataclass
class GradientBoostedTreesBaseline:
    """Per-locus gradient-boosted tree classifier over the shared multi-hot input encoding."""

    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.1
    seed: int = 0

    encoder: Optional[InputEncoder] = None
    target_locus: Optional[str] = None
    classes_: List[str] = field(default_factory=list)
    model: Optional[GradientBoostingClassifier] = None

    def fit(
        self,
        genotypes: Sequence[Genotype],
        ref: ReferencePopulation,
        target_locus: str,
    ) -> "GradientBoostedTreesBaseline":
        self.encoder = InputEncoder.fit(ref)
        self.target_locus = target_locus

        X_rows: List[np.ndarray] = []
        y_rows: List[str] = []
        for g in genotypes:
            truth = (g.truth or {}).get(target_locus)
            if truth is None:
                continue
            X_rows.append(self.encoder.encode(g, ref))
            y_rows.append(truth[0])

        if not X_rows:
            raise ValueError(f"No genotype had confirmatory truth at locus {target_locus}.")

        X = np.vstack(X_rows)
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )
        self.model.fit(X, y_rows)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_top1(self, genotype: Genotype, ref: ReferencePopulation) -> Tuple[Optional[str], Optional[float]]:
        x = self.encoder.encode(genotype, ref).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return self.classes_[idx], float(probs[idx])
