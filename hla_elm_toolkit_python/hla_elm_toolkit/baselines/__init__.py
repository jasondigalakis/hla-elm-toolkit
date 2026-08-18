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
Comparator methods benchmarked against the ELM family in the article:
GRIMM (Algorithm 3), HaploStats-style imputation (Algorithm 4),
Hapl-o-Mat-style EM estimation (Algorithm 5, used for both the two-locus
Section 3.7 baseline and the five-locus Section 3.11 baseline), plus the
MLP and gradient-boosted-tree comparators of Section 2.7.
"""

from .haplo_em import HaploEM
from .haplostats_style import HaploStatsStyleImputer
from .grimm_style import GrimmStyleGraph, GrimmStyleImputer, GRIMM_IMPUTED, GRIMM_NO_MATCH

__all__ = [
    "HaploEM",
    "HaploStatsStyleImputer",
    "GrimmStyleGraph",
    "GrimmStyleImputer",
    "GRIMM_IMPUTED",
    "GRIMM_NO_MATCH",
]

# MLPBaseline and GradientBoostedTreesBaseline are imported lazily by name
# (not into this __init__) since they require scikit-learn, which is an
# optional dependency (see requirements.txt); import them directly from
# hla_elm_toolkit.baselines.mlp_baseline / gbt_baseline.
