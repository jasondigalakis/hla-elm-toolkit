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
hla_elm_toolkit
================

Reference implementations of every method compared in:

    Kepentzis, S. et al. "Development of an Extreme Learning Machine
    approach to upgrade low/mid to high resolution HLA data improving
    the usability of donor data from registries." (Genes, submitted).

Includes the base Extreme Learning Machine (ELM) and its three
extensions (KELM, WELM, Ensemble ELM; Sections 2.4-2.6), the three
EM/Bayesian comparator baselines described in pseudocode as Algorithms
3-5 (GRIMM-style, HaploStats-style, Hapl-o-Mat-style; Section 2.7.1),
and the MLP / gradient-boosted-tree comparators used in the article's
framework-level benchmarking (Section 2.7).

See the top-level README.md for scope, limitations, and how this
package's simplifications relate to the article's original,
registry-scale implementation (now open-source; see the article's
Data Availability Statement and README.md Section 1 for the repository
link).
"""

__version__ = "0.1.0"
