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

from setuptools import setup, find_packages

setup(
    name="hla-elm-toolkit",
    version="0.1.0",
    description=(
        "Reference implementations of the ELM family (base ELM, KELM, WELM, "
        "Ensemble ELM) and comparator methods (GRIMM-style, HaploStats-style, "
        "Hapl-o-Mat-style EM, MLP, gradient-boosted trees) benchmarked in "
        "Kepentzis et al., 'Development of an Extreme Learning Machine "
        "approach to upgrade low/mid to high resolution HLA data improving "
        "the usability of donor data from registries.'"
    ),
    author="Stavros Kepentzis et al.",
    python_requires=">=3.9",
    packages=find_packages(include=["hla_elm_toolkit", "hla_elm_toolkit.*"]),
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "trees": ["scikit-learn>=1.3"],
        "test": ["pytest>=7.0"],
        "all": ["scikit-learn>=1.3", "pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
