# hla-elm-toolkit

Reference Python implementations of **every method compared** in:

> Kepentzis, S.; Chatzistamatiou, T.; Digalakis, J.; Petropoulou, O.;
> Matsopoulos, G.K.; Koutsouris, D. *Development of an Extreme Learning
> Machine approach to upgrade low/mid to high resolution HLA data
> improving the usability of donor data from registries.* Genes
> (submitted).

This package implements the base Extreme Learning Machine (ELM) and its
three extensions (KELM, WELM, Ensemble ELM), the three EM/Bayesian
comparator baselines the article specifies in pseudocode (GRIMM-style,
HaploStats-style, and Hapl-o-Mat-style; Section 2.7.1, Algorithms 3-5),
and the MLP / gradient-boosted-tree comparators used in the article's
framework-level benchmarking (Section 2.7).

**Note on the requested Python version.** The request that accompanied
this package asked for "Python 3.2". Python 3.2 reached end-of-life in
2016 and lacks language features (e.g. f-strings, `dataclasses`) that
this codebase — and most modern scientific Python — relies on. The
article itself states its own implementation ran on **Python 3.12**
(Section 2.8, "Titanas" server specification), so this package targets
**Python 3.9+** (developed and tested on 3.12) on the assumption that
"3.2" was a typo. If Python 3.2 compatibility is genuinely required,
please say so and the package can be restructured accordingly — it would
require removing `dataclasses`, f-strings, and `pathlib`/typing-module
usage throughout, and pinning much older `numpy`/`scikit-learn` releases
(if any compatible wheel still exists for those library versions).

---

## 1. Scope and honesty about limitations

This is a **from-scratch reference reimplementation** written to
accompany the article's revision process, not the article's original,
registry-scale codebase. That original implementation has since been
released as open-source at the repository of the NTUA Biomedical
Engineering Laboratory:
https://biomedntuagr.sharepoint.com/:f:/g/IgCgDy6zAJ8HTYmovHlYxrQXAeRFyCh7S8HutOUI6ddvDLg?e=0bO3bG
(see the article's Data Availability Statement). If you need the exact
original implementation for reproducibility review, use that repository
rather than this package. Three things follow from the fact that this
remains a separate, simplified reimplementation:

1. **No real registry data is included or reachable.** The HTO, ORAM,
   and GRPT donor registries used in the article are not distributable
   (donor privacy; Hellenic Transplant Organization data-sharing
   restrictions). `hla_elm_toolkit.data.make_synthetic_population()`
   generates a small synthetic reference population with a Zipf-like,
   rare-allele-dominated frequency spectrum (loosely mirroring the
   pattern quantified in the article's Table 5) purely for
   demonstration and unit testing. **Do not** interpret any
   accuracy/call-rate number produced by this package's demo or tests as
   comparable to the article's reported figures.

2. **The base ELM's domain-structured topology is simplified.** The
   article's network activates input nodes for observed
   alleles/genotype fragments, then propagates through three
   hierarchical hidden layers (HL-1: three-locus partial haplotypes,
   HL-2: five-locus partial haplotypes, HL-3: complete haplotypes) whose
   connectivity is fixed in advance by which partial haplotypes are
   supersets of which (Section 2.4). Reproducing that exact multi-stage
   topology from the article's prose description alone, without the
   original code, is out of scope for this package. `elm/base_elm.py`
   instead implements the same *core ELM mathematics* — randomly
   generated, untrained input-to-hidden weights and a closed-form
   Moore-Penrose-pseudo-inverse solution for the output weights — over a
   **single random hidden layer** applied to a flat multi-hot
   locus/allele input encoding. This preserves the algorithm's defining
   property (no iterative back-propagation) and is sufficient to
   benchmark KELM/WELM/Ensemble/comparators against each other in a
   like-for-like way, but it is not a byte-for-byte reproduction of the
   original network, and its accuracy is not expected to match the
   article's reported figures.

3. **GRIMM-, HaploStats-, and Hapl-o-Mat-style baselines are the
   article's own documented simplifications**, not the published tools.
   The article is explicit about this (Section 2.7.1): "these pseudocode
   summaries omit implementation-specific details (e.g., GRIMM's exact
   graph construction and traversal optimizations, HaploStats'
   race/ethnicity-specific reference tables, and Hapl-o-Mat's
   population-weighting options) ... based on their published method
   descriptions ... rather than on proprietary source code, which we did
   not have access to for any of the three tools." This package
   implements exactly those pseudocode algorithms (Algorithms 3-5) as
   given in the article, faithfully, but they remain approximations of
   the real tools by the article's own account, and running the
   published Hapl-o-Mat tool or the NMDP HaploStats web service itself
   is explicitly listed as outstanding future work in the article
   (Sections 3.10-3.11, 5).

If you need the exact original implementation for reproducibility
review, it is not available through this package; see the article's Data
Availability Statement for the release plan.

---

## 2. What corresponds to what

| Article section / algorithm | Module |
|---|---|
| Section 2.2 (resolution levels, ambiguity, missing-locus handling) | `hla_elm_toolkit/data.py` |
| Section 2.3 (accuracy, call rate, posterior probability, top-k) | `hla_elm_toolkit/metrics.py` |
| Section 2.8 (95% CI via Wilson score, chosen over naive bootstrap) | `hla_elm_toolkit/metrics.py::wilson_score_interval` |
| Section 2.4-2.5, Algorithm 1 (ELM training), Algorithm 2 (inference) | `hla_elm_toolkit/elm/base_elm.py` |
| Section 2.6, Kernel ELM (KELM) | `hla_elm_toolkit/elm/kelm.py` |
| Section 2.6, Weighted ELM (WELM) | `hla_elm_toolkit/elm/welm.py` |
| Section 2.6, Ensemble ELM | `hla_elm_toolkit/elm/ensemble_elm.py` |
| Section 2.7, MLP comparator (Table 7) | `hla_elm_toolkit/baselines/mlp_baseline.py` |
| Section 2.7, gradient-boosted-tree comparator (Table 7) | `hla_elm_toolkit/baselines/gbt_baseline.py` |
| Section 2.7.1, Algorithm 3 (GRIMM-style, graph-based) | `hla_elm_toolkit/baselines/grimm_style.py` |
| Section 2.7.1, Algorithm 4 (HaploStats-style) | `hla_elm_toolkit/baselines/haplostats_style.py` |
| Section 2.7.1, Algorithm 5 (Hapl-o-Mat-style EM) | `hla_elm_toolkit/baselines/haplo_em.py` |
| Section 3.7 (two-locus in-registry EM baseline) | `HaploEM(loci=("A","B"))` — same class, different `loci` argument |
| Section 3.11 (five-locus in-house EM baseline) | `HaploEM(loci=("A","B","C","DRB1","DQB1"))` — same class |

---

## 3. Installation

```bash
# From the extracted archive directory:
cd hla_elm_toolkit

# Option A: install as a package (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # base install (numpy only)
pip install -e ".[all]"            # + scikit-learn (MLP/GBT) + pytest

# Option B: no installation, just add the folder to PYTHONPATH
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## 4. Quick start

```python
from hla_elm_toolkit.data import make_synthetic_population
from hla_elm_toolkit.elm import BaseELM
from hla_elm_toolkit.metrics import summarize, format_summary, PredictionResult

# Synthetic reference population (NOT real registry data -- see Section 1 above)
ref = make_synthetic_population(n_donors=1000, n_haplotypes=40, seed=0)
train, test = ref.donors[:800], ref.donors[800:]

model = BaseELM(hidden_size=150, seed=0)
model.fit(train, ref, max_diplotypes=3000)

results = []
for g in test:
    dt, post_p = model.predict_top1(g, ref, theta=0.0)
    called = dt is not None
    correct = None  # compare dt against g.truth yourself, per-locus or jointly
    results.append(PredictionResult(donor_id=g.donor_id, called=called, correct=correct))

print(format_summary("Base ELM", summarize(results)))
```

A complete, runnable, end-to-end comparison of **every** method in the
article (ELM, KELM, WELM, Ensemble ELM, HaploStats-style, Hapl-o-Mat-style
EM, GRIMM-style) is provided in:

```bash
python examples/run_demo_comparison.py --n-donors 1500 --seed 0
```

which prints a per-method summary table in the same
`value [95% CI]` style used throughout the article's tables (e.g. Table 16).

---

## 5. Running the tests

```bash
python tests/test_basic.py     # standalone, no pytest required
# or
pytest tests/                  # if pytest is installed
```

The test suite exercises every method on a small synthetic population and
checks basic invariants (frequencies sum to 1, CIs are well-formed,
predictions are in range, etc.); it does not — and cannot, without the
real registry data — check numerical agreement with the article's
reported accuracy/call-rate figures.

---

## 6. Package layout

```
hla_elm_toolkit/
├── README.md                       (this file)
├── requirements.txt
├── setup.py
├── hla_elm_toolkit/
│   ├── __init__.py
│   ├── data.py                     genotype/haplotype/diplotype types,
│   │                                synthetic reference-population generator
│   ├── metrics.py                  accuracy, call rate, Wilson score 95% CI
│   ├── elm/
│   │   ├── __init__.py
│   │   ├── base_elm.py             Algorithm 1 (training) + Algorithm 2 (inference)
│   │   ├── kelm.py                 Kernel ELM (Section 2.6)
│   │   ├── welm.py                 Weighted ELM (Section 2.6)
│   │   └── ensemble_elm.py         Ensemble ELM (Section 2.6)
│   └── baselines/
│       ├── __init__.py
│       ├── haplo_em.py             Algorithm 5 (Hapl-o-Mat-style EM; used for
│       │                            both the Section 3.7 two-locus and
│       │                            Section 3.11 five-locus baselines)
│       ├── haplostats_style.py     Algorithm 4 (HaploStats-style)
│       ├── grimm_style.py          Algorithm 3 (GRIMM-style, graph-based)
│       ├── mlp_baseline.py         MLP comparator (Section 2.7; requires scikit-learn)
│       └── gbt_baseline.py         Gradient-boosted-tree comparator (Section 2.7; requires scikit-learn)
├── examples/
│   └── run_demo_comparison.py      end-to-end demo running every method
└── tests/
    └── test_basic.py
```

---

## 7. Citation

If you use this code, please cite the article above. Please also note,
per the article's own acknowledgment, that the published Hapl-o-Mat tool
and the NMDP HaploStats web service were not run directly in the
article's evaluation (no compatible batch interface was available in the
authors' environment); this package's `haplostats_style.py` and
`haplo_em.py` are the article's own open, reproducible approximations of
those tools' core algorithms, not the tools themselves.

## 8. Article gap-closing scripts (Γ.1, Β.2)

Two additional scripts, added after the article's supervisory review
round, are provided to help close two of the manuscript's flagged
critical gaps without requiring a full new experiment:

* **`examples/recompute_first_field_accuracy.py`** (article gap **Γ.1**):
  GRIMM, hlaR ImputeHaplo, and the in-house EM baseline (Sections
  3.8-3.11) are scored at first-field resolution, while the ELM
  framework's headline accuracy is reported at two-field resolution --
  making Tables 16-19 not strictly like-for-like. This script takes the
  ELM's *already-computed* two-field predictions (exported to a simple
  CSV; see the script's docstring for the exact format) and truncates
  both predictions and truth to first-field before rescoring, so the
  same "value \[95% CI\]" figure can be added to the "ELM (reported,
  overall)" row of each affected table. No new model training or
  inference is required -- only re-scoring of existing predictions.

* **`BETA2_DIAGNOSTIC_CHECKLIST.md`** (article gap **Β.2** -- **now
  resolved**): Section 3.11's in-house Hapl-o-Mat-style baseline
  previously reported an identical evaluable sample size (n=45) and
  joint accuracy (100.0%) across three very differently sized cohorts
  (GRPT, ORAM, HTO). This was traced to a copy-paste transcription
  error made while writing up Section 3.11 (correct counts: n=45 GRPT,
  n=38 ORAM, n=29 HTO; the article has been corrected), not a code
  bug. The checklist is kept as a general step-by-step reference for
  auditing suspiciously identical figures across differently sized
  cohorts.

## 9. License

This reference toolkit (a separate, simplified reimplementation; see
Section 1) is not yet assigned a license; treat it as "all rights
reserved, provided for internal review purposes" until the authors
specify otherwise. The article's original, registry-scale implementation
is a separate codebase, now released as open-source at
https://biomedntuagr.sharepoint.com/:f:/g/IgCgDy6zAJ8HTYmovHlYxrQXAeRFyCh7S8HutOUI6ddvDLg?e=0bO3bG;
consult that repository directly for its license terms.
