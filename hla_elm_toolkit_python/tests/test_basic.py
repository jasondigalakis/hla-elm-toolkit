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
Basic smoke tests for hla_elm_toolkit. Run with:  pytest tests/
(or, without pytest installed:  python tests/test_basic.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hla_elm_toolkit.data import make_synthetic_population, compatible_diplotypes
from hla_elm_toolkit.metrics import wilson_score_interval, PredictionResult, summarize
from hla_elm_toolkit.elm import BaseELM, KELM, WELM, EnsembleELM
from hla_elm_toolkit.baselines import HaploEM, HaploStatsStyleImputer, GrimmStyleGraph, GrimmStyleImputer


def _small_population(seed=0):
    return make_synthetic_population(n_donors=200, n_haplotypes=20, n_alleles_per_locus=6, seed=seed)


def test_synthetic_population_shape():
    ref = _small_population()
    assert len(ref.donors) == 200
    assert len(ref.haplotype_freq) == 20
    total_freq = sum(ref.haplotype_freq.values())
    assert abs(total_freq - 1.0) < 1e-9


def test_compatible_diplotypes_nonempty_for_most_donors():
    ref = _small_population()
    n_with_candidates = sum(1 for g in ref.donors if compatible_diplotypes(g, ref))
    assert n_with_candidates > len(ref.donors) * 0.5


def test_wilson_score_interval_bounds():
    lo, hi = wilson_score_interval(45, 45)  # 100% success, small n
    assert 0.0 < lo < 1.0
    assert hi == 1.0  # upper bound at n successes / n trials is exactly 1.0
    lo2, hi2 = wilson_score_interval(0, 2)  # 0% success, tiny n
    assert lo2 == 0.0
    assert hi2 < 1.0


def test_wilson_score_interval_invalid_inputs():
    try:
        wilson_score_interval(5, 0)
        assert False, "expected ValueError for n=0"
    except ValueError:
        pass
    try:
        wilson_score_interval(-1, 10)
        assert False, "expected ValueError for negative successes"
    except ValueError:
        pass


def test_summarize_and_format():
    results = [
        PredictionResult(donor_id=f"d{i}", called=(i % 3 != 0), correct=(i % 2 == 0))
        for i in range(30)
    ]
    s = summarize(results)
    assert 0.0 <= s["call_rate"] <= 1.0
    assert "call_rate_ci" in s


def test_base_elm_trains_and_predicts():
    ref = _small_population()
    train = ref.donors[:150]
    test = ref.donors[150:]
    model = BaseELM(hidden_size=40, seed=0)
    model.fit(train, ref, max_diplotypes=500)
    n_called = 0
    for g in test:
        dt, p = model.predict_top1(g, ref, theta=0.0)
        if dt is not None:
            n_called += 1
    assert n_called > 0


def test_kelm_trains_and_predicts():
    ref = _small_population()
    train = ref.donors[:150]
    test = ref.donors[150:]
    model = KELM(C=1.0, gamma=0.05)
    model.fit(train, ref, max_diplotypes=500)
    ranked = model.predict_ranked(test[0], ref, top_k=3)
    assert isinstance(ranked, list)


def test_welm_trains_and_predicts():
    ref = _small_population()
    train = ref.donors[:150]
    model = WELM(hidden_size=40, seed=0)
    model.fit(train, ref, max_diplotypes=500)
    ranked = model.predict_ranked(train[0], ref, top_k=1)
    assert isinstance(ranked, list)


def test_ensemble_elm_trains_and_predicts():
    ref = _small_population()
    train = ref.donors[:150]
    model = EnsembleELM(n_members=3, hidden_size=30)
    model.fit(train, ref, max_diplotypes=500)
    dt, p = model.predict_top1(train[0], ref)
    # not asserting dt is not None: with only 3 members and a small
    # reference population a NO_CALL is possible; just check no crash
    assert p is None or 0.0 <= p <= 1.0


def test_haplostats_style_imputer():
    ref = _small_population()
    imputer = HaploStatsStyleImputer(reference=ref)
    dt, p = imputer.top1_call(ref.donors[0])
    assert p is None or 0.0 <= p <= 1.0


def test_haplo_em_converges_and_scores():
    ref = _small_population()
    train = ref.donors[:150]
    em = HaploEM(loci=ref.loci, max_iter=30)
    em.fit(train, ref)
    assert em.n_iterations_run > 0
    assert abs(sum(em.haplotype_freq.values()) - 1.0) < 1e-6
    dt, p = em.score_genotype(train[0])
    assert p is None or 0.0 <= p <= 1.0


def test_grimm_style_imputer():
    ref = _small_population()
    graph = GrimmStyleGraph.from_reference(ref)
    imputer = GrimmStyleImputer(graph=graph)
    status, dt, like = imputer.impute(ref.donors[0])
    assert status in ("imputed", "no_match_in_reference_panel")


def test_mlp_and_gbt_baselines_optional():
    """Skipped automatically if scikit-learn is not installed."""
    try:
        from hla_elm_toolkit.baselines.mlp_baseline import MLPBaseline
        from hla_elm_toolkit.baselines.gbt_baseline import GradientBoostedTreesBaseline
    except ImportError:
        print("  (scikit-learn not installed; skipping MLP/GBT test)")
        return

    ref = _small_population()
    train, test = ref.donors[:150], ref.donors[150:]

    mlp = MLPBaseline(hidden_size=30, max_iter=150)
    mlp.fit(train, ref, target_locus=ref.loci[0])
    pred, prob = mlp.predict_top1(test[0], ref)
    assert prob is None or 0.0 <= prob <= 1.0

    gbt = GradientBoostedTreesBaseline(n_estimators=30)
    gbt.fit(train, ref, target_locus=ref.loci[0])
    pred2, prob2 = gbt.predict_top1(test[0], ref)
    assert prob2 is None or 0.0 <= prob2 <= 1.0


if __name__ == "__main__":
    # Allow running without pytest: execute every test_* function and report.
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
