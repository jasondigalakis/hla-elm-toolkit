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

#!/usr/bin/env python3
"""
run_demo_comparison.py
=======================

End-to-end demonstration: builds a small synthetic reference population
(see hla_elm_toolkit.data.make_synthetic_population -- NOT the real
HTO/ORAM/GRPT registry data used in the article, which is not
distributable), trains every method compared in the article on a
training split, evaluates each on a held-out split, and prints a
summary table in the same "value [95% CI]" style used throughout the
article's tables.

Run with:  python examples/run_demo_comparison.py
Optional:  python examples/run_demo_comparison.py --n-donors 3000 --seed 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hla_elm_toolkit.data import make_synthetic_population, Genotype
from hla_elm_toolkit.metrics import PredictionResult, summarize, format_summary
from hla_elm_toolkit.elm import BaseELM, KELM, WELM, EnsembleELM
from hla_elm_toolkit.baselines import HaploEM, HaploStatsStyleImputer, GrimmStyleGraph, GrimmStyleImputer


def train_test_split(genotypes, test_fraction=0.2, seed=0):
    import random
    rng = random.Random(seed)
    idx = list(range(len(genotypes)))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_fraction)
    test_idx = set(idx[:n_test])
    train = [g for i, g in enumerate(genotypes) if i not in test_idx]
    test = [g for i, g in enumerate(genotypes) if i in test_idx]
    return train, test


def evaluate_top1(model, predict_fn, test_set, ref, theta=0.0):
    results = []
    for g in test_set:
        dt, p = predict_fn(model, g, ref, theta)
        called = dt is not None
        correct = None
        if called and g.truth is not None:
            truth_haps = tuple(g.truth[locus] for locus in ref.loci)
            # g.truth stores (allele1, allele2) per locus; build per-locus truth
            # comparison instead of exact multi-locus haplotype match, since the
            # synthetic generator's ambiguity model does not guarantee the
            # predicted haplotype pair is drawn from the same phase as truth.
            correct = _matches_truth(dt, g, ref)
        results.append(PredictionResult(donor_id=g.donor_id, called=called, correct=correct, post_p=p))
    return results


def _matches_truth(dt, genotype: Genotype, ref) -> bool:
    h1, h2 = dt
    pred_pairs = {locus: frozenset((h1[i], h2[i])) for i, locus in enumerate(ref.loci)}
    truth = genotype.truth or {}
    for locus in ref.loci:
        if locus not in truth:
            continue
        if frozenset(truth[locus]) != pred_pairs.get(locus):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-donors", type=int, default=1500)
    parser.add_argument("--n-haplotypes", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Building synthetic reference population (n_donors={args.n_donors}, "
          f"n_haplotypes={args.n_haplotypes}, seed={args.seed})...")
    ref = make_synthetic_population(
        n_donors=args.n_donors, n_haplotypes=args.n_haplotypes, seed=args.seed
    )
    train, test = train_test_split(ref.donors, test_fraction=0.2, seed=args.seed)
    print(f"Train: {len(train)} donors | Test: {len(test)} donors | "
          f"Reference haplotypes: {len(ref.haplotype_freq)}\n")

    max_dt = 3000

    # ---- Base ELM ----
    t0 = time.time()
    elm = BaseELM(hidden_size=args.hidden_size, seed=args.seed)
    elm.fit(train, ref, max_diplotypes=max_dt, rng_seed=args.seed)
    elm_results = evaluate_top1(
        elm, lambda m, g, r, th: m.predict_top1(g, r, theta=th), test, ref
    )
    print(f"[Base ELM]      trained in {time.time()-t0:.2f}s")
    print("  " + format_summary("Base ELM", summarize(elm_results)))

    # ---- KELM ----
    t0 = time.time()
    kelm = KELM(C=1.0, gamma=0.05)
    kelm.fit(train, ref, max_diplotypes=max_dt, rng_seed=args.seed)
    kelm_results = evaluate_top1(
        kelm,
        lambda m, g, r, th: (m.predict_ranked(g, r, top_k=1)[0] if m.predict_ranked(g, r, top_k=1) else (None, None)),
        test, ref,
    )
    print(f"[KELM]          trained in {time.time()-t0:.2f}s")
    print("  " + format_summary("KELM", summarize(kelm_results)))

    # ---- WELM ----
    t0 = time.time()
    welm = WELM(hidden_size=args.hidden_size, seed=args.seed)
    welm.fit(train, ref, max_diplotypes=max_dt, rng_seed=args.seed)
    welm_results = evaluate_top1(
        welm,
        lambda m, g, r, th: (m.predict_ranked(g, r, top_k=1)[0] if m.predict_ranked(g, r, top_k=1) else (None, None)),
        test, ref,
    )
    print(f"[WELM]          trained in {time.time()-t0:.2f}s")
    print("  " + format_summary("WELM", summarize(welm_results)))

    # ---- Ensemble ELM ----
    t0 = time.time()
    ens = EnsembleELM(n_members=5, hidden_size=args.hidden_size)
    ens.fit(train, ref, max_diplotypes=max_dt, rng_seed=args.seed)
    ens_results = evaluate_top1(
        ens, lambda m, g, r, th: m.predict_top1(g, r, theta=th), test, ref
    )
    print(f"[Ensemble ELM]  trained in {time.time()-t0:.2f}s")
    print("  " + format_summary("Ensemble ELM", summarize(ens_results)))

    # ---- HaploStats-style (using the TRUE reference frequencies, as the
    #      article's Algorithm 4 assumes a pre-estimated table) ----
    t0 = time.time()
    hs = HaploStatsStyleImputer(reference=ref)
    hs_results = evaluate_top1(
        hs, lambda m, g, r, th: m.top1_call(g, theta=th), test, ref
    )
    print(f"[HaploStats-style] scored in {time.time()-t0:.2f}s")
    print("  " + format_summary("HaploStats-style", summarize(hs_results)))

    # ---- Hapl-o-Mat-style EM (estimates its own frequencies from `train`) ----
    t0 = time.time()
    em = HaploEM(loci=ref.loci, max_iter=50)
    em.fit(train, ref)
    print(f"[Hapl-o-Mat-style EM] converged in {em.n_iterations_run} iterations, "
          f"{time.time()-t0:.2f}s")
    em_results = evaluate_top1(
        em, lambda m, g, r, th: m.score_genotype(g, theta=th), test, ref
    )
    print("  " + format_summary("Hapl-o-Mat-style EM", summarize(em_results)))

    # ---- GRIMM-style ----
    t0 = time.time()
    graph = GrimmStyleGraph.from_reference(ref)
    grimm = GrimmStyleImputer(graph=graph)
    grimm_results = []
    for g in test:
        status, dt, like = grimm.impute(g)
        called = dt is not None
        correct = _matches_truth(dt, g, ref) if called else None
        grimm_results.append(PredictionResult(donor_id=g.donor_id, called=called, correct=correct))
    print(f"[GRIMM-style]   scored in {time.time()-t0:.2f}s")
    print("  " + format_summary("GRIMM-style", summarize(grimm_results)))

    print("\nNote: this demo uses a small synthetic reference population for")
    print("illustration only; absolute accuracy/call-rate figures are not")
    print("comparable to the article's real-registry results (Sections 3.5,")
    print("3.7-3.11), which used the (non-distributable) HTO/ORAM/GRPT data.")


if __name__ == "__main__":
    main()
