// -----------------------------------------------------------------------------
// This file is part of hla_elm_toolkit, open-source software developed to
// accompany:
//   Kepentzis, S.; Chatzistamatiou, T.; Digalakis, J.; Petropoulou, O.;
//   Matsopoulos, G.K.; Koutsouris, D. "Development of an Extreme Learning
//   Machine approach to upgrade low/mid to high resolution HLA data
//   improving the usability of donor data from registries." Genes (MDPI).
//
// Software authors: Stavros Kepentzis (skepenjis@biomed.ntua.gr)
//                    Jason Digalakis  (jdigalakis@biomed.ntua.gr)
// Affiliation: Biomedical Engineering Laboratory, School of Electrical and
//              Computer Engineering, National Technical University of
//              Athens (NTUA), Athens, Greece
//
// Released as open-source software.
// -----------------------------------------------------------------------------

// test_basic.cpp
// Basic smoke tests for the C++ port, mirroring tests/test_basic.py.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "hla_elm/BaseELM.hpp"
#include "hla_elm/Data.hpp"
#include "hla_elm/HaploEM.hpp"
#include "hla_elm/Matrix.hpp"
#include "hla_elm/Metrics.hpp"

using namespace hla_elm;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, name)                                        \
    do {                                                         \
        if (cond) { std::cout << "PASS  " << name << "\n"; ++g_pass; } \
        else { std::cout << "FAIL  " << name << "\n"; ++g_fail; }      \
    } while (0)

static ReferencePopulation small_population(std::uint64_t seed = 0) {
    return make_synthetic_population(200, kDefaultLoci, 6, 20, seed);
}

int main() {
    // Matrix basics.
    {
        Matrix a(2, 2, 0.0);
        a(0, 0) = 1; a(0, 1) = 2; a(1, 0) = 3; a(1, 1) = 4;
        Matrix b = Matrix::identity(2);
        Matrix c = a * b;
        CHECK(c(0, 0) == 1 && c(1, 1) == 4, "matrix_multiply_identity");

        Matrix rhs(2, 1, 0.0);
        rhs(0, 0) = 5; rhs(1, 0) = 6;
        Matrix x = a.solve(rhs);
        Matrix check = a * x;
        CHECK(std::fabs(check(0, 0) - 5) < 1e-6 && std::fabs(check(1, 0) - 6) < 1e-6, "matrix_solve_roundtrip");
    }

    // Synthetic population.
    {
        ReferencePopulation ref = small_population();
        CHECK(ref.donors.size() == 200, "synthetic_population_donor_count");
        CHECK(ref.haplotype_freq.size() == 20, "synthetic_population_haplotype_count");
        double total = 0.0;
        for (const auto& [h, f] : ref.haplotype_freq) { (void)h; total += f; }
        CHECK(std::fabs(total - 1.0) < 1e-9, "synthetic_population_freq_sums_to_1");
    }

    // compatible_diplotypes.
    {
        ReferencePopulation ref = small_population();
        int n_with_candidates = 0;
        for (const auto& g : ref.donors)
            if (!compatible_diplotypes(g, ref).empty()) ++n_with_candidates;
        CHECK(n_with_candidates > static_cast<int>(ref.donors.size()) * 0.5, "compatible_diplotypes_nonempty_for_most_donors");
    }

    // Wilson score interval.
    {
        auto [lo, hi] = wilson_score_interval(45, 45);
        CHECK(lo > 0.0 && lo < 1.0, "wilson_ci_lower_bound_in_range");
        CHECK(hi == 1.0, "wilson_ci_upper_bound_at_100pct");

        auto [lo2, hi2] = wilson_score_interval(0, 2);
        CHECK(lo2 == 0.0 && hi2 < 1.0, "wilson_ci_zero_successes");

        bool threw = false;
        try { wilson_score_interval(5, 0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "wilson_ci_rejects_zero_n");
    }

    // summarize / format_summary.
    {
        std::vector<PredictionResult> results;
        for (int i = 0; i < 30; ++i) {
            PredictionResult r;
            r.donor_id = "d" + std::to_string(i);
            r.called = (i % 3 != 0);
            if (r.called) r.correct = (i % 2 == 0);
            results.push_back(r);
        }
        Summary s = summarize(results);
        CHECK(s.call_rate >= 0.0 && s.call_rate <= 1.0, "summarize_call_rate_in_range");
        std::string formatted = format_summary("test", s);
        CHECK(!formatted.empty(), "format_summary_nonempty");
    }

    // BaseELM trains and predicts.
    {
        ReferencePopulation ref = small_population();
        std::vector<Genotype> train(ref.donors.begin(), ref.donors.begin() + 150);
        std::vector<Genotype> test(ref.donors.begin() + 150, ref.donors.end());

        BaseELM model(40, 1.0, 0);
        model.fit(train, ref, 500);
        int n_called = 0;
        for (const auto& g : test) {
            auto [dt, p] = model.predict_top1(g, ref, 0.0);
            if (dt.has_value()) ++n_called;
        }
        CHECK(n_called > 0, "base_elm_trains_and_predicts");
    }

    // HaploEM converges and scores.
    {
        ReferencePopulation ref = small_population();
        std::vector<Genotype> train(ref.donors.begin(), ref.donors.begin() + 150);

        HaploEM em(ref.loci, 30);
        em.fit(train, ref);
        CHECK(em.n_iterations_run() > 0, "haplo_em_converges");

        double total = 0.0;
        for (const auto& [h, f] : em.haplotype_freq()) { (void)h; total += f; }
        CHECK(std::fabs(total - 1.0) < 1e-6, "haplo_em_freq_sums_to_1");

        auto [dt, p] = em.score_genotype(train[0]);
        CHECK(!p.has_value() || (*p >= 0.0 && *p <= 1.0), "haplo_em_score_in_range");
    }

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
