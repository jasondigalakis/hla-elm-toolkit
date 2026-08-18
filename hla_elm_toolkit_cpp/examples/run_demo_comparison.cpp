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

// run_demo_comparison.cpp
// End-to-end demonstration matching examples/run_demo_comparison.py in
// the Python package: builds a small synthetic reference population,
// trains the base ELM and the Hapl-o-Mat-style EM baseline, evaluates
// both on a held-out split, and prints a summary in the article's
// "value [95% CI]" style.
//
// NOT the real HTO/ORAM/GRPT registry data (not distributable; see
// README.md and the article's Data Availability Statement).

#include <algorithm>
#include <iostream>
#include <random>

#include "hla_elm/BaseELM.hpp"
#include "hla_elm/Data.hpp"
#include "hla_elm/HaploEM.hpp"
#include "hla_elm/Metrics.hpp"

using namespace hla_elm;

static void train_test_split(const std::vector<Genotype>& all, double test_fraction, std::uint64_t seed,
                              std::vector<Genotype>& train, std::vector<Genotype>& test) {
    std::vector<std::size_t> idx(all.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    std::size_t n_test = static_cast<std::size_t>(all.size() * test_fraction);
    std::set<std::size_t> test_idx(idx.begin(), idx.begin() + n_test);
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (test_idx.count(i)) test.push_back(all[i]);
        else train.push_back(all[i]);
    }
}

static bool matches_truth(const Diplotype& dt, const Genotype& g, const ReferencePopulation& ref) {
    for (std::size_t i = 0; i < ref.loci.size(); ++i) {
        const std::string& locus = ref.loci[i];
        auto it = g.truth.find(locus);
        if (it == g.truth.end()) continue;
        std::set<Allele> pred = {dt.first[i], dt.second[i]};
        std::set<Allele> truth = {it->second.first, it->second.second};
        if (pred != truth) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int n_donors = 1500;
    int n_haplotypes = 50;
    std::size_t hidden_size = 150;
    std::uint64_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-donors" && i + 1 < argc) n_donors = std::stoi(argv[++i]);
        else if (arg == "--n-haplotypes" && i + 1 < argc) n_haplotypes = std::stoi(argv[++i]);
        else if (arg == "--hidden-size" && i + 1 < argc) hidden_size = std::stoul(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    }

    std::cout << "Building synthetic reference population (n_donors=" << n_donors
              << ", n_haplotypes=" << n_haplotypes << ", seed=" << seed << ")...\n";
    ReferencePopulation ref = make_synthetic_population(n_donors, kDefaultLoci, 12, n_haplotypes, seed);

    std::vector<Genotype> train, test;
    train_test_split(ref.donors, 0.2, seed, train, test);
    std::cout << "Train: " << train.size() << " donors | Test: " << test.size()
              << " donors | Reference haplotypes: " << ref.haplotype_freq.size() << "\n\n";

    std::size_t max_dt = 3000;

    // ---- Base ELM ----
    BaseELM elm(hidden_size, 1.0, seed);
    elm.fit(train, ref, max_dt, seed);
    std::vector<PredictionResult> elm_results;
    for (const auto& g : test) {
        auto [dt, p] = elm.predict_top1(g, ref, 0.0);
        PredictionResult r;
        r.donor_id = g.donor_id;
        r.called = dt.has_value();
        if (r.called) r.correct = matches_truth(*dt, g, ref);
        elm_results.push_back(r);
    }
    std::cout << format_summary("Base ELM", summarize(elm_results)) << "\n";

    // ---- Hapl-o-Mat-style EM baseline ----
    HaploEM em(ref.loci, 50);
    em.fit(train, ref);
    std::cout << "Hapl-o-Mat-style EM converged in " << em.n_iterations_run() << " iterations\n";
    std::vector<PredictionResult> em_results;
    for (const auto& g : test) {
        auto [dt, p] = em.score_genotype(g, 0.0);
        PredictionResult r;
        r.donor_id = g.donor_id;
        r.called = dt.has_value();
        if (r.called) r.correct = matches_truth(*dt, g, ref);
        em_results.push_back(r);
    }
    std::cout << format_summary("Hapl-o-Mat-style EM", summarize(em_results)) << "\n";

    std::cout << "\nNote: this demo uses a small synthetic reference population for\n"
                 "illustration only; absolute accuracy/call-rate figures are not\n"
                 "comparable to the article's real-registry results (Sections 3.5,\n"
                 "3.7-3.11), which used the (non-distributable) HTO/ORAM/GRPT data.\n";
    return 0;
}
