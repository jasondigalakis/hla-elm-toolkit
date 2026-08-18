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

// HaploEM.hpp
// Expectation-maximization (EM) haplotype-frequency estimation and
// imputation, matching hla_elm_toolkit.baselines.haplo_em in the Python
// package (Algorithm 5, article Section 2.7.1). The same core class
// serves both the two-locus baseline (Section 3.7) and the five-locus
// in-house baseline (Section 3.11) by varying which loci are passed in.

#pragma once

#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "Data.hpp"

namespace hla_elm {

class HaploEM {
public:
    explicit HaploEM(std::vector<std::string> loci, int max_iter = 100, double tol = 1e-6)
        : loci_(std::move(loci)), max_iter_(max_iter), tol_(tol) {}

    // Phase 1 (offline) of Algorithm 5: estimate haplotype frequencies
    // from a reference panel of unphased, possibly ambiguous genotypes.
    void fit(const std::vector<Genotype>& genotypes, const ReferencePopulation& ref_for_enumeration) {
        std::vector<std::vector<Diplotype>> per_genotype_dts;
        for (const auto& g : genotypes) {
            auto dts = compatible_diplotypes(g, ref_for_enumeration, &loci_);
            if (!dts.empty()) per_genotype_dts.push_back(std::move(dts));
        }
        if (per_genotype_dts.empty())
            throw std::runtime_error("No genotype had any phase-compatible diplotype for the given loci.");

        std::set<Haplotype> all_haps_set;
        for (const auto& dts : per_genotype_dts)
            for (const auto& dt : dts) { all_haps_set.insert(dt.first); all_haps_set.insert(dt.second); }
        std::vector<Haplotype> all_haps(all_haps_set.begin(), all_haps_set.end());

        std::map<Haplotype, double> F;
        for (const auto& h : all_haps) F[h] = 1.0 / all_haps.size();

        n_iterations_run_ = 0;
        for (int iter = 0; iter < max_iter_; ++iter) {
            std::map<Haplotype, double> counts;
            for (const auto& h : all_haps) counts[h] = 0.0;

            for (const auto& dts : per_genotype_dts) {
                std::vector<double> weights(dts.size());
                double total = 0.0;
                for (std::size_t i = 0; i < dts.size(); ++i) {
                    const auto& [h1, h2] = dts[i];
                    double like = (h1 == h2) ? F[h1] * F[h1] : F[h1] * F[h2];
                    weights[i] = like;
                    total += like;
                }
                if (total <= 0.0) continue;
                for (std::size_t i = 0; i < dts.size(); ++i) {
                    double post = weights[i] / total;
                    counts[dts[i].first] += post;
                    counts[dts[i].second] += post;
                }
            }

            double total_count = 0.0;
            for (const auto& [h, c] : counts) { (void)h; total_count += c; }
            if (total_count <= 0.0) break;

            std::map<Haplotype, double> F_new;
            double delta = 0.0;
            for (const auto& h : all_haps) {
                F_new[h] = counts[h] / total_count;
                delta = std::max(delta, std::fabs(F_new[h] - F[h]));
            }
            F = std::move(F_new);
            n_iterations_run_ = iter + 1;
            if (delta < tol_) break;
        }

        haplotype_freq_ = std::move(F);
    }

    ReferencePopulation as_reference_population() const {
        ReferencePopulation ref;
        ref.loci = loci_;
        ref.haplotype_freq = haplotype_freq_;
        return ref;
    }

    // Phase 2 (online) of Algorithm 5: rank diplotypes compatible with a
    // query genotype by posterior probability, return the top-1 call if
    // it clears the confidence threshold theta.
    std::pair<std::optional<Diplotype>, std::optional<double>> score_genotype(
        const Genotype& genotype, double theta = 0.0) const {
        ReferencePopulation ref = as_reference_population();
        auto dts = compatible_diplotypes(genotype, ref, &loci_);
        if (dts.empty()) return {std::nullopt, std::nullopt};

        std::vector<std::pair<Diplotype, double>> scored;
        double total = 0.0;
        for (const auto& [h1, h2] : dts) {
            double f1 = haplotype_freq_.count(h1) ? haplotype_freq_.at(h1) : 0.0;
            double f2 = haplotype_freq_.count(h2) ? haplotype_freq_.at(h2) : 0.0;
            double like = (h1 == h2) ? f1 * f1 : f1 * f2;
            scored.emplace_back(Diplotype{h1, h2}, like);
            total += like;
        }
        if (total <= 0.0) return {std::nullopt, std::nullopt};
        for (auto& [dt, l] : scored) { (void)dt; l /= total; }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        if (scored[0].second >= theta) return {scored[0].first, scored[0].second};
        return {std::nullopt, scored[0].second};
    }

    int n_iterations_run() const { return n_iterations_run_; }
    const std::map<Haplotype, double>& haplotype_freq() const { return haplotype_freq_; }

private:
    std::vector<std::string> loci_;
    int max_iter_;
    double tol_;
    int n_iterations_run_ = 0;
    std::map<Haplotype, double> haplotype_freq_;
};

}  // namespace hla_elm
