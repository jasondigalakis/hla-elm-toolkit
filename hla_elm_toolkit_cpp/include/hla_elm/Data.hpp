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

// Data.hpp
// Genotype / haplotype / diplotype representations, matching
// hla_elm_toolkit.data in the Python package. See article Sections
// 2.1-2.2. Also provides a synthetic reference-population generator
// for the examples and tests (the real HTO/ORAM/GRPT registry data
// used in the article is not distributable; Data Availability Statement).

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace hla_elm {

using Allele = std::string;
using Haplotype = std::vector<Allele>;   // one allele per locus, in a fixed locus order
using Diplotype = std::pair<Haplotype, Haplotype>;

inline const std::vector<std::string> kDefaultLoci = {"A", "B", "C", "DRB1", "DQB1"};

struct Genotype {
    std::string donor_id;
    // calls[locus] = set of alleles compatible with the (possibly ambiguous)
    // typing at that locus; empty optional = untyped (missing-locus imputation).
    std::map<std::string, std::optional<std::set<Allele>>> calls;
    // Confirmatory high-resolution truth, if known: truth[locus] = (allele1, allele2).
    std::map<std::string, std::pair<Allele, Allele>> truth;

    bool is_typed(const std::string& locus) const {
        auto it = calls.find(locus);
        return it != calls.end() && it->second.has_value();
    }
};

struct ReferencePopulation {
    std::vector<std::string> loci;
    std::map<Haplotype, double> haplotype_freq;
    std::vector<Genotype> donors;

    std::vector<Allele> alleles_at(const std::string& locus) const {
        auto idx_it = std::find(loci.begin(), loci.end(), locus);
        std::size_t idx = static_cast<std::size_t>(idx_it - loci.begin());
        std::set<Allele> seen;
        for (const auto& [hap, freq] : haplotype_freq) {
            (void)freq;
            seen.insert(hap[idx]);
        }
        return std::vector<Allele>(seen.begin(), seen.end());
    }
};

// Enumerate diplotypes in `ref.haplotype_freq` compatible with `genotype`
// at the requested loci (default: all loci in `ref`). Mirrors
// hla_elm_toolkit.data.compatible_diplotypes in the Python package.
//
// Reference implementation only: enumerates all O(n^2) haplotype pairs
// directly rather than using an indexed structure, so it is not intended
// for registry-scale (10^5+ donor) throughput without further
// optimization -- see README.md.
inline std::vector<Diplotype> compatible_diplotypes(
    const Genotype& genotype, const ReferencePopulation& ref,
    const std::vector<std::string>* loci_override = nullptr) {
    const std::vector<std::string>& loci = loci_override ? *loci_override : ref.loci;
    std::vector<std::size_t> idxs;
    for (const auto& l : loci) {
        auto it = std::find(ref.loci.begin(), ref.loci.end(), l);
        idxs.push_back(static_cast<std::size_t>(it - ref.loci.begin()));
    }

    auto allele_ok = [&](const Haplotype& hap) {
        for (std::size_t k = 0; k < loci.size(); ++k) {
            auto call_it = genotype.calls.find(loci[k]);
            if (call_it == genotype.calls.end() || !call_it->second.has_value()) continue;  // untyped: any allele allowed
            const auto& allowed = *call_it->second;
            if (allowed.find(hap[idxs[k]]) == allowed.end()) return false;
        }
        return true;
    };

    std::vector<Haplotype> candidates;
    for (const auto& [hap, freq] : ref.haplotype_freq) {
        (void)freq;
        if (allele_ok(hap)) candidates.push_back(hap);
    }

    std::vector<Diplotype> out;
    for (std::size_t i = 0; i < candidates.size(); ++i)
        for (std::size_t j = i; j < candidates.size(); ++j)
            out.emplace_back(candidates[i], candidates[j]);
    return out;
}

inline double hardy_weinberg_likelihood(const Diplotype& dt, const ReferencePopulation& ref) {
    auto it1 = ref.haplotype_freq.find(dt.first);
    auto it2 = ref.haplotype_freq.find(dt.second);
    double f1 = it1 != ref.haplotype_freq.end() ? it1->second : 0.0;
    double f2 = it2 != ref.haplotype_freq.end() ? it2->second : 0.0;
    if (dt.first == dt.second) return f1 * f1;
    return 2.0 * f1 * f2;
}

// Posterior probabilities over a compatible-diplotype set (Section 2.3):
// Post-P(dt) = L(dt) / sum_i L(d_i). Returned sorted descending by posterior.
inline std::vector<std::pair<Diplotype, double>> posterior_probabilities(
    const std::vector<Diplotype>& diplotypes, const ReferencePopulation& ref) {
    std::vector<std::pair<Diplotype, double>> scored;
    double total = 0.0;
    for (const auto& dt : diplotypes) {
        double l = hardy_weinberg_likelihood(dt, ref);
        scored.emplace_back(dt, l);
        total += l;
    }
    if (total <= 0.0) return {};
    for (auto& [dt, l] : scored) { (void)dt; l /= total; }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return scored;
}

// Synthetic reference-population generator (examples/tests only).
// NOT the real HTO/ORAM/GRPT registry data used in the article, which
// is not distributable (Data Availability Statement).
inline ReferencePopulation make_synthetic_population(
    int n_donors = 1500, std::vector<std::string> loci = kDefaultLoci,
    int n_alleles_per_locus = 12, int n_haplotypes = 50, std::uint64_t seed = 0) {
    std::mt19937_64 rng(seed);

    std::map<std::string, std::vector<Allele>> alleles_by_locus;
    for (const auto& locus : loci) {
        std::vector<Allele> alleles;
        for (int i = 1; i <= n_alleles_per_locus; ++i) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s*%02d:01", locus.c_str(), i);
            alleles.emplace_back(buf);
        }
        alleles_by_locus[locus] = alleles;
    }

    std::vector<Haplotype> haplotypes;
    std::set<Haplotype> seen;
    while (static_cast<int>(haplotypes.size()) < n_haplotypes) {
        Haplotype hap;
        for (const auto& locus : loci) {
            std::uniform_int_distribution<std::size_t> pick(0, alleles_by_locus[locus].size() - 1);
            hap.push_back(alleles_by_locus[locus][pick(rng)]);
        }
        if (seen.insert(hap).second) haplotypes.push_back(hap);
    }

    // Zipf-like frequency spectrum (mirrors the Python generator).
    std::vector<double> freqs(haplotypes.size());
    std::uniform_real_distribution<double> jitter(0.85, 1.15);
    double total = 0.0;
    for (std::size_t i = 0; i < haplotypes.size(); ++i) {
        double raw = 1.0 / std::pow(static_cast<double>(i + 1), 1.3);
        freqs[i] = raw * jitter(rng);
        total += freqs[i];
    }
    for (auto& f : freqs) f /= total;

    ReferencePopulation ref;
    ref.loci = loci;
    for (std::size_t i = 0; i < haplotypes.size(); ++i) ref.haplotype_freq[haplotypes[i]] = freqs[i];

    std::discrete_distribution<std::size_t> hap_dist(freqs.begin(), freqs.end());
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    for (int i = 0; i < n_donors; ++i) {
        const Haplotype& h1 = haplotypes[hap_dist(rng)];
        const Haplotype& h2 = haplotypes[hap_dist(rng)];

        Genotype g;
        char idbuf[16];
        std::snprintf(idbuf, sizeof(idbuf), "D%06d", i);
        g.donor_id = idbuf;

        for (std::size_t k = 0; k < loci.size(); ++k) {
            const std::string& locus = loci[k];
            g.truth[locus] = {h1[k], h2[k]};

            double r = unif(rng);
            std::set<Allele> true_alleles = {h1[k], h2[k]};
            if (r < 0.15) {
                g.calls[locus] = std::nullopt;  // untyped
            } else if (r < 0.45) {
                std::set<Allele> ambiguous = true_alleles;
                std::vector<Allele> pool;
                for (const auto& a : alleles_by_locus[locus])
                    if (true_alleles.find(a) == true_alleles.end()) pool.push_back(a);
                std::shuffle(pool.begin(), pool.end(), rng);
                for (std::size_t p = 0; p < std::min<std::size_t>(2, pool.size()); ++p)
                    ambiguous.insert(pool[p]);
                g.calls[locus] = ambiguous;
            } else {
                g.calls[locus] = true_alleles;
            }
        }
        ref.donors.push_back(std::move(g));
    }

    return ref;
}

}  // namespace hla_elm
