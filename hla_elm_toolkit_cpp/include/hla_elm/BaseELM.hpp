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

// BaseELM.hpp
// Base Extreme Learning Machine, matching hla_elm_toolkit.elm.base_elm
// in the Python package (Algorithms 1-2, article Sections 2.4-2.5).
//
// As in the Python package: this implements the article's core ELM
// mathematics (randomly generated, untrained input-to-hidden weights;
// closed-form, Moore-Penrose-pseudo-inverse-equivalent output weights
// via ridge-regularized least squares) over a flat multi-hot
// locus/allele input encoding, rather than the exact domain-structured
// HL-1/HL-2/HL-3 topology described in the article. See the top-level
// README.md Section 1 for the same caveat as the Python package.

#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include "Data.hpp"
#include "Matrix.hpp"

namespace hla_elm {

// Multi-hot encoder: one input dimension per (locus, allele) pair
// observed anywhere in the reference population.
class InputEncoder {
public:
    static InputEncoder fit(const ReferencePopulation& ref) {
        InputEncoder enc;
        enc.loci_ = ref.loci;
        std::size_t idx = 0;
        for (const auto& locus : ref.loci) {
            for (const auto& allele : ref.alleles_at(locus)) {
                enc.index_[{locus, allele}] = idx++;
            }
        }
        return enc;
    }

    std::size_t n_inputs() const { return index_.size(); }

    Matrix encode(const Genotype& g, const ReferencePopulation& ref) const {
        Matrix x(1, n_inputs(), 0.0);
        for (const auto& locus : loci_) {
            std::vector<Allele> alleles;
            auto it = g.calls.find(locus);
            if (it == g.calls.end() || !it->second.has_value()) {
                alleles = ref.alleles_at(locus);  // untyped -> all reference alleles active
            } else {
                alleles.assign(it->second->begin(), it->second->end());
            }
            for (const auto& a : alleles) {
                auto key_it = index_.find({locus, a});
                if (key_it != index_.end()) x(0, key_it->second) = 1.0;
            }
        }
        return x;
    }

private:
    std::vector<std::string> loci_;
    std::map<std::pair<std::string, std::string>, std::size_t> index_;
};

// Maps candidate diplotypes (built from the reference haplotype table)
// to output-node indices.
class OutputCatalog {
public:
    static OutputCatalog from_reference(const ReferencePopulation& ref,
                                         std::optional<std::size_t> max_diplotypes = std::nullopt) {
        OutputCatalog cat;
        std::vector<Haplotype> haps;
        for (const auto& [h, f] : ref.haplotype_freq) { (void)f; haps.push_back(h); }

        for (std::size_t i = 0; i < haps.size(); ++i) {
            for (std::size_t j = i; j < haps.size(); ++j) {
                cat.diplotypes_.emplace_back(haps[i], haps[j]);
                if (max_diplotypes && cat.diplotypes_.size() >= *max_diplotypes) break;
            }
            if (max_diplotypes && cat.diplotypes_.size() >= *max_diplotypes) break;
        }
        for (std::size_t i = 0; i < cat.diplotypes_.size(); ++i) cat.index_[cat.diplotypes_[i]] = i;
        return cat;
    }

    std::size_t n_outputs() const { return diplotypes_.size(); }
    const std::vector<Diplotype>& diplotypes() const { return diplotypes_; }
    std::optional<std::size_t> index_of(const Diplotype& dt) const {
        auto it = index_.find(dt);
        if (it == index_.end()) return std::nullopt;
        return it->second;
    }

private:
    std::vector<Diplotype> diplotypes_;
    std::map<Diplotype, std::size_t> index_;
};

class BaseELM {
public:
    explicit BaseELM(std::size_t hidden_size, double reg_lambda = 1.0, std::uint64_t seed = 0)
        : hidden_size_(hidden_size), reg_lambda_(reg_lambda), seed_(seed) {}

    // Train following Algorithm 1: for each training genotype, sample a
    // target diplotype uniformly at random from the diplotypes
    // compatible with that (possibly ambiguous) genotype, accumulate
    // the hidden-layer output matrix H and one-hot target matrix, then
    // solve beta in closed form.
    void fit(const std::vector<Genotype>& genotypes, const ReferencePopulation& ref,
             std::optional<std::size_t> max_diplotypes = 4000, std::optional<std::uint64_t> rng_seed = std::nullopt) {
        encoder_ = InputEncoder::fit(ref);
        catalog_ = OutputCatalog::from_reference(ref, max_diplotypes);

        std::mt19937_64 rng(rng_seed.value_or(seed_));
        W_ = Matrix::random_uniform(encoder_->n_inputs(), hidden_size_, -1.0, 1.0, seed_);
        b_ = Matrix::random_uniform(1, hidden_size_, -1.0, 1.0, seed_ + 1);

        std::vector<Matrix> x_rows;
        std::vector<Matrix> t_rows;
        for (const auto& g : genotypes) {
            std::vector<Diplotype> all = compatible_diplotypes(g, ref);
            std::vector<Diplotype> candidates;
            for (const auto& dt : all) if (catalog_->index_of(dt)) candidates.push_back(dt);
            if (candidates.empty()) continue;

            std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
            const Diplotype& target = candidates[pick(rng)];

            x_rows.push_back(encoder_->encode(g, ref));
            Matrix t(1, catalog_->n_outputs(), 0.0);
            t(0, *catalog_->index_of(target)) = 1.0;
            t_rows.push_back(t);
        }
        if (x_rows.empty()) throw std::runtime_error("No training genotype had a compatible diplotype in the output catalog.");

        Matrix X(x_rows.size(), encoder_->n_inputs());
        Matrix T(t_rows.size(), catalog_->n_outputs());
        for (std::size_t i = 0; i < x_rows.size(); ++i) {
            for (std::size_t j = 0; j < encoder_->n_inputs(); ++j) X(i, j) = x_rows[i](0, j);
            for (std::size_t j = 0; j < catalog_->n_outputs(); ++j) T(i, j) = t_rows[i](0, j);
        }

        Matrix H = hidden_activation(X);
        Matrix HtH = H.transpose() * H;
        HtH.add_scaled_identity(1.0 / reg_lambda_);
        Matrix HtT = H.transpose() * T;
        beta_ = HtH.solve(HtT);  // Algorithm 1, line 18
    }

    // Rank candidate diplotypes by output score, restricted to those
    // compatible with this genotype (Algorithm 2), with scores
    // converted to a softmax-normalized distribution over the
    // compatible subset for readability.
    std::vector<std::pair<Diplotype, double>> predict_ranked(
        const Genotype& g, const ReferencePopulation& ref, std::size_t top_k = 5) const {
        std::vector<Diplotype> all = compatible_diplotypes(g, ref);
        std::vector<Diplotype> candidates;
        for (const auto& dt : all) if (catalog_->index_of(dt)) candidates.push_back(dt);
        if (candidates.empty()) return {};

        Matrix x = encoder_->encode(g, ref);
        Matrix h = hidden_activation(x);
        Matrix scores = h * (*beta_);  // 1 x n_outputs

        std::vector<double> raw;
        for (const auto& dt : candidates) raw.push_back(scores(0, *catalog_->index_of(dt)));
        double max_raw = *std::max_element(raw.begin(), raw.end());
        double sum_exp = 0.0;
        for (auto& v : raw) { v = std::exp(v - max_raw); sum_exp += v; }
        for (auto& v : raw) v /= sum_exp;

        std::vector<std::pair<Diplotype, double>> ranked;
        for (std::size_t i = 0; i < candidates.size(); ++i) ranked.emplace_back(candidates[i], raw[i]);
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        if (ranked.size() > top_k) ranked.resize(top_k);
        return ranked;
    }

    std::pair<std::optional<Diplotype>, std::optional<double>> predict_top1(
        const Genotype& g, const ReferencePopulation& ref, double theta = 0.0) const {
        auto ranked = predict_ranked(g, ref, 1);
        if (ranked.empty()) return {std::nullopt, std::nullopt};
        if (ranked[0].second >= theta) return {ranked[0].first, ranked[0].second};
        return {std::nullopt, ranked[0].second};
    }

private:
    Matrix hidden_activation(const Matrix& X) const {
        // sigmoid(X @ W + b), b broadcast across rows.
        Matrix pre = X * W_;
        for (std::size_t i = 0; i < pre.rows(); ++i)
            for (std::size_t j = 0; j < pre.cols(); ++j)
                pre(i, j) += b_(0, j);
        return pre.sigmoid();
    }

    std::size_t hidden_size_;
    double reg_lambda_;
    std::uint64_t seed_;

    std::optional<InputEncoder> encoder_;
    std::optional<OutputCatalog> catalog_;
    Matrix W_, b_;
    std::optional<Matrix> beta_;
};

}  // namespace hla_elm
