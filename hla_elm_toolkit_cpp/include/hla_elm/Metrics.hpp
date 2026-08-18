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

// Metrics.hpp
// Evaluation metrics matching hla_elm_toolkit.metrics in the Python
// package. See article Section 2.3 (accuracy, call rate) and Section
// 2.8 (95% CI via Wilson score method).

#pragma once

#include <cmath>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace hla_elm {

struct PredictionResult {
    std::string donor_id;
    bool called = false;
    std::optional<bool> correct;  // unset if truth unknown or not called
};

// Wilson score confidence interval for a binomial proportion. Preferred
// over a naive percentile bootstrap or normal-approximation (Wald)
// interval for the same reasons given in the article (Section 2.8):
// several comparator results sit at small n and/or at 0%/100% observed
// accuracy, where a naive bootstrap is degenerate.
inline std::pair<double, double> wilson_score_interval(int successes, int n, double confidence = 0.95) {
    if (n <= 0) throw std::invalid_argument("n must be positive");
    if (successes < 0 || successes > n) throw std::invalid_argument("successes must be in [0, n]");

    double z;
    if (confidence == 0.90) z = 1.644853627;
    else if (confidence == 0.95) z = 1.959963985;
    else if (confidence == 0.99) z = 2.575829304;
    else throw std::invalid_argument("confidence must be 0.90, 0.95, or 0.99");

    double p = static_cast<double>(successes) / n;
    double denom = 1.0 + z * z / n;
    double center = (p + z * z / (2.0 * n)) / denom;
    double half = (z * std::sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))) / denom;
    double lo = std::max(0.0, center - half);
    double hi = std::min(1.0, center + half);
    return {lo, hi};
}

struct Summary {
    int n_total = 0, n_called = 0, n_scoreable = 0, n_correct = 0;
    double call_rate = 0.0;
    std::optional<double> accuracy;
    std::pair<double, double> call_rate_ci{0, 0};
    std::pair<double, double> accuracy_ci{0, 0};
};

inline Summary summarize(const std::vector<PredictionResult>& results, double confidence = 0.95) {
    Summary s;
    s.n_total = static_cast<int>(results.size());
    for (const auto& r : results) {
        if (r.called) {
            ++s.n_called;
            if (r.correct.has_value()) {
                ++s.n_scoreable;
                if (*r.correct) ++s.n_correct;
            }
        }
    }
    if (s.n_total > 0) {
        s.call_rate = static_cast<double>(s.n_called) / s.n_total;
        s.call_rate_ci = wilson_score_interval(s.n_called, s.n_total, confidence);
    }
    if (s.n_scoreable > 0) {
        s.accuracy = static_cast<double>(s.n_correct) / s.n_scoreable;
        s.accuracy_ci = wilson_score_interval(s.n_correct, s.n_scoreable, confidence);
    }
    return s;
}

inline std::string format_summary(const std::string& name, const Summary& s) {
    std::ostringstream out;
    out << name << ": call rate " << (s.call_rate * 100) << "% (" << s.n_called << "/" << s.n_total << ")"
        << " [" << (s.call_rate_ci.first * 100) << "-" << (s.call_rate_ci.second * 100) << "%]";
    if (s.accuracy.has_value()) {
        out << " | accuracy " << (*s.accuracy * 100) << "% (" << s.n_correct << "/" << s.n_scoreable << ")"
            << " [" << (s.accuracy_ci.first * 100) << "-" << (s.accuracy_ci.second * 100) << "%]";
    } else {
        out << " | accuracy n/a (no scoreable donors)";
    }
    return out.str();
}

}  // namespace hla_elm
