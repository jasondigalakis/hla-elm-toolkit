# hla-elm-toolkit (C++)

C++17 port of the Python `hla_elm_toolkit` reference package, covering
the **core** methods compared in:

> Kepentzis, S.; Chatzistamatiou, T.; Digalakis, J.; Petropoulou, O.;
> Matsopoulos, G.K.; Koutsouris, D. *Development of an Extreme Learning
> Machine approach to upgrade low/mid to high resolution HLA data
> improving the usability of donor data from registries.* Genes.

This is a **companion, narrower** deliverable to the Python package
(`hla_elm_toolkit_python`, distributed alongside this archive), not a
1:1 port of every module. See Section 2 below for exactly what is and
is not included, and why.

---

## 1. Scope and honesty about limitations

Everything said in the Python package's README Section 1 applies here
too: this is a **from-scratch reference reimplementation**, not the
article's original, registry-scale codebase (now open-source at
https://biomedntuagr.sharepoint.com/:f:/g/IgCgDy6zAJ8HTYmovHlYxrQXAeRFyCh7S8HutOUI6ddvDLg?e=0bO3bG,
per the article's Data Availability Statement). The base ELM's input
encoding and hidden layer are simplified relative to the article's
domain-structured HL-1/HL-2/HL-3 topology (Section 2.4), in exactly the
same way as the Python package. No real registry data is included; the
synthetic generator is for demonstration only.

**No external dependencies.** This package deliberately implements its
own minimal dense-matrix class (`include/hla_elm/Matrix.hpp`) — random
initialization, multiply, transpose, and a Gaussian-elimination linear
solve — rather than depending on Eigen, BLAS, or LAPACK. This keeps the
build to "a C++17 compiler and nothing else," at the cost of the
performance a tuned linear-algebra library would offer at registry
scale. For serious performance work, replacing `Matrix` with an
Eigen-backed type would be the natural next step; the public API used
by `BaseELM.hpp` and `HaploEM.hpp` is small enough that this is a
localized change.

## 2. What is included, and what is not

This C++ port covers the two methods most central to the article's own
comparator benchmarking pipeline:

| Component | Status |
|---|---|
| `Matrix` (dense linear algebra) | Implemented |
| `Data` (genotype/haplotype types, synthetic population) | Implemented |
| `Metrics` (Wilson score 95% CI, call rate, accuracy) | Implemented |
| `BaseELM` (Algorithm 1-2, Section 2.4-2.5) | Implemented |
| `HaploEM` (Algorithm 5, Hapl-o-Mat-style EM; Sections 3.7, 3.11) | Implemented |
| KELM, WELM, Ensemble ELM (Section 2.6) | **Not ported** — see Python package |
| GRIMM-style, HaploStats-style baselines (Section 2.7.1) | **Not ported** — see Python package |
| MLP, gradient-boosted-tree comparators (Section 2.7) | **Not ported** — these rely on scikit-learn in the Python package and have no equivalent here |

If you need the extensions, the graph-based/Bayesian baselines, or the
MLP/GBT comparators, use the Python package (`hla_elm_toolkit_python`),
which implements all of them. This C++ package exists specifically for
users who need a dependency-free, compiled implementation of the core
ELM-vs-EM-baseline comparison (e.g. for embedding in a larger C++
pipeline, or for a performance-sensitive setting), not as a complete
replacement for the Python package's broader coverage.

## 3. Building

Requires a C++17 compiler and CMake 3.15+. No other dependencies.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

./test_basic                                    # run the test suite
./run_demo_comparison --n-donors 1500 --seed 0   # run the demo
```

Without CMake, direct compilation also works:

```bash
g++ -std=c++17 -O2 -Iinclude tests/test_basic.cpp -o test_basic
g++ -std=c++17 -O2 -Iinclude examples/run_demo_comparison.cpp -o run_demo_comparison
```

## 4. Quick start

```cpp
#include "hla_elm/BaseELM.hpp"
#include "hla_elm/Data.hpp"
#include "hla_elm/Metrics.hpp"

using namespace hla_elm;

int main() {
    ReferencePopulation ref = make_synthetic_population(1000, kDefaultLoci, 12, 40, /*seed=*/0);
    std::vector<Genotype> train(ref.donors.begin(), ref.donors.begin() + 800);
    std::vector<Genotype> test(ref.donors.begin() + 800, ref.donors.end());

    BaseELM model(150, /*reg_lambda=*/1.0, /*seed=*/0);
    model.fit(train, ref, /*max_diplotypes=*/3000);

    std::vector<PredictionResult> results;
    for (const auto& g : test) {
        auto [dt, post_p] = model.predict_top1(g, ref, /*theta=*/0.0);
        PredictionResult r;
        r.donor_id = g.donor_id;
        r.called = dt.has_value();
        // compare dt against g.truth yourself, per-locus or jointly, to set r.correct
        results.push_back(r);
    }
    std::cout << format_summary("Base ELM", summarize(results)) << "\n";
}
```

## 5. Package layout

```
hla_elm_toolkit_cpp/
├── README.md                        (this file)
├── CMakeLists.txt
├── include/hla_elm/
│   ├── Matrix.hpp                   dependency-free dense matrix + linear solve
│   ├── Data.hpp                     genotype/haplotype types, synthetic population
│   ├── Metrics.hpp                  Wilson score 95% CI, call rate, accuracy
│   ├── BaseELM.hpp                  Algorithm 1 (training) + Algorithm 2 (inference)
│   └── HaploEM.hpp                  Algorithm 5 (Hapl-o-Mat-style EM)
├── examples/
│   └── run_demo_comparison.cpp      end-to-end ELM vs. EM baseline demo
└── tests/
    └── test_basic.cpp               16 smoke tests (all passing)
```

## 6. Citation

If you use this code, please cite the article above. This package and
`hla_elm_toolkit_python` are companion deliverables prepared alongside
the article's revision; the article's own original implementation is
the canonical reference (see Section 1).

## 7. License

Not yet assigned; treat as "all rights reserved, provided for internal
review purposes" until the authors specify otherwise. The article's
original implementation is a separate codebase with its own license;
see https://biomedntuagr.sharepoint.com/:f:/g/IgCgDy6zAJ8HTYmovHlYxrQXAeRFyCh7S8HutOUI6ddvDLg?e=0bO3bG.
