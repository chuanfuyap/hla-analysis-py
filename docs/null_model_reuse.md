# Null-model reuse

Standard and survival omnibus analyses reuse a baseline null model when the final modelling observations and baseline design match. The public analysis functions, configuration options and output columns are unchanged. Single-coefficient tests continue to use their existing fitting methods.

For standard association, the baseline contains the phenotype, covariates and any `condition_on` terms. Conditioning genotype matrices are constructed once before parallelisation. Multiple-phenotype analysis prepares a separate null for each phenotype; phenotype missingness and outcome values are checked separately.

For survival association, the baseline contains the endpoint and covariates. Static Cox and time-varying Cox models have separate specifications. Time-varying validation includes subject IDs, interval starts and stops, event indicators and the encoded covariate values. No conditional survival API has been added.

## When a null can be reused

The built-in genotype adapters obtain the available genetic samples through the existing sample-alignment utilities, including paired hardcall columns. Existing complete-case preprocessing determines the baseline observations.

Workers receive a compact immutable summary containing the null log-likelihood, model degrees of freedom, diagnostics, sample IDs and a fingerprint of the ordered baseline observations and design. Fitted statsmodels or lifelines objects are not transferred to workers.

Each omnibus alternative still receives its own genotype preparation and complete-case filtering. Its baseline is checked against the summary, including column order, values, data types and categorical levels. A different sample set, phenotype, covariate design or survival interval structure triggers a local null fit on that variant's actual observations. Missing genotype data at one position does not exclude samples from other positions.

Null diagnostics remain part of the existing `INFO` handling. Standard LRT degrees of freedom use the fitted model ranks; Cox LRT degrees of freedom use the difference in fitted parameter counts.

A Cox model without baseline predictors has no intercept parameter to estimate. Its null partial log-likelihood is evaluated directly using the Efron tie rule and the same risk-set conventions as lifelines. For start–stop data, a subject is at risk when `start < event_time <= stop`.

## Implementation

| File under `hapy/stats/` | Change |
| --- | --- |
| `models/_null.py` | Lightweight null summary and observation/design validation. |
| `adapters/_samples.py` | Shared sample lookup and conservative omnibus eligibility check. |
| `adapters/aa.py`, `adapters/hla.py`, `adapters/snp.py` | Sample accessors using existing alignment conventions. |
| `models/standard.py` | Baseline fitting and cached/local null selection for omnibus LRTs. |
| `models/survival.py` | Static/time-varying Cox baseline summaries, zero-predictor partial likelihood and cached/local selection. |
| `runners/standard_runner.py` | Shared conditioning preparation and null fitting before workers, including one null per phenotype. |
| `runners/survival_runner.py` | Shared survival baseline preparation before workers. |
| `preprocess.py` | Allow survival baseline tables with zero predictor columns without calling dummy encoding on an empty column set. |

## Validation

Local tests in `tests/test_null_reuse.py` cover independent-fit equivalence, reuse counts, linear/logistic regression, static/time-varying Cox, missing observations, conditioning, multiple phenotypes, softcalls/hardcalls, categorical design changes, numerical failures, no covariates and sequential/thread/process execution. The zero-predictor Cox calculation is checked against lifelines and explicit tied-event risk sets with delayed entry.

The affected regression selection passed **148 tests** (`test_null_reuse.py`, `test_analyse_many.py`, `test_survival.py`, `test_survival_diagnostics.py` and `test_fit_diagnostics.py`). Dependency warnings remain visible; this was not a full-repository test run.

Tests and the benchmark script remain in the locally ignored `tests/` directory.

## Synthetic benchmark

Median elapsed time over three interleaved runs with one worker and one OpenBLAS thread. The reference performs the original independent full/null fitting at every position; the optimised run uses shared null summaries. All result tables matched numerically (`rtol=1e-8`, `atol=1e-10`). Timings are illustrative and depend on data and hardware.

| Analysis | Samples | AA positions | Independent | Shared null | Null fits | Reduction in null fits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic omnibus | 600 | 24 | 0.268 s | 0.226 s | 24 → 1 | 95.8% |
| Static Cox omnibus | 300 | 12 | 0.271 s | 0.212 s | 12 → 1 | 91.7% |
| Time-varying Cox omnibus | 120 | 6 | 1.341 s | 1.428 s | 6 → 1 | 83.3% |

The first two examples improved by approximately 1.18× and 1.28×. The small time-varying example was approximately 6% slower: preparing an additional common timeline outweighed the saved null fits. Null-fit reduction does not guarantee an end-to-end speed improvement.

Run `tests/benchmark_null_reuse.py` from the repository with the package dependencies installed to repeat the comparison.

## Limitations

- Genotype preparation, per-variant complete-case tables and dynamic survival preprocessing still run for each tested position. These can dominate elapsed time.
- The metadata eligibility check is conservative. It can prepare a baseline for a position that subsequently collapses to a single usable genotype column or is skipped after QC.
- Changed observation order causes a conservative local refit even when an equivalent model could be fitted after reordering.
- Custom internal adapters without the sample accessor use local null fits.
- Incomplete hardcall pairs are omitted from the common baseline; if a variant adapter produces different observations, its null is fitted locally.
- Existing lifelines warnings remain visible. The existing explicit diagnostics do not turn every warning into an `INFO` failure.
