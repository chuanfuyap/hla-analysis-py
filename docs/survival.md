# Survival analysis

`hp.stats.survival()` runs Cox regression for the AA, HLA or SNP block of a loaded
`HLAdata` object. `hp.survival()` is the same function. It returns a pandas
DataFrame; results are not saved automatically.

Without `dynamic_covar`, hapy uses `lifelines.CoxPHFitter`. Supplying time-varying
covariates selects `lifelines.CoxTimeVaryingFitter` and constructs start–stop
intervals internally. Covariate values can change over follow-up; this does not
fit a separate time-varying regression coefficient.

## Endpoint and sample tables

`event_time` must contain exactly these columns, with one row per subject:

| Column | Meaning |
|---|---|
| `sample_id` | Unique subject IID, matching genotype sample IDs. |
| `time` | Finite positive follow-up duration to the event or censoring. |
| `event` | `1` if the event occurred; `0` for right censoring at the recorded time. |

For a censored subject, `time` is the last observed follow-up time, not zero.
Use a consistent time origin and unit, such as days since recruitment, for all
subjects and any dynamic updates. No time unit is inferred or converted.

For example, an endpoint file might begin:

```text
sample_id	time	event
001	10	1
002	12	0
003	8	1
```

`famfile` is a pandas DataFrame containing an `IID` column defining the target
samples. A full PLINK FAM table can be used, but its `PHENO` column is not used
by survival analysis. Alternatively, construct an IID table from the endpoints.
Baseline predictors belong in the IID-indexed `covar` table, not `event_time`.

Read identifiers as strings to preserve leading zeros. Endpoint IDs must be
unique and non-missing. Invalid endpoint rows, including missing/non-positive
times, missing events or events other than 0/1, raise validation errors; missing
endpoints should be represented by omitting the subject's endpoint row.

## Static Cox example

With `hladat` already loaded, prepare endpoint and baseline covariate files:

```python
import pandas as pd
import hapy as hp

# survival.tsv: sample_id, time, event
# baseline_covariates.tsv: IID, age, SEX, PC1, PC2, ...
event_time = pd.read_csv(
    "survival.tsv", sep="\t", dtype={"sample_id": str},
)
famfile = event_time[["sample_id"]].rename(columns={"sample_id": "IID"})
covar = pd.read_csv(
    "baseline_covariates.tsv", sep="\t", dtype={"IID": str},
).set_index("IID")
covar["SEX"] = covar["SEX"].astype("category")

config = hp.stats.SurvivalConfig(n_jobs=1)
results = hp.stats.survival(
    hladat,
    famfile=famfile,
    config=config,
    kind="AA",
    event_time=event_time,
    covar=covar,
    verbose=True,
)
results.to_csv("aa_survival.txt", sep="\t", index=False)
```

Numeric covariates enter as numeric predictors. String/object and categorical
covariates are dummy encoded with the first level omitted. Convert numerically
coded categories explicitly, as shown for SEX, when they should not be treated
as continuous predictors. Models include genotype terms alongside the selected
covariates; hapy does not fit a separate estimated Cox intercept.

Change `kind` to `"HLA"` or `"SNP"` to analyse another loaded block.

## Time-varying covariates

`dynamic_covar` describes changes in predictor values over follow-up. It contains
`sample_id`, `time` and one or more covariate columns. Here, `time` is the change
time measured from the same origin and in the same unit as the endpoint times.
Repeated subject IDs are expected, but each subject/change-time pair must be
unique.

An illustrative update table is:

```text
sample_id	time	treatment
001	0	0
001	4	1
002	0	0
003	0	1
003	3	0
```

For subject 001, treatment is zero from time 0 to 4 and one thereafter until the
endpoint. Subject 002 keeps its baseline value throughout follow-up. The rows
contain the current covariate state, not increments to accumulate.

Every analysed subject must have a row at time zero with non-missing values for
all dynamic covariates. A subject missing any required baseline dynamic value
is excluded in its entirety. Later missing update cells mean no change to that
covariate; its previous value carries forward. Updates at or after the subject's
endpoint are ignored. Change times must be finite and nonnegative.

Load the full update table for your cohort and pass it to the same API:

```python
dynamic_covar = pd.read_csv(
    "dynamic_covariates.tsv", sep="\t", dtype={"sample_id": str},
)
time_varying_results = hp.stats.survival(
    hladat,
    famfile=famfile,
    config=config,
    kind="AA",
    event_time=event_time,
    covar=covar,
    dynamic_covar=dynamic_covar,
)
time_varying_results.to_csv("aa_survival_dynamic.txt", sep="\t", index=False)
```

Use distinct names for dynamic covariates, baseline covariates and genotype
predictors. Predictor names cannot use the reserved structural names
`sample_id`, `time`, `event`, `start` or `stop`.

hapy constructs intervals with `start < t <= stop`. The event indicator is set
only on the final interval for an observed event. The endpoint table still has
one row per subject; do not supply repeated events or preconstructed start–stop
rows through `event_time`.

## What is tested?

The genotype adapter determines the eligible genotype columns for each variant
or position, using its existing QC and reference selection.

- **One genotype column:** fit a covariate-adjusted Cox model and report the
  coefficient's Wald P-value and hazard ratio.
- **Multiple genotype columns:** compare the full model containing all genotype
  terms with a null model containing the same baseline and dynamic covariates.
  Report the omnibus partial-likelihood-ratio P-value for the genotype block.

Both models in an omnibus comparison use the same subjects and survival
observations. Compatible null summaries can be reused across positions; see
[null-model reuse](null_model_reuse.md). This optimisation does not change the
test or complete-case rules.

## Samples and missing information

For each position, genotype and baseline covariates are matched to endpoints
by subject ID. Subjects need an endpoint, available genotype data and complete
fitted predictors. Baseline/dynamic missingness can exclude subjects; no missing
values are imputed. Dynamic subjects are retained or excluded as whole people,
rather than retaining selected incomplete intervals.

`N_used`, `N_event` and `N_censored` count subjects, even when each subject has
multiple intervals. `N_intervals` separately records the total number of interval
rows in a time-varying model. Counts can differ across positions because of
genotype missingness.

## Reading the results

| Column | Meaning |
|---|---|
| `VARIANT` and genotype annotations | Tested identifier and available gene, amino acid position, residue/reference or allele metadata from the selected adapter. |
| `P-value` | Wald coefficient P-value for a one-column test, or omnibus partial-likelihood-ratio P-value for multiple genotype columns. |
| `HR` | Hazard ratio `exp(coef)` per unit increase in genotype dosage for a one-column test. HR > 1 indicates increased hazard; HR < 1 indicates decreased hazard, conditional on the fitted covariates. |
| `StdErr` | Standard error of the log-hazard coefficient, not the standard error of HR. |
| `CI-lower`, `CI-upper` | Lower and upper limits of the hazard ratio's 95% confidence interval. These are already exponentiated; do not exponentiate them again. |
| `N_used` | Final number of modelling subjects. |
| `N_event`, `N_censored` | Event and censored subject counts among those included. |
| `N_dropped_na` | Subjects removed after the endpoint/predictor merge because required modelling information was missing, including incomplete dynamic baseline values. Does not count every target IID lacking an endpoint. |
| `N_intervals` | Total start–stop rows for a time-varying model; absent for a static model. |
| `HLA_AF`, `MAF` | HLA allele frequency or SNP minor allele frequency for one-column tests, using included subjects once each, regardless of interval count. |
| `AA_AF` | AA frequency annotation from genotype preparation. Unlike standard association and interaction outputs, the current survival runner does not recompute this field after final survival complete-case filtering. |
| `INFO` | Last column; missing when explicit checks pass, otherwise a fit/skip reason. |

`HR`, `StdErr` and CI fields remain missing for omnibus tests. The direction of
a one-column effect follows the genotype dosage encoding. All tests are
covariate-adjusted associations; a fitted hazard ratio is not a survival
probability.

Possible `INFO` values include `insufficient variation`, `no complete-case
samples`, `no observed events`, `not converged`, `singular matrix`, `numerical
fitting error`, non-finite coefficients/standard errors/log-likelihood and
`feature not estimated`. Omnibus fit messages identify the null or full model.
Failed checks leave statistical estimates missing.

Lifelines warnings are not suppressed or automatically copied into `INFO`.
A missing `INFO` value means the implemented checks passed; proportional-hazards
assumptions and other model diagnostics are not checked automatically by this
API. Unexpected input or fitting exceptions can still stop the run.

## API and execution options

| Parameter | Type and default | Meaning |
|---|---|---|
| `hladat` | `HLAdata` | Loaded genotype object. |
| `famfile` | `pandas.DataFrame` | Required IID table; the survival endpoint is not read from `PHENO`. |
| `config` | `hp.stats.SurvivalConfig` | Parallel settings; Cox fitting is selected automatically. |
| `kind` | `"AA"`, `"HLA"` or `"SNP"` | Loaded block to analyse. |
| `event_time` | `pandas.DataFrame` | Exactly `sample_id`, `time` and `event`, one endpoint per subject. |
| `covar` | `DataFrame`, CSV path or `None` | IID-indexed baseline covariates. Default `None`. |
| `dynamic_covar` | `DataFrame` or `None` | Time-zero values and subsequent updates; default `None` fits static Cox. |
| `variant_filter` | callable or `None` | Select variants before testing. Reuse the ID/gene filter helpers shown in the association guide. |
| `verbose` | `bool`, `True` | Print execution settings, progress and timing. |
| `use_progress_bar` | `bool`, `False` | Use tqdm progress when available and verbose output is enabled. |
| `print_every` | `int`, `100` | Completed variants between periodic progress messages. |

`SurvivalConfig` accepts `n_jobs=1`, `backend="process"`, `batch_size=32` and
`chunksize=8` by default. A single worker runs sequentially. With multiple
process workers, put script execution inside `if __name__ == "__main__":`.
`backend="thread"` is also available. Parallel settings do not select a different
statistical model.

To select a gene, for example:

```python
selected_results = hp.stats.survival(
    hladat, famfile=famfile, config=config, kind="AA",
    event_time=event_time, covar=covar,
    variant_filter=hp.stats.filters.only_genes_variant(["DRB1"]),
)
```
