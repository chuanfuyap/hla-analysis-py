# Amino acid position association analysis

`hp.analyse()` runs association tests for one phenotype using the AA, HLA
or SNP block of an `HLAdata` object. It returns a pandas DataFrame; results are
not saved automatically.

This page focuses on amino acid position tests with `kind="AA"`. Input rows can
represent individual residues or residue encodings, but `hapy` groups them into
positions before testing. A position such as `AA_DRB1_11_32552129` represents
HLA-DRB1 amino acid position 11 at genomic coordinate 32552129, not one particular
residue. Coordinates alone do not identify the reference genome build.

## Included files

The example files are in `examples/association`:

| File | Contents |
|---|---|
| `example.raw` | PLINK RAW data: 18 samples, six metadata columns and nine AA marker columns spanning four AA positions. |
| `example_pheno.txt` | Whitespace-delimited phenotype table; sample IDs are the first column and `PHENO` is coded 1/2. |
| `example_covar.txt` | Matching sample IDs followed by `SEX`, `PC1`, `PC2`, `PC3`. |
| `run_association.py` | Runnable association example with an optional conditioning argument commented out. |

This is a small demonstration dataset, with 7 cases and 11 controls. Its P-values
and effect estimates illustrate software usage rather than stable population
estimates. Some fitted models may be unstable in such a small dataset; inspect
`INFO` alongside the results.

## Quick start

Install `hapy`, then run from the repository root:

```bash
python examples/association/run_association.py
```

The script writes `aa_association.csv` into the current working directory. Input
paths in the script are relative to the script itself.

The following example loads the same data and explicitly treats SEX as
categorical. Run it from the repository root:

```python
from pathlib import Path
import pandas as pd
import hapy as hp

base = Path("examples/association")
phenotype = pd.read_csv(
    base / "example_pheno.txt", sep=r"\s+", index_col=0, converters={0: str},
)
covariates = pd.read_csv(
    base / "example_covar.txt", sep=r"\s+", index_col=0, converters={0: str},
)
covariates["SEX"] = covariates["SEX"].astype("category")

hladat = hp.read_plink(
    base / "example.raw", format="raw", load_types=("AA",),
    dosage_allele="counted",
)
config = hp.stats.StandardConfig(model_type="logit", n_jobs=1)
results = hp.stats.analyse(
    hladat, config, kind="AA", y=phenotype["PHENO"], covar=covariates,
)
results.to_csv("aa_association.csv", index=False)
```

No PVAR or R² file accompanies this example. The RAW reader preserves counted
dosages without PVAR and prints a note that imputation-quality filtering was not
applied. `simpleQC=True` still applies the shared 0.5% allele-frequency filter
while preserving missing genotypes.

For imputed data, filter on imputation quality upstream or provide a suitable
quality field when reading the data:

```python
hladat = hp.read_plink(
    "cohort.raw", format="raw", pvarfileloc="cohort.pvar",
    pvar_R2_column="DR2", R2_minimum=0.8, load_types=("AA",),
)
```

Use the field actually present in your file; `DR2` is an example. An external
`filter_R2` file is another option. Reader filtering and the association model's
complete-case sample selection are separate steps.

## `analyse()` parameters

```python
results = hp.stats.analyse(
    hladat,
    config,
    kind="AA",
    famfile=None,
    covar=None,
    y=None,
    variant_filter=None,
    condition_on=None,
    verbose=True,
    use_progress_bar=False,
    print_every=100,
)
```

| Parameter | Type | Description |
|---|---|---|
| `hladat` | `HLAdata` | Loaded HLA genotype object. The block selected by `kind` must already be loaded. Use `read_plink()` or another genotype loader, such as `read_bgl()`. |
| `config` | `hp.stats.StandardConfig` | Model type and parallel execution settings; see below. |
| `kind` | `"AA"`, `"HLA"` or `"SNP"` | Required. Selects the genotype block to analyse. AA tests group markers by amino acid position; HLA and SNP tests use their corresponding variant IDs. |
| `famfile` | `pandas.DataFrame` or `None` | Optional FAM table containing `IID` and, if `y` is omitted, `PHENO`. Required when `y` is an unindexed array. |
| `covar` | `pandas.DataFrame`, `str`, `Path` or `None` | Optional IID-indexed covariates. A path is read as comma-separated CSV with the first column as the index. Load whitespace-delimited files with pandas first. All supplied covariate columns enter the model. |
| `y` | `pandas.Series`, one-dimensional array-like or `None` | Phenotype. A Series must be indexed by IID. Arrays follow FAM row order. If omitted, use `famfile["PHENO"]`. |
| `variant_filter` | callable or `None` | Function receiving a variant context dictionary and returning `True` to keep or `False` to exclude the variant. Use the supplied filter helpers for lists of IDs or genes; see below. |
| `condition_on` | `str`, sequence of strings or `None` | Variant IDs from the same selected genotype block to include as adjustment covariates. Conditioning variants are excluded from testing. For AA, supply position IDs without residue suffixes. |
| `verbose` | `bool`, default `True` | Print scheduled variant counts, execution settings, progress and timing. |
| `use_progress_bar` | `bool`, default `False` | Use a tqdm progress bar when available and `verbose=True`; otherwise use periodic progress messages. |
| `print_every` | `int`, default `100` | Number of completed variants between periodic progress messages. |

### `StandardConfig`

```python
config = hp.stats.StandardConfig(
    model_type="logit",
    n_jobs=1,
    backend="process",
    batch_size=32,
    chunksize=8,
)
```

| Parameter | Type | Description |
|---|---|---|
| `model_type` | `"logit"` or `"linear"` | Required. Logistic regression for binary outcomes, or ordinary least squares for continuous outcomes. |
| `n_jobs` | `int`, default `1` | Number of workers. Use 1 for sequential execution. |
| `backend` | `"process"` or `"thread"`, default `"process"` | Parallel execution method. Used when `n_jobs > 1`. |
| `batch_size` | `int`, default `32` | Variants submitted in each batch. Larger batches reduce scheduling overhead but can make work distribution less even. |
| `chunksize` | `int`, default `8` | Maximum number of batch futures in flight; this is not the number of variants in a batch. |

## Phenotypes, covariates and sample IDs

The simplest input is an IID-indexed phenotype Series and an optional
IID-indexed covariate DataFrame. Samples are matched by ID, not table row order.
IDs are converted to strings and must be unique and non-missing. Read IDs as
strings before pandas can discard leading zeros.

Without `famfile`, the phenotype Series defines the target sample IDs. With
`famfile`, its `IID` column defines the target IDs and a supplied Series is
aligned to them. An array-like phenotype instead follows FAM row order.

If the observed non-missing phenotype values are exactly `{1, 2}`, `hapy` recodes
1 to control (0) and 2 to case (1). This recoding happens during phenotype
preparation, before model selection. For logistic regression, otherwise supply
0/1 values and represent missing outcomes as `NaN` rather than numeric sentinel
values such as -9. Use `model_type="linear"` for continuous phenotypes.

Numeric covariates enter as continuous predictors. String/object and categorical
covariates are represented using categorical model terms. Convert numeric codes
such as SEX or treatment categories to `category` when they should be treated
as categories. Column names containing spaces or punctuation are quoted for
formula construction. An intercept is included automatically.

For each position, samples missing the phenotype, a fitted genotype column, a
covariate or a conditioning genotype are excluded. This can leave different
sample counts across tested positions. `hapy` does not impute these missing values.

## What is tested?

The genotype adapter constructs columns for the selected position. The number
of fitted genotype columns determines the test:

- **SINGLE**: one genotype column. Fit phenotype against genotype and covariates.
  The reported P-value is a two-sided coefficient test: a Wald z-test for
  logistic regression or a t-test for linear regression.
- **MULTI**: multiple genotype columns. Compare a full model containing all
  genotype columns with a null model containing only the same covariates and
  intercept. Report the omnibus likelihood-ratio P-value.

Conditioning genotype columns enter both models as covariates. For multi-residue
AA positions, one residue is omitted as the reference; reported coefficients
are relative to that reference. The omnibus P-value tests the genotype block
jointly, rather than choosing the smallest residue-specific P-value.

## Reading the results

`N_used` reports the final number of complete-case samples for each test.

The return value is a pandas DataFrame with position IDs such as
`AA_DPB1_69_33048646` in `VARIANT`. Use `VARIANT` to identify or join 
results. Entirely missing columns and some zero-valued housekeeping columns may be omitted.

| Column | Meaning |
|---|---|
| `VARIANT`, `GENE`, `AA_POS` | Position-level identifier, gene and amino acid sequence position. |
| `P-value` | P-value of either omnibus test or univariate amino acid position test. |
| `TEST` | **MULTI**: omnibus test performed on multi-allelic positions, **SINGLE**: bi-allelic positions, i.e. two amino acids, **OVR**: one vs rest for single amino acid test in multi-allelic positions. |
| `Statistic`, `DoF` | Test statistic and associated degrees of freedom. MULTI tests report the likelihood-ratio statistic; SINGLE tests report the t-statistic for linear regression or the Wald z-statistic for logistic regression. |
| `Beta`, `StdErr` | Association effect estimate and standard error. MULTI tests report residue-specific estimates relative to the reference amino acid, formatted as `A=value;G=value`. Logistic regression coefficients are on the log-odds scale (not odds ratios); apply `np.exp()` to each coefficient to obtain odds ratios. |
| `CI-lower`, `CI-upper` | Lower and upper limits of the 95% confidence interval, reported for SINGLE tests only. Their direction depends on the counted allele encoding. For logistic regression, apply `np.exp()` to these limits to obtain the 95% confidence interval for the odds ratio. |
| `Ref_AA`, `Amino_Acids`, `AAcount` | Reference amino acid, available residue labels and allele count reported by the AA adapter. Labels reflect the input encoding. |
| `INFO` | Blank or missing for successful tests. `insufficient variation` identifies AA positions with no eligible genotype columns after sample subsetting and allele-frequency processing. Fit diagnostics can instead report `not converged`, `non-finite coefficients`, `non-finite standard errors` or `non-finite log-likelihood`. MULTI diagnostics identify the null or full model. Failed checks leave test estimates missing. |
| `COND_KIND`, `COND_VARIANTS`, `N_COND_VARIANTS` | Selected block kind, semicolon-separated conditioning variant IDs and number of conditioning variants. Present when applicable. |

Model warnings are not suppressed or automatically copied into `INFO`.
Unexpected fitting exceptions can still stop analysis; `INFO` records the
explicit checks described above.

For survival analysis, the effect columns have a different interpretation:

| Column | Meaning |
|---|---|
| `HR`, `StdErr`, `CI-lower`, `CI-upper` | Hazard ratio, standard error of the log-hazard coefficient and lower and upper limits of the 95% confidence interval for the hazard ratio. Reported for SINGLE tests only. HR > 1 indicates increased hazard and HR < 1 indicates decreased hazard per unit increase in genotype dosage. |

Survival results also keep `INFO` as the last column. It is missing when the
explicit checks pass. Reasons include `not converged` for a lifelines
`ConvergenceError`, `singular matrix`, `numerical fitting error`, non-finite
coefficients/standard errors/log-likelihood, and `feature not estimated`.
Multi-column Cox tests identify the `null model` or `full model` in fit messages.
No usable genotype columns, no complete-case samples or no observed events are
reported before fitting. Affected rows retain their sample counts and missing
estimates, allowing other variants to continue.

Lifelines does not provide the same public convergence flag as the standard
logistic models. Warning-only convergence concerns remain visible as warnings
and are not copied into `INFO`; a missing `INFO` means the explicit checks
passed, not that every convergence concern has been ruled out. Invalid input or
unexpected programming errors can still stop execution.

## Conditional association

Adjust for one or more positions from the same AA block:

```python
conditional = hp.stats.analyse(
    hladat, config, kind="AA", y=phenotype["PHENO"], covar=covariates,
    condition_on="AA_DRB1_231_32548054",
)
```

The conditioning position is added to the model as covariates and excluded from
the tested positions. This asks about associations after adjustment; it is not
an interaction test. Multiple positions can be supplied as a list. Conditioning
uses the full eligible genotype block, not one manually selected residue.

Conditioning variants must exist in the selected `kind` block and retain eligible
genotype columns. Missing or unusable conditioning variants raise an error.
`variant_filter` selects tested variants; it does not remove the requested
conditioning adjustment. For HLA alleles, use `kind="HLA"` and an allele ID in
that block rather than an AA position ID.

## Variant filtering

Use a helper to select position IDs:

```python
variant_filter = hp.stats.filters.only_variants_variant([
    "AA_DRB1_11_32552129",
    "AA_DRB1_13_32552123",
])
results = hp.stats.analyse(
    hladat, config, kind="AA", y=phenotype["PHENO"], covar=covariates,
    variant_filter=variant_filter,
)
```

These IDs illustrate the format; use IDs present in your loaded data. Available
AA position IDs can be inspected with `hladat.AA.info["AA_ID"].unique()`.
An ID list is passed to the helper, not directly as `variant_filter`.

Or select HLA genes:

```python
variant_filter = hp.stats.filters.only_genes_variant(["DRB1", "DQB1"])
results = hp.stats.analyse(
    hladat, config, kind="AA", y=phenotype["PHENO"], covar=covariates,
    variant_filter=variant_filter,
)
```

These filters select variants to test; they do not change the loaded genotype
object or apply an additional frequency threshold. Gene filters need gene
annotations and are primarily useful for AA/HLA blocks.

In `analyse()`, filtering occurs before genotype construction with an empty
`geno_cols` list, then again for retained tested variants after construction.
Use stable ID/gene rules for selection, rather than relying on genotype columns
being available or on a callback being called exactly once.

## Parallel execution

For parallel execution in Jupyter notebooks, use threads:

```python
config = hp.stats.StandardConfig(
    model_type="logit", n_jobs=4, backend="thread", batch_size=32,
)
```

Process workers require importable callbacks and a script entry-point guard
(`if __name__ == "__main__":`). Define custom callbacks at module level when
using processes. Increasing workers does not guarantee faster execution; model
size, scheduling overhead and numerical-library threads affect performance.

## Saving and further analysis

```python
results.to_csv("output.txt", sep="\t", index=False)

# Select rows with available P-values and sort them.
ranked = results.dropna(subset=["P-value"]).sort_values("P-value")
```

Parse labelled MULTI coefficients before numerical calculations; do not apply
`np.exp()` directly to a whole mixed numeric/string `Beta` column. The omnibus
P-value remains a joint test, even when individual coefficients are exported.

F