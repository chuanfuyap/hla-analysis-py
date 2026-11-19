# Amino acid position association analysis

The availability of HLA imputation methods and the ability to map HLA alleles to 
their corresponding amino acid sequences have made it possible to investigate 
genetic associations at the amino acid level. This provides an opportunity to move 
beyond conventional HLA allele associations and identify specific amino acid positions 
that may contribute to disease susceptibility.

However, unlike conventional genetic variants, HLA amino acid positions can contain 
multiple residues, requiring methods that account for their multi-allelic nature. 
To facilitate HLA fine-mapping, `hapy` provides a range of association testing 
functions designed specifically for analysing amino acid variation.

`hp.analyse()` runs association tests for one phenotype using the AA, HLA
or SNP block of an `HLAdata` object. It returns a pandas DataFrame; results are
not saved automatically.

The underlying linear and logistic regression models are fitted using the [statsmodels](https://www.statsmodels.org/stable/index.html) Python package, supporting both single-variant association tests and multi-allelic omnibus tests.

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
estimates. 

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
    residue=False,
    verbose=True,
    use_progress_bar=False,
    print_every=100,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hladat` | `HLAdata` | **Required** | Loaded genotype object. The block selected by `kind` must already be loaded. |
| `config` | `hp.stats.StandardConfig` | **Required** | Model type and parallel execution settings; see below. |
| `kind` | `str` | **Required** | Loaded genotype block: `"AA"`, `"HLA"` or `"SNP"`. |
| `famfile` | `pandas.DataFrame` or `None` | `None` | Sample table containing `IID` and, when `y` is omitted, `PHENO`. Required when `y` is an unindexed array. |
| `covar` | `pandas.DataFrame`, `str`, `pathlib.Path` or `None` | `None` | IID-indexed covariates, or a comma-separated CSV path with sample IDs in its first column. Read other delimiters explicitly with pandas. All supplied covariate columns enter the model. |
| `y` | `pandas.Series`, one-dimensional array-like or `None` | `None` | Phenotype Series indexed by IID, array aligned to FAM row order, or `None` to use `famfile["PHENO"]`. |
| `variant_filter` | callable or `None` | `None` | Function receiving a variant context and returning `True` to keep or `False` to exclude it. Use ID/gene filter helpers rather than a bare list. |
| `condition_on` | `str`, sequence of `str` or `None` | `None` | Same-kind focal variant IDs included as adjustment terms. Conditioning variants are excluded from focal testing. For AA, use position IDs without residue suffixes. |
| `residue` | `bool` | `False` | For `kind="AA"`, retain position-level results and append residue OVR tests, including an available QC-passing reference dosage. |
| `verbose` | `bool` | `True` | Print execution settings, task counts, progress and timing. |
| `use_progress_bar` | `bool` | `False` | Use tqdm when available and `verbose=True`; otherwise use periodic progress messages. |
| `print_every` | `int` | `100` | Completed analysis tasks between periodic progress messages. |

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

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_type` | `str` | **Required** | `"logit"` for binary outcomes or `"linear"` for continuous outcomes. |
| `n_jobs` | `int` | `1` | Number of workers. Use `1` for sequential execution. |
| `backend` | `str` | `"process"` | Parallel backend: `"process"` or `"thread"`. Used when `n_jobs > 1`. |
| `batch_size` | `int` | `32` | Number of analysis tasks per submitted batch. |
| `chunksize` | `int` | `8` | Maximum number of batch futures in flight, not the number of tasks per batch. |

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
of genotype columns determines the test:

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

The return value is a pandas DataFrame with position IDs such as
`AA_DPB1_69_33048646` in `VARIANT`. Use `VARIANT` to identify or join 
results. Entirely missing columns and some zero-valued housekeeping columns may be omitted.

| Column | Meaning |
|---|---|
| `VARIANT`, `GENE`, `AA_POS` | Variant identifier, HLA gene and amino acid position. |
| `P-value` | Association P-value. |
| `TEST` | `SINGLE`: single-residue test; `MULTI`: position-level omnibus test; `OVR`: residue one-vs-rest test. |
| `Statistic`, `DoF` | Test statistic and degrees of freedom. `MULTI` reports a likelihood-ratio statistic; `SINGLE` and `OVR` report a t-statistic (linear) or Wald z-statistic (logistic). |
| `Beta`, `StdErr` | Effect estimate and standard error. `MULTI` reports residue-specific effects relative to the reference (`A=value;G=value`), while `SINGLE` and `OVR` report individual effects. Logistic coefficients are log-odds; use `np.exp()` to obtain odds ratios. |
| `CI-lower`, `CI-upper` | 95% confidence interval for `SINGLE` and `OVR` tests. |
| `Ref_AA`, `Amino_Acids`, `AAcount` | Reference amino acid, available residues and allele count. For OVR tests, `Ref_AA` identifies the tested residue rather than a reference. |
| `AA_AF` | Amino acid frequencies in the analysed samples (`F=0.2;S=0.8`). OVR reports the tested residue frequency. |
| `HLA_AF`, `MAF` | HLA allele frequency or SNP minor allele frequency, where applicable. |
| `N_used`, `N_case`, `N_control` | Number of samples analysed and case/control counts for binary outcomes. |
| `INFO` | Diagnostic information for excluded or failed tests, such as insufficient variation or model convergence issues. |
| `COND_KIND`, `COND_VARIANTS`, `N_COND_VARIANTS` | Conditioning type, variant identifiers and number of conditioning variants, where applicable. |

Unexpected fitting exceptions can still stop analysis; `INFO` records the
explicit checks described above.

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

## Optional residue one-vs-rest tests

Use residue=True with hp.stats.analyse() and kind="AA" to perform individual residue association tests alongside the standard position-level tests.

```python
results = hp.stats.analyse(
    hladat, hp.stats.StandardConfig(model_type="logit", n_jobs=4),
    kind="AA", y=phenotype, covar=covar, residue=True,
)
position_results = results[results["TEST"].isin(["SINGLE", "MULTI"])]
residue_results = results[results["TEST"] == "OVR"]
```

For multiallelic positions, the original omnibus (MULTI) result is retained, with an additional one-vs-rest (OVR) test for each QC-passing residue. Each OVR test compares the tested residue against all other residues.

For biallelic positions, the existing SINGLE test already evaluates one residue, so only the opposite residue is added as an OVR result.

OVR rows use Ref_AA="tested: S" to identify the tested residue.

## Position-based stepwise conditional analysis

hp.stats.stepwise() performs stepwise conditional association analysis of amino acid positions. It identifies the most significant position, adds it to the conditioning list, and repeats the analysis until no further positions meet the significance threshold.

```python
winners = hp.stats.stepwise(
    hladat=hladat,
    config=cfg,                 # StandardConfig: linear or logit
    kind="AA",
    selection="position",
    y=pheno[2],                 # IID-indexed Series, or supply famfile
    covar=covar,
    condition_on=None,          # optional initial full AA position IDs
    significance_threshold=5e-6,
    max_steps=20,
    residue=False,
    verbose=True,
    use_progress_bar=False,
    print_every=10,
)
```

The function uses the existing single-variant and omnibus tests for biallelic and multiallelic positions, respectively. Previously selected positions are automatically included as covariates in subsequent scans.

Output

Returns a pandas DataFrame containing the most significant position from each scan, with additional columns:

- STEP: Scan number.
- SELECTED: Whether the position passed the significance threshold.

The final non-significant result is retained but not added to the conditioning list.

The selected positions, stopping reason, total runtime and other metadata are available through res.attrs.

### Continue from previous results

```python
continued = hp.stats.stepwise(
    hladat=hladat,
    y=pheno[2],
    config=cfg,
    kind="AA",
    covar=covar,
    condition_on=res.attrs["condition_on"],
    significance_threshold=1e-5,
)
```

This starts a new stepwise analysis conditional on all previously selected positions.
