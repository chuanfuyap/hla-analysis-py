# Meta-analysis

`hp.meta.meta_analyse()` combines independent cohort result tables and returns `(ivw, stouffer)`. Each argument is a pandas DataFrame, or `None` when there are no applicable inputs for that method. A result requires at least two contributing cohorts for the position or interaction.

Create a text file with one cohort results path per line:

```text
/path/to/study_a.csv
/path/to/study_b.csv
/path/to/study_c.csv
```

Then run:

```python
ivw, stouffer = hp.meta.meta_analyse("studies.txt", data_type="AA")
```

The default input columns match current association output: `VARIANT`, `TEST`, `Beta`, `StdErr`, `P-value`, `N_used`, `N_case` and `N_control`. Legacy column names are not detected automatically, and standard errors are not reconstructed from confidence intervals.

## AA position routing

The reader checks test labels across all cohorts before selecting the meta-analysis method for each position.

| Tests found for a position | Method |
| --- | --- |
| All position-level rows are `SINGLE` | Inverse-variance meta-analysis of scalar `Beta` and `StdErr`. |
| Any position-level row is `MULTI` | Weighted Stouffer using every available `SINGLE` or `MULTI` P-value at that position. |
| `OVR` | Excluded from position-level meta-analysis. |
| Missing test label | Omitted, including unlabelled skipped rows. |

For example, if a position is `SINGLE` in studies A and B and `MULTI` in study C, all three available position-level P-values contribute to one Stouffer result. The position is excluded from IVW. The combined result describes evidence for association at the position, rather than a common residue coefficient.

A row labelled `MULTI` still determines routing when its P-value is missing, but that row does not contribute to the combined statistic, study count or sample totals. Missing IVW coefficients or standard errors are also omitted. Duplicate position-level rows within a cohort raise an error after OVR and unlabelled rows are excluded; the reader does not choose one or count them as independent cohorts.

For IVW, effect encodings must describe the same contrast across cohorts. The reader does not realign residues or counted alleles. Cohorts must be independent.

## Custom column names and delimiters

All input roles can be named explicitly:

```python
ivw, stouffer = hp.meta.meta_analyse(
    "studies.txt", data_type="AA",
    variant_col="position_id",
    test_col="test_type",
    beta="coefficient",
    se="standard_error",
    pval="pvalue",
    n_col="sample_size",
    case_col="cases",
    control_col="controls",
    sep="\t",  # tab-separated cohort tables
    prefix="META",
)
```

Use the same column mapping and delimiter for all cohort tables in a call. `sep=","` is the default. The list of paths itself always uses one path per line. `variant_col` also names the identifier column in standard meta-analysis output.

| Parameter | Type and default | Meaning |
| --- | --- | --- |
| `fileloc` | `str` or `Path` | File listing the independent cohort result files. |
| `data_type` | `"AA"`, `"HLA"` or `"SNP"` | Genotype result type. HLA/SNP effects use IVW. |
| `beta`, `se` | `str`: `"Beta"`, `"StdErr"` | Scalar effect and standard-error columns for IVW. Not required for positions routed only to Stouffer. |
| `pval` | `str`: `"P-value"` | P-value column for Stouffer, requiring `0 < P <= 1`. |
| `variant_col` | `str`: `"VARIANT"` | Position or variant identifier. |
| `test_col` | `str`: `"TEST"` | AA standard test labels: `SINGLE`, `MULTI`, `OVR`. Not required for HLA/SNP or interaction tables. |
| `n_col` | `str` or `None`: `"N_used"` | Actual modelling sample-count column. |
| `case_col`, `control_col` | `str` or `None`: `"N_case"`, `"N_control"` | Case/control counts for effective sample-size weights. |
| `interaction` | `bool`: `False` | Read interaction results. |
| `partner_col` | `str`: `"Partner_col"` | Partner identifier for interaction tables. |
| `feature_col` | `str` or `None`: `"Variant_col"` | Optional encoded feature for HLA/SNP pairwise interaction keys. |
| `prefix` | `str`: `"META"` | Prefix for combined statistical output columns. |
| `sep` | `str`: `","` | Delimiter in cohort result files. |

Optional count columns may be absent. Set a count parameter to `None` to disable it. The selected counts are mapped internally to the weighting fields; unrelated existing count columns do not override that selection.

## Weights and outputs

- IVW uses inverse squared standard errors and reports a combined coefficient, standard error and two-sided P-value.
- Stouffer uses `z = norm.isf(P)` and sample-size weights. Valid positive case/control counts use `sqrt(N_eff)`, where `N_eff = 4 / (1/N_case + 1/N_control)`. Otherwise, a positive selected sample count uses `sqrt(N_used)`. Without usable counts, the weight is one.

Stouffer combines evidence without imposing a residue effect direction. Its transform accepts valid cohort P-values from both SINGLE and MULTI position tests; it does not divide them by two. Weights should be independent of the observed P-values.

Both outputs include `N studies` and available cohort totals. `Total N` sums the selected modelling sample counts for contributing rows. Totals are missing when a contributing cohort lacks the corresponding count, rather than reporting a partial sum as complete. Combined `META_log10P` is minus log10(P) and remains available when the combined floating-point P-value underflows. Exact zero input P-values remain unsupported.

## Interaction tables

AA interaction meta-analysis continues to accept position-level omnibus results, using `VARIANT` and `Partner_col` to build the `INTERACTION` key. AA pairwise/residue coefficients are not accepted by this position-level route. HLA/SNP pairwise interactions use the selected `Beta`/`StdErr` columns for IVW; an available `Variant_col` distinguishes encoded features. There is no automatic detection of historical interaction column names: supply custom names explicitly when needed.
