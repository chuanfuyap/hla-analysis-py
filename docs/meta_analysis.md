# Meta-analysis

Sharing individual-level genetic data is not always possible due to privacy, 
ethical and data access restrictions. However, summary statistics from association 
analyses can often be shared, allowing results from multiple studies or cohorts 
to be combined through meta-analysis. By integrating evidence across datasets, 
meta-analysis can increase statistical power and improve the detection of genetic
associations.

To facilitate this, `hapy` provides functions for combining association results 
across studies or cohorts using summary statistics.

## `meta_analyse()`

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

## AA position routing

For amino acid positions, `hapy` automatically selects the meta-analysis method based on the tests reported across cohorts:

| Tests across cohorts | Method |
|---|---|
| All `SINGLE` | Inverse-variance weighting (IVW) of effect estimates. |
| Any `MULTI` | Weighted Stouffer's method combining `SINGLE` and `MULTI` P-values. |
| `OVR` | Excluded from position-level meta-analysis. |

For example, if a position is `SINGLE` in two cohorts and `MULTI` in another, all three P-values are combined using Stouffer's method. This tests overall association at the position rather than a common residue effect.

Only valid results contribute to the meta-analysis. Missing P-values or effect estimates are excluded, and duplicate position-level results within a cohort are not permitted. A `MULTI` result still determines the method even when its P-value is missing.

For IVW, effect estimates must represent the same allele or residue contrast across cohorts, as `hapy` does not automatically align them. Cohorts must be independent.

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

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fileloc` | `str` or `pathlib.Path` | **Required** | Text file listing one independent cohort result file per nonblank line. |
| `data_type` | `str` | **Required** | Result type: `"AA"`, `"HLA"` or `"SNP"`. HLA/SNP effects use IVW. |
| `beta` | `str` | `"Beta"` | Scalar effect column for IVW. Not required for positions routed only to Stouffer. |
| `se` | `str` | `"StdErr"` | Standard-error column for IVW. Not reconstructed from confidence intervals. |
| `pval` | `str` | `"P-value"` | P-value column for Stouffer, requiring `0 < P <= 1`. |
| `variant_col` | `str` | `"VARIANT"` | Input position/variant identifier. Also names the standard meta-analysis output key. |
| `prefix` | `str` | `"META"` | Prefix for combined statistical output columns. |
| `interaction` | `bool` | `False` | Read interaction results. AA inputs must be position-level omnibus results; HLA/SNP pairwise effects use IVW. |
| `test_col` | `str` | `"TEST"` | AA standard test-label column: `SINGLE`, `MULTI` or `OVR`. Not required for HLA/SNP or interaction tables. |
| `n_col` | `str` or `None` | `"N_used"` | Modelling sample-count column. `None` disables this count. |
| `case_col` | `str` or `None` | `"N_case"` | Case-count column for effective sample-size weights. `None` disables this count. |
| `control_col` | `str` or `None` | `"N_control"` | Control-count column for effective sample-size weights. `None` disables this count. |
| `partner_col` | `str` | `"Partner_col"` | Partner identifier column for interaction tables. |
| `feature_col` | `str` or `None` | `"Variant_col"` | Optional encoded feature for HLA/SNP pairwise interaction keys. `None` disables it. |
| `sep` | `str` | `","` | Delimiter in cohort result files. Use `"\t"` for TSV; the path-list file always uses one path per line. |

Optional count columns may be absent. Set a count parameter to `None` to disable it. The selected counts are mapped internally to the weighting fields; unrelated existing count columns do not override that selection.

## ## Weights and outputs

- **Inverse-variance weighting (IVW):** Combines effect estimates using inverse-variance weights (`1/SE²`), reporting a combined coefficient, standard error and two-sided P-value.
- **Stouffer's method:** Combines P-values using [`scipy.stats.combine_pvalues()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html) with sample-size weights. P-values are transformed into Z-scores using `norm.isf(P)`, without incorporating effect direction.

For Stouffer's method, weights are calculated as:

- **Case-control studies:** `sqrt(N_eff)`, where `N_eff = 4 / (1/N_case + 1/N_control)`.
- **Other studies:** `sqrt(N_used)`.
- **Missing sample counts:** Equal weight (`1`).

Both methods report the number of contributing studies (`N studies`), available sample totals (`Total N`) and combined P-values. Sample totals are reported as missing if counts are unavailable for any contributing cohort.

`META_log10P` reports `-log10(P)`, retaining very small P-values that may otherwise underflow to zero.

## Interaction tables

AA interaction meta-analysis continues to accept position-level omnibus results, using `VARIANT` and `Partner_col` to build the `INTERACTION` key. AA pairwise/residue coefficients are not accepted by this position-level route. HLA/SNP pairwise interactions use the selected `Beta`/`StdErr` columns for IVW; an available `Variant_col` distinguishes encoded features. 
