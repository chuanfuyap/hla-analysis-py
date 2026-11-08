# Amino acid position association analysis

## Included files

| File | Contents |
|---|---|
| `example.raw` | PLINK RAW data: 18 samples, six metadata columns, nine AA marker columns spanning four AA positions. |
| `example_pheno.txt` | Whitespace-delimited phenotype table; sample IDs are the first column and `PHENO` is coded 1/2. |
| `example_covar.txt` | Matching sample IDs followed by `SEX`, `PC1`, `PC2`, `PC3`. |
| `run_association.py` | Simple runnable example with `condition_on` commented out but can be used when uncommented |

The current phenotype/covariate files leave the first (sample-ID) header unnamed;
`pd.read_csv(..., sep=r"\s+", index_col=0)` reads that column as the index.

This is a small demonstration dataset, with 7 cases and 11 controls.
Its p-values and effect estimates illustrate software usage rather than stable
population estimates.

## Quick start

Install `hapy`, then run from the repository root:

```bash
python examples/association/run_association.py 
```

The script writes `./aa_association.csv`.
The default is sequential logistic regression, adjusting for categorical SEX and
numeric PC1–PC3. With the supplied files, all 18 samples enter four position tests.
The default input paths are relative to the script, not your current directory.

## Walk through the Python API

This is similar to `run_association.py`

```python
from pathlib import Path
import pandas as pd
import hapy as hp

base = Path("examples/association")
phenotype = pd.read_csv(base / "example_pheno.txt", sep=r"\s+", index_col=0)
covariates = pd.read_csv(base / "example_covar.txt", sep=r"\s+", index_col=0)
phenotype.index = phenotype.index.astype(str)
covariates.index = covariates.index.astype(str)
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

### Allele orientation and imputation quality

No PVAR or R² file accompanies this example.
The purpose of this RAW-only example is to exercise the pipeline.

For real data, adding PVAR data is highly recommended to have info score for imputation quality filtering. 

```python
hladat = hp.read_plink(
    "cohort.raw", format="raw", pvarfileloc="cohort.pvar",
    dosage_allele="presence", pvar_R2_column="DR2", R2_minimum=0.8,
    load_types=("AA",),
)
```

Use the imputation-quality field actually present in your file; DR2 is an example.
An external `filter_R2` file is another option. No R² filtering is performed on the
bundled files; the reader prints a note. `simpleQC=True` still applies the shared
0.5% allele-frequency filter while preserving missing genotypes. 

## Model and sample handling

`model_type="logit"` is appropriate for this binary phenotype. HAPY automatically
recodes PLINK phenotype values `{1, 2}` to `{0, 1}`: 1 is control and 2 is case.
For a continuous phenotype in your own data, use `StandardConfig(model_type="linear")` 
with that phenotype, rather than treating the bundled binary outcome as continuous.

Pass `y` as an IID-indexed Series. Arrays require a FAM table whose row order
matches the array. `famfile` is unnecessary when supplying the indexed Series.
Covariates should be an IID-indexed DataFrame. For the bundled whitespace-delimited
files, load them explicitly with `pandas` rather than passing their paths to the
CSV-oriented covariate loader.

Per position, rows missing phenotype, genotype or a model covariate are excluded.
Output QC fields include `N_total`, `N_used`, `N_missing_Y`, `N_dropped_other`,
`N_case` and `N_control`. Zero-valued housekeeping columns may be omitted by the
runner; `N_used` remains available. The example loader additionally requires all
three sample-ID sets to match, to make accidental input mismatches obvious.

## Reading the results

Input marker rows describe individual residues or residue encodings. The AA adapter
groups these by position into output IDs such as `AA_DPB1_69_33048646`.

| Column | Meaning |
|---|---|
| `VARIANT`, `GENE`, `AA_POS` | Position-level identifier, gene and amino acid sequence position. |
| `LR_p`, `LR_stat`, `DoF` | Omnibus likelihood-ratio comparison for a position with multiple fitted genotype columns. |
| `Uni_p`, `Uni_Coef`, `Uni_StdErr`, `CI_0.025`, `CI_0.975` | Single-column association result and coefficient uncertainty. Logistic coefficients are log-odds coefficients; their direction refers to the counted encoding. |
| `LRp_Unip` | `LR_p` when available, otherwise `Uni_p`; a convenient plotting column, not a combination of tests. |
| `AA_AF`, `Ref_AA`, `Amino_Acids`, `AAcount` | Adapter annotations; biological interpretation depends on correct dosage orientation. |

For a multi-column position, there is no single overall signed effect coefficient;
use its omnibus P-value. Missing `Uni_p` or `LR_p` is expected when the other test
applies. The example does not choose the smaller of those P-values. Preserve full
precision when saving coefficients and standard errors for later meta-analysis.

## Conditional association

Adjust for one or more positions from the same AA block:

```python
conditional = hp.stats.analyse(
    hladat, config, kind="AA", y=phenotype["PHENO"], covar=covariates,
    condition_on="AA_DRB1_231_32548054",
)
```

The conditioning position is added to the model as covariates and excluded from
the tested positions. This asks about associations after adjustment; it is not an
interaction test. It changes the fitted model and can change p-values. Multiple
positions can be supplied as a list. Equivalent script invocation:


## Parallel execution

"Threads" are the option to use in jupyter notebooks when parallel execution is used.
"Process" workers require importable callbacks and a script entry-point guard (`if __name__ == "__main__":`), meaning when using a `.py` file.

## Variant Filtering

Two options are available, either on specific variant, or amino position, e.g.

```python
variant_filter=hp.stats.filters.only_variants_variant([
    "AA_DRB1_11_32552129",
    "AA_DRB1_13_32552123",
])

res = hp.stats.analyse_many(
    hladat=hladat,
    phenotypes=pheno,
    config=cfg,
    kind="AA",
    covar=covar,
    variant_filter=variant_filter,
)
```

Or filtering on specific HLA genes, e.g.

```python
variant_filter=hp.stats.filters.only_genes_variant([
    "DRB1", "DQB1",
])


res = hp.stats.analyse_many(
    hladat=hladat,
    phenotypes=pheno,
    config=cfg,
    kind="AA",
    covar=covar,
    variant_filter=variant_filter,
)
```