# Reading PGEN data for interaction analysis

`hp.read_pgen()` reads a PLINK `.pgen`/`.pvar`/`.psam` fileset into a pandas
DataFrame instead of an `HLAdata` object. Sample IIDs form the index and original
variant IDs form the columns. Sample IDs remain strings, including leading zeros.

```python
import hapy as hp

# Accepts a fileset prefix or the .pgen filename.
partners = hp.read_pgen("test_data/chr7.pgen")

# Read only selected variants, in this order.
partners = hp.read_pgen("test_data/chr7", variants=["rs1", "rs2"])
```

Values count the PVAR ALT allele by default. Fractional imputed dosages are
preserved and missing genotypes are `NaN`. Use `dosage_allele="ref"` to count
REF instead. No HLA presence/absence reorientation, marker grouping, R2 filtering
or frequency filtering is performed. Selected variants must be biallelic with
known, distinct REF and ALT alleles; duplicate variant IDs or sample IIDs are
rejected. Dosages use PLINK's diploid 0–2 scale.

| Parameter | Type | Description |
|---|---|---|
| `fileloc` | `str` or `Path` | Fileset prefix or `.pgen` filename. |
| `pvarfileloc` | `str` or `Path`, optional | PVAR path; defaults to `<prefix>.pvar`. |
| `psamfileloc` | `str` or `Path`, optional | PSAM path; defaults to `<prefix>.psam`. |
| `variants` | iterable of strings or string, optional | Exact variant IDs to read. Defaults to all variants; unknown or repeated IDs raise an error. An empty list returns samples with no genotype columns. |
| `dosage_allele` | `"alt"` or `"ref"` | Counted allele; defaults to `"alt"`. |

## Using the DataFrame in interactions

With an existing `hladat`, phenotype Series `phenotype` indexed by IID, and
optional IID-indexed covariate DataFrame `covar`:

```python
partners = hp.read_pgen("test_data/chr7", variants=["rs1", "rs2"])

config = hp.stats.InteractionConfig(
    model_type="logit",
    mode="one_vs_block_omnibus",
)
results = hp.interaction(
    hladat,
    config,
    a_kind="AA",
    b_kind="DF",
    block_b_df=partners,
    y=phenotype,
    covar=covar,
)
```

This uses each partner column as an anchor for an AA-block omnibus interaction.
For individual genotype-column interactions use `mode="pairwise"` instead.
HAPY aligns the partner matrix to phenotype sample IDs; missing required values
are handled by the existing interaction analysis complete-case processing.

The returned DataFrame is dense. For large filesets, select the partner variants
before decoding to limit memory use. Apply imputation quality and other desired
QC filters upstream; this reader performs no automatic QC filtering.
