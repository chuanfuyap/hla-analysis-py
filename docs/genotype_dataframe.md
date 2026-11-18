# Reading PLINK genotype data for interaction analysis

Genetic analyses often require integrating data from different sources, particularly 
when investigating interactions between HLA amino acid variation and genetic variants
elsewhere in the genome. While genotype data are commonly stored in PLINK formats, 
these files need to be converted into a suitable structure for downstream analyses.

To simplify this process, `hapy` provides `read_genotype`, a utility function 
that reads PLINK genotype files and converts them into pandas DataFrames, allowing them 
to be readily used in genetic interaction analyses and other downstream applications.

## `read_genotype()`

`hp.read_genotype()` reads PLINK PGEN, BED/BIM/FAM (bfile), or RAW input into a pandas
DataFrame instead of an `HLAdata` object. Sample IIDs form the index and original
variant IDs form the columns. Sample IDs remain strings, including leading zeros.

The PGEN file format is the preferred option among the three, as it preserves fractional 
dosages with the highest precision. The BFILE format is the least preferred, as it 
converts dosages into hard genotype calls.

For RAW input, the returned DataFrame column names have the final
`_COUNTED_ALLELE` suffix removed: `rs123_A` becomes `rs123`, and
`chr7_37338247_T_C_C` becomes `chr7_37338247_T_C`. Everything before the final
underscore is preserved. Use these returned variant IDs in `variants=[...]`.
PGEN and bfile column names use the IDs in PVAR and BIM unchanged.

```python
import hapy as hp

# Accepts a fileset prefix or the genotype filename.
partners = hp.read_genotype("test_data/chr7.pgen", format="pgen")
partners = hp.read_genotype("test_data/chr7.bed", format="bfile")
partners = hp.read_genotype("test_data/chr7.raw", format="raw")

# Read only selected variants, in this order.
partners = hp.read_genotype("test_data/chr7", variants=["rs1", "rs2"])
```

PGEN and bfile count ALT by default (BIM A1 for bfile). RAW preserves the
counted allele named by each column's final suffix by default, even when PVAR
is available. This means differently oriented RAW and binary files may return
complementary dosages by default. To align them, use `dosage_allele="alt"` or
`"ref"` explicitly; RAW requires PVAR for these requests. `"counted"` is RAW-only.
No HLA presence/absence reorientation is performed.

Fractional dosages are preserved and missing genotypes are `NaN`. PGEN requires
PVAR and PSAM; bfile requires BIM and FAM. RAW uses a matching PVAR if it exists,
but works without one. Selected variants with metadata must be biallelic with
known, distinct REF and ALT alleles. Duplicate variant IDs or sample IIDs are
rejected. Dosages use PLINK's diploid 0–2 scale.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fileloc` | `str` or `pathlib.Path` | **Required** | Fileset prefix or `.pgen`, `.bed` or `.raw` filename. |
| `format` | `str` | `"pgen"` | PLINK input format: `"pgen"`, `"bfile"` or `"raw"`. Specify explicitly for bfile and RAW. |
| `pvarfileloc` | `str`, `pathlib.Path` or `None` | `None` | PVAR path for PGEN or RAW. `None` uses `<prefix>.pvar` when available; PGEN requires it, RAW does not. Not used for bfile. |
| `psamfileloc` | `str`, `pathlib.Path` or `None` | `None` | PGEN-only PSAM path. `None` uses `<prefix>.psam`. |
| `variants` | `str`, iterable of `str` or `None` | `None` | Exact variant IDs in requested order. `None` reads all variants; `[]` returns samples with no genotype columns. Unknown or repeated IDs raise an error. |
| `pvar_R2_column` | `str` or `None` | `None` | PVAR INFO key or column containing imputation R². Requires PVAR; not available for bfile. |
| `filter_R2` | `str`, `pathlib.Path`, `pandas.DataFrame` or `None` | `None` | Headerless variant-ID/R² file, or a variant-indexed DataFrame with scores in integer column `1`. Available for all formats. |
| `R2_minimum` | `float` | `0.5` | Retain R² strictly greater than this cutoff when a score source is supplied. |
| `dosage_allele` | `str` or `None` | `None` | `"alt"`, `"ref"` or RAW-only `"counted"`. `None` counts ALT for binary input and the header allele for RAW. Explicit RAW REF/ALT requests require PVAR. |

### Optional imputation R² filtering

```python
# Read R² from the PVAR INFO field (use the key in your file, e.g. DR2 or R2).
partners = hp.read_genotype(
    "test_data/chr7", pvar_R2_column="DR2", R2_minimum=0.8,
)

# Alternatively, supply a two-column, headerless variant-ID/R² file.
partners = hp.read_genotype(
    "test_data/chr7.bed", format="bfile",
    filter_R2="test_data/chr7.r2", R2_minimum=0.8,
)
```

Use either `pvar_R2_column` or `filter_R2`, not both. Filtering is disabled
unless a score source is supplied; setting `R2_minimum` alone does not enable it.


## Using the DataFrame in interactions

With an existing `hladat`, phenotype Series `phenotype` indexed by IID, and
optional IID-indexed covariate DataFrame `covar`:

```python
partners = hp.read_genotype("test_data/chr7", variants=["rs1", "rs2"])

config = hp.stats.InteractionConfig(
    model_type="logit",
    mode="one_vs_block_omnibus",
)
results = hp.interaction(
    hladat,
    config,
    kind="AA",
    partner_kind="DF",
    partner_data=partners,
    y=phenotype,
    covar=covar,
)
```

`hapy` aligns the partner matrix to phenotype sample IDs; missing required values
are handled by the existing interaction analysis complete-case processing.

For large genotype files, use `variants=[...]` to select specific variants and reduce memory usage. Apply imputation quality and other QC filters beforehand, as `read_genotype()` does not perform automatic QC filtering.