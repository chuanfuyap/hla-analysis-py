# Reading PLINK genotype data for interaction analysis

`hp.read_genotype()` reads PLINK PGEN, BED/BIM/FAM (bfile), or RAW input into a pandas
DataFrame instead of an `HLAdata` object. Sample IIDs form the index and original
variant IDs form the columns. Sample IDs remain strings, including leading zeros.

The PGEN file format is the preferred option among the three, as it preserves fractional dosages with the highest precision. The BFILE format is the least preferred, as it converts dosages into hard genotype calls.

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
| `dosage_allele` | `str` or `None` | `None` | `"alt"`, `"ref"` or RAW-only `"counted"`. `None` counts ALT for binary input and the header allele for RAW. Explicit RAW REF/ALT requests require PVAR. |

Variant selection happens **before binary dosage decoding**. For RAW it restricts
the genotype columns loaded numerically, while the text file is still scanned.
RAW selection uses variant IDs without their final `_COUNTED_ALLELE` suffix.
For example, a RAW column `chr7_37338247_T_C_C` is selected with
`variants=["chr7_37338247_T_C"]`. No genomic regions, patterns or automatic
frequency filters are applied by `variants`.

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

hapy aligns the partner matrix to phenotype sample IDs; missing required values
are handled by the existing interaction analysis complete-case processing.

The returned DataFrame is dense. For large filesets, select the partner variants
before decoding to limit memory use. Apply imputation quality and other desired
QC filters upstream; this reader performs no automatic QC filtering.

This function currently supports PLINK formats only. VCF support can be added
later through the same `read_genotype()` API; it is not implemented yet.
The initially introduced `read_pgen()` function has been renamed to
`read_genotype()`; update calls to use the new name.
