# hla-analysis-python (hapy)

A Python framework for HLA allele, amino-acid, SNP, interaction, survival, and meta-analysis workflows in immunogenetic association studies.

## Overview

hapy is a Python package for downstream analysis of HLA imputation outputs from genetic association studies. The framework supports analysis of classical HLA alleles, amino-acid polymorphisms, and SNP markers across the HLA region using a unified workflow interface.

Currently supported input formats include:
- SNP2HLA output
- Michigan Imputation Server HLA output
- PLINK-derived genotype formats
- Beagle-derived genotype format

## Features

- Amino-acid omnibus association testing
- Classical HLA allele association analysis
- SNP-level association analysis
- Covariate-adjusted regression models
- HLA variant conditional analysis
- Gene/Variant filtering
- MAF filtering
- Interaction analysis
- Survival analysis
- Cohort-level meta-analysis using Stouffer’s method and inverse-variance weighted method
- Parallelised association testing workflows

Supported models include:
- Linear regression
- Logistic regression
- Cox proportional hazards models

## Installation

Clone the repository and install locally:

It is recommended to install hapy within a clean conda environment.

```bash
conda create -n hapy python=3.11
conda activate hapy
pip install .
```

## Quick example

### PLINK input

> When using imputed data from Michigan server, you can convert to PLINK with PLINK2 which can be read by hapy. It is preferable to use pgen as it would retain the imputation accuracy as well as fratctional dosage values.
>
> If .raw file (exported by PLINK with `--export A`) is preferred, try to retain pgen's pvarfile to have imputation accuracy. 
>
> bfile format is also implemented but should be avoided as it loses fractional dosage value, giving only 0/1/2 counts and imputation accuracy information is not maintained. 

Use `read_plink` to choose a PLINK input format:

```python
import hapy as hp

# Text export plus its variant metadata.
hladat = hp.read_plink("cohort.raw", format="raw", pvarfileloc="cohort.pvar")

# Binary PLINK 2 fileset: cohort.pgen, cohort.pvar, cohort.psam.
hladat = hp.read_plink("cohort", format="pgen")

# Binary PLINK 1 fileset: cohort.bed, cohort.bim, cohort.fam.
hladat = hp.read_plink("cohort", format="bfile")
```

The `pgenlib` reader for both binary formats is installed automatically with
`pip install .` from the cloned repository. A PGEN or BED filename can
also be supplied instead of its prefix. The corresponding companion files must
share that prefix. Plain-text PVAR/BIM/PSAM/FAM companion files are supported.

By default, `dosage_allele="presence"` counts **P** for HLA/AA A/P pairs or
**T** for HLA/AA A/T pairs. Other HLA/AA allele pairs (e.g. A/G or G/T) count
the labelled ALT allele, as do all SNPs including `SNP_`/`SNPS_` markers.
The reader preserves allele labels; it does not infer biological amino-acid
identities from nucleotide or other surrogate codes. SNP IDs do not need a particular prefix: `rs` IDs,
chromosome-position IDs, and custom names are preserved in full. Only `HLA_`/
`HLA*` and `AA_` identify the special HLA/AA marker types. `load_types` skips
excluded marker types before allele
validation and dosage decoding. The RAW header identifies
the counted allele: the reader applies `2 - dosage` when that is the opposite
of the requested allele. Thus the existing HLA absence-to-presence conversion
is retained, while a RAW export already counting presence is left as-is.
Use `dosage_allele="ref"` or `"alt"` to request those allele counts explicitly;
for BIM files these mean A2 and A1, respectively. Returned `info.alleleA` records
the allele counted by the dosage, and `info.alleleB` the other allele.

SNP markers starting with `SNP_` or `SNPS_` are stored in `hladat.SNP.extradata`,
with their metadata in `hladat.SNP.extrainfo`. Other SNPs are stored in
`hladat.SNP.data` and `hladat.SNP.info`. The two genotype tables do not overlap.

PLINK formats use the shared `_load_and_process` pipeline for R2 filtering,
`simpleQC`, sample/variant summaries, and timing. The default 0.5% MAF filter
runs once through the shared `HLAdata.maf_filter`, covering primary data and
SNP `extradata` with aligned metadata. PLINK QC preserves missing dosages and
calculates frequency from observed samples. Set `simpleQC=False` to skip MAF
filtering and keep monomorphic/rare variants; R2 filtering still applies. Use
`pvar_R2_column="DR2"` for an imputation score in PVAR INFO (RAW/PGEN), or
`filter_R2="cohort.r2"` for an external score file (all formats). These two
score sources are mutually exclusive.

This first binary reader supports biallelic variants and diploid-scaled dosages,
with unique sample IIDs. PGEN imputation dosages are retained; BED contains
hardcalls. Multiallelic variants must be split upstream. The existing
`read_plinkraw(...)` API retains its original behaviour for compatibility.


### Beagle input

```python
import hapy as hp

# Load SNP2HLA or Michigan HLA imputation output in Beagle format
hladat = hp.read_bgl(
    fileloc="example.bgl.phased",
    load_types=("HLA","AA"), # loads on HLA allele and Amino Acid data
    filter_R2="example.bgl.r2", R2_minimum=0.9) # input imputation confidence and filter
)

# load pheno and covariate file with pandas (be sure the index have same ID, these are sample IDs )
pheno = pd.read_csv("example_pheno.txt",index_col=1, )
covar = pd.read_csv("example_covar.txt", sep="\s+", index_col=1, )


# Run amino-acid omnibus association analysis for binary phenotype, with 8 threads, 
config = hp.stats.StandardConfig(model_type="logit", n_jobs=8, backend="thread",)
results = hp.stats.analyse(
    hladat=hladat,
    y=pheno[2], # if using y, it is a pandas Series format
    config=config,
    kind="AA", 
    covar=covar,
    verbose=True,
    condition_on="AA_DRB1_11_32552129", # condition on a VARIANT of choice (optional)
    use_progress_bar=False,
    print_every=10,
)
print(results.head())
```

## Status

The core analysis framework is functional and actively under development. Current work includes:
- plotting and visualisation utilities
- workflow optimisation and refactoring
- expanded documentation and tutorials

## Intended use

hapy is designed for:
- HLA fine-mapping studies
- amino-acid association analysis
- autoimmune disease genetics
- immunogenetics workflows
- multi-cohort HLA association studies

## Contact

For questions, feedback, or collaboration enquiries:

> Chuan Fu Yap
>
> University of Manchester
>
> chuanfu.yap@manchester.ac.uk

## Citation
If you use HLA Analysis Py in your research, please cite this repository. If a corresponding publication becomes available, please cite the publication instead.
