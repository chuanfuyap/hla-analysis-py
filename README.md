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

> When using imputed data from Michigan server, you can convert to PLINK file formats (v1 `bilfe` or v2 `pgen`) with PLINK2 which can be read by `hapy`. It is preferable to use `pgen` format as it would retain the imputation accuracy as well as fratctional dosage values.
>
> If `.raw` file (exported by PLINK with `--export A`) is preferred, try to retain `pgen`'s pvarfile to have imputation accuracy.
>
> `bfile` format is also implemented but should be avoided as it loses fractional dosage value, giving only 0/1/2 (hard) counts and imputation accuracy information is not maintained.

To read PLINK files, use `read_plink` and choose a PLINK input format:

```python
import hapy as hp

# Text export plus its variant metadata.
hladat = hp.read_plink("cohort.raw", format="raw", pvarfileloc="cohort.pvar")

# Binary PLINK 2 fileset: cohort.pgen, cohort.pvar, cohort.psam.
hladat = hp.read_plink("cohort", format="pgen")

# Binary PLINK 1 fileset: cohort.bed, cohort.bim, cohort.fam.
hladat = hp.read_plink("cohort", format="bfile")
```

The corresponding companion files for pgen and bilfes  must share that prefix (e.g. pgen, pvar and psam, or bed, bim, fam) just like PLINK.

SNP markers starting with `SNP_` or `SNPS_` are stored in `hladat.SNP.extradata`,
with their metadata in `hladat.SNP.extrainfo`. Other SNPs are stored in
`hladat.SNP.data` and `hladat.SNP.info`. The two genotype tables do not overlap.


### Beagle input

```python
import hapy as hp

# Load SNP2HLA or Michigan HLA imputation output in Beagle format
hladat = hp.read_bgl(
    fileloc="example.bgl.phased",
    load_types=("HLA","AA"), # loads on HLA allele and Amino Acid data
    filter_R2="example.bgl.r2", R2_minimum=0.9) # input imputation confidence and filter

```

### Accesing data

```python
# reading amino acid dosage/count data
hladat.AA.data # returns Amino Acid variant by sample matrix in pandas DataFrame format

hladat.HLA.data # returns HLA allele variant by sample matrix in pandas DataFrame format

hladat.SNP.data # returns SNP variant by sample matrix in pandas DataFrame format

hladat.SNPS.extradata # returns SNP (that has SNP_ or SNPS_ prefix.) variant by sample matrix in pandas DataFrame format 

```

### Running association analyses

```python
# load pheno and covariate file with pandas (be sure the index have same ID, these are sample IDs )
pheno = pd.read_csv("example_pheno.txt",index_col=1, ) # Binary PLINK-style {1,2} is auto-recoded to {0,1}.
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



### Meta-analysis

```python
inv, stouf = hp.meta.meta_analyse("example_studies.txt", data_type="AA", pval="LR_p", variant_col="VARIANT", prefix="META")
# `example_studies.txt` refers to a txt file listing path to all the results that will be meta-analysed, this return two separate results where 
```

example_studies.txt refers to a text file listing the paths to all association results to be meta-analysed. The function returns two separate results: `inv`, containing inverse-variance meta-analysis results for individual variants, and `stouf`, containing Stouffer's meta-analysis results for omnibus tests.

## Status

The core analysis framework is functional and actively under development. Current work includes:
- plotting and visualisation utilities

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
