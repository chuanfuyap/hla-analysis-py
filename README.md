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
