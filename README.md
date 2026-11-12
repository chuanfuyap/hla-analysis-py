# hla-analysis-python (hapy)

A Python framework for HLA allele, amino-acid, SNP, interaction, survival, and meta-analysis workflows in immunogenetic association studies.

## Overview

hapy is a Python package for downstream analysis of HLA imputation outputs from genetic association studies. The framework supports analysis of classical HLA alleles, amino-acid polymorphisms, and SNP markers across the HLA region using a unified workflow interface.

Currently supported input formats include:
- SNP2HLA output
- Michigan Imputation Server HLA output
- PLINK-derived genotype formats
- Beagle-derived genotype format

## Intended use

`hapy` is designed for:
- HLA fine-mapping studies
- amino-acid association analysis
- autoimmune disease genetics
- immunogenetics workflows
- multi-cohort HLA association studies

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

## Quick Start

### PLINK input

For imputed data, PGEN is preferred when fractional dosages are stored in the
file. The BED/BIM/FAM (`bfile`) format stores hard genotype calls rather than
fractional dosages. RAW input preserves the dosage values exported into it, but
with reduced numerical precision due to fewer decimal places.

Imputation-quality scores must be retained separately, for example in a PVAR
column or INFO field, or an external R² file. A PVAR file does not necessarily
contain quality scores, and supplying one alone does not enable quality
filtering. Request filtering with `pvar_R2_column` or `filter_R2`, together with
`R2_minimum`; see the [association guide](docs/association.md) for examples.
RAW can also be read without PVAR, but imputation-quality filtering requires a
suitable quality source.

To read PLINK files, use `read_plink` and choose a PLINK input format:

```python
import hapy as hp

# .raw (exported by PLINK with `--export A`) 
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

hladat.SNP.extradata # returns SNP (that has SNP_ or SNPS_ prefix.) variant by sample matrix in pandas DataFrame format

```

### Running association analyses

See the [association guide](docs/association.md) for more details and
[bundled example](examples/association/) to have an idea what the data formats should look like.

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
inv, stouf = hp.meta.meta_analyse("example_studies.txt", data_type="AA")
# `example_studies.txt` refers to a txt file listing path to all the results that will be meta-analysed, this return two separate results where 
```

`example_studies.txt`refers to a text file listing the paths to all association results to be meta-analysed. The function returns `inv` for inverse-variance results and `stouf` for weighted Stouffer results. For AA analysis, any position labelled `MULTI` in a cohort sends all available `SINGLE`/`MULTI` P-values for that position to Stouffer; all-`SINGLE` positions use IVW. OVR rows are excluded. Input column names are configurable; see [meta-analysis](docs/meta_analysis.md).

### HLA amino acid position plots

Use `hp.vis.plot_aa_association()` for AA position association plots and
`hp.vis.plot_aa_interaction()` with `kind="scatter3d"` for
position-level genetic (G×G) interactions. Both functions default to numeric
amino acid sequence positions within one HLA gene; select it with, for example,
`genes="DRB1"`. Use `x_axis="genomic"` for physical genomic spacing or
`x_axis="equally_spaced"` for evenly spaced positions across genes.
See [visualisation documentation](docs/visualisation.md) and
[synthetic examples](examples/visualisation/plot_examples.py).

## Status

The core analysis framework is functional and actively under development. Current work includes:
- stepwise conditional analyses
- allele mapping
- haplotype analysis

## Contact

For questions, feedback, or collaboration enquiries:

> Chuan Fu Yap
>
> University of Manchester
>
> chuanfu.yap@manchester.ac.uk

## Citation
If you use HLA Analysis Py in your research, please cite this repository. If a corresponding publication becomes available, please cite the publication instead.
