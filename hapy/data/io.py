"""
Functions to read in genomics files. (io short for input/output)

Currently support:
- [Phased Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

Some of the hardcall data processing code adapted from [here](https://github.com/immunogenomics/HLA-TAPAS/blob/master/HLAassoc/run_omnibus_test_WS.R)
"""
__all__ = ["read_famfile", "read_bgl", "read_gprobs", "read_dosage"]
from typing import Optional, List
import time

import pandas as pd
import numpy as np
from hapy.data.HLAdat import HLAdata

def read_famfile(fileloc: str) -> pd.DataFrame:
    """
    Reads PLINK fam file and gives it appropriate headers

    Parameters
    ------------
    fileloc: str,
        file location of the fam file
    Returns
    ------------
    df: pandas DataFrame
        processed beagle file ready for haplotype matrix generation
    """
    fam = pd.read_csv(fileloc,sep=r"\s+", names=["FID", "IID", "FAT", "MOT", "SEX", "PHENO"] ,na_values = [-9,"-9"])

    return fam

def read_bgl(fileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5, simpleQC: bool = True) -> HLAdata:
    """
    Processes Phased Beagle file and store it as HLAdat object. This gives the hardcall of the variants.

    Parameters
    ------------
    fileloc: str,
        file location of the beagle (phased) file
    filter_R2: str
        file location of the bgl.r2 file, this is needed for filtering out variants with low r2 (imputation) values.
    R2_minimum: float
        minimum R2 value to filter out variants with low imputation quality, default is 0.5
    simpleQC: bool
        if True, performs a simple MAF filter by dropping variants with allele frequency below 0.5% (default)
    Returns
    ------------
    hladat: HLAdat
        HLAdat object that has dataframes of the genomic data files
    """
    start = time.time()
    print("----------------", flush=True)
    print("READING IN DATA", flush=True)
    print(f"BGL file:\t{fileloc}", flush=True)
    print("----------------", flush=True)
    df = pd.read_csv(fileloc, sep=r"\s+", header=0, index_col=1)
    markers = df.columns[0]
    df = df[df[markers]=="M"]  #pylint: disable=E1136
    df = df.drop(markers, axis=1)

    if filter_R2:
        r2 = pd.read_csv(filter_R2, sep=r"\s+", header=None, index_col=0)
        safe = r2[r2[1]>R2_minimum].index
        df = df.loc[safe] #pylint: disable=E1136

    df.index.name = "SNP"

    df["SNP"] = df.index
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand")

    df = df.drop(columns=["SNP"], axis=1)
    hladat = HLAdata(df, "hardcall")
    if simpleQC:
        print("----------------------------------------------------", flush=True)
        print("PERFORMING SIMPLE MAF FILTER: droppping 0.5% allele frequency", flush=True)
        print("----------------------------------------------------", flush=True)
        hladat.maf_filter()

    end = time.time()

    print(f"Elapsed time for loading: {end - start:.4f} seconds", flush=True)
    print("---------------------", flush=True)
    print(f"Sample Size:\t {len(hladat.SNP.data.columns)/2:.0f}", flush=True)
    print("Number of SNPs:\t", hladat.SNP.info.AA_ID.nunique(), flush=True)
    print("Number of HLA Alleles:\t", hladat.HLA.info.AA_ID.nunique(), flush=True)
    print("Number of Amino Acids:\t", hladat.AA.info.AA_ID.nunique(), flush=True)
    print("---------------------", flush=True)    

    return hladat

def read_gprobs(fileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5, simpleQC: bool = True) -> HLAdata:
    """
    Processes Beagle probability file, transform it into dosage file and store it as HLAdat object. Dosage is the probabilistic gene copy information.

    Parameters
    ------------
    fileloc: str,
        file location of the beagle probability file
    filter_R2: str
        file location of the bgl.r2 file, this is needed for filtering out variants with low r2 (imputation) values.
    R2_minimum: float
        minimum R2 value to filter out variants with low imputation quality, default is 0.5
    simpleQC: bool
        if True, performs a simple MAF filter by dropping variants with allele frequency below 0.5% (default)
    Returns
    ------------
    hladat: HLAdat
        HLAdat object that has dataframes of the genomic data files
    """
    start = time.time()
    print("----------------", flush=True)
    print("READING IN DATA", flush=True)
    print(f"GPROB file:\t{fileloc}", flush=True)
    print("----------------", flush=True)
    df = pd.read_csv(fileloc, sep=r"\s+", header=0, index_col=0)
    namecheck = df.index.name
    assert namecheck == "marker", "ERROR: File appears to be modified. If this is a SNP2HLA output, please use the dosage output with `read_dosage(dosagefileloc, phasedfileloc)` instead."

    if filter_R2:
        r2 = pd.read_csv(filter_R2, sep=r"\s+", header=None, index_col=0)
        safe = r2[r2[1]>R2_minimum].index
        df = df.loc[safe] #pylint: disable=E1101,E1137

    ## gets information from variant ID
    df.index.name = "SNP" #pylint: disable=E1101
    df["SNP"] = df.index #pylint: disable=E1101,E1137
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand") #pylint: disable=E1101,E1137
    df = df.drop(columns=["SNP"], axis=1) #pylint: disable=E1101

    hladat = HLAdata(df, "softcall")
    print("---------------------", flush=True)
    print("CONVERTING TO DOSAGE", flush=True)
    print("---------------------", flush=True)
    hladat.convert_dosage()

    if simpleQC:
        print("----------------------------------------------------", flush=True)
        print("PERFORMING SIMPLE MAF FILTER: droppping 0.5% allele frequency", flush=True)
        print("----------------------------------------------------", flush=True)
        hladat.maf_filter()

    end = time.time()

    print(f"Elapsed time for loading: {end - start:.4f} seconds", flush=True)
    print("---------------------", flush=True)
    print(f"Sample Size:\t {len(hladat.SNP.data.columns)/2:.0f}", flush=True)
    print("Number of SNPs:\t", hladat.SNP.info.AA_ID.nunique(), flush=True)
    print("Number of HLA Alleles:\t", hladat.HLA.info.AA_ID.nunique(), flush=True)
    print("Number of Amino Acids:\t", hladat.AA.info.AA_ID.nunique(), flush=True)
    print("---------------------", flush=True)  

    return hladat

def read_dosage(dosagefileloc: str, phasedfileloc: str, filter_R2: Optional[str] = None,R2_minimum: float = 0.5, simpleQC: bool = True) -> HLAdata:
    """
    Processes dosage file and store it as HLAdat object. Dosage is the probabilistic gene copy information.

    Parameters
    ------------
    dosagefileloc: str,
        file location of the dosage file
    phasefileloc: str
        file location of the phase file, this is needed for append sample IDs to the dosage file.
    filter_R2: str
        file location of the bgl.r2 file, this is needed for filtering out variants with low r2 (imputation) values.
    R2_minimum: float
        minimum R2 value to filter out variants with low imputation quality, default is 0.5
    simpleQC: bool
        if True, performs a simple MAF filter by dropping variants with allele frequency below 0.5% (default)
    Returns
    ------------
    hladat: HLAdat
        HLAdat object that has dataframe of the genomic data files
    """
    start = time.time()
    sampleIDs = getSampleIDs(phasedfileloc)
    header = ["alleleA", "alleleB"]
    header.extend(sampleIDs)
    print("----------------", flush=True)
    print("READING IN DATA", flush=True)
    print(f"Dosage file:\t{dosagefileloc}", flush=True)
    print("----------------", flush=True)
    df = pd.read_csv(dosagefileloc, sep=r"\s+", header=None, index_col=0)
    df.columns = header

    if filter_R2:
        r2 = pd.read_csv(filter_R2, sep=r"\s+", header=None, index_col=0)
        safe = r2[r2[1]>R2_minimum].index
        df = df.loc[safe] #pylint: disable=E1136

    ## gets information from variant ID
    df.index.name = "SNP" #pylint: disable=E1101
    df["SNP"] = df.index #pylint: disable=E1101,E1137
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand") #pylint: disable=E1101,E1137
    df = df.drop(columns=["SNP"], axis=1) #pylint: disable=E1101

    hladat = HLAdata(df, "softcall")
    ## drops ["alleleA", "alleleB"] off from "data" as this is now stored in "info" of the HLAData object
    hladat.SNP.data.drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)
    hladat.HLA.data.drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)
    hladat.AA.data.drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)

    if simpleQC:
        print("----------------------------------------------------", flush=True)
        print("PERFORMING SIMPLE MAF FILTER: droppping 0.5% allele frequency", flush=True)
        print("----------------------------------------------------", flush=True)
        hladat.maf_filter()

    end = time.time()

    print(f"Elapsed time for loading: {end - start:.4f} seconds", flush=True)
    print("---------------------", flush=True)
    print(f"Sample Size:\t {len(hladat.SNP.data.columns)/2:.0f}", flush=True)
    print("Number of SNPs:\t", hladat.SNP.info.AA_ID.nunique(), flush=True)
    print("Number of HLA Alleles:\t", hladat.HLA.info.AA_ID.nunique(), flush=True)
    print("Number of Amino Acids:\t", hladat.AA.info.AA_ID.nunique(), flush=True)
    print("---------------------", flush=True)    

    return hladat

def getSampleIDs(phasedfileloc: str) -> List[str]:
    """
    Extracts sample ID from phased file

    """
    with open(phasedfileloc, "r") as f:
        sampIDs = f.readline().split()[2:]
        sampIDs = np.array(sampIDs)

        idCount2 = len(sampIDs)
        ix = [i for i in range(1,idCount2,2)]

    return list(sampIDs[ix])

def breakitup(variantID: str):
    """
    Breaks variant IDs into columns for sorting.
    
    Parameters
    ------------
    variantID: str,
        variant ID from genotype files
    Returns
    ------------
    Tuple[str, str, str, str, str]
        idname, variantype, genename, aapos, genepos

        - idname - cleaned up id
        - variantype - SNPS, HLA or AA
        - genename - name of Gene if it is an amino acid variant
        - aapos - amino acid position number
        - genepos - genomic coordinate
    """
    tmpID = variantID.replace("*", "_").replace("SNPS_", "SNP_")
    tokens = tmpID.split("_")
    
    if len(tokens) > 1:
        idname = ("_").join(tokens[:4])
        while len(tokens) < 4:
            tokens.append(np.nan)
        variantype, genename, aapos, genepos = tokens[0], tokens[1], tokens[2], tokens[3]
    else:
        idname = variantID
        if variantID.startswith("rs"):
            variantype = "SNP"
        else:
            variantype = variantID
        genename = aapos = genepos = np.nan

    return idname,variantype,genename,aapos,genepos
