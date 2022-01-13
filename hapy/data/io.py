"""
Functions to read in genomics files. (io short for input/output)

Currently support:
- [Phased Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

Some of the hardcall data processing code adapted from [here](https://github.com/immunogenomics/HLA-TAPAS/blob/master/HLAassoc/run_omnibus_test_WS.R)
"""
__all__ = ["read_famfile", "read_bgl", "read_gprobs", "read_dosage"]
import pandas as pd
import numpy as np

from hapy.data.HLAdat import HLAdata

def read_famfile(fileloc):
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
    fam = pd.read_csv(fileloc,
                    sep=r"\s+", names=["FID", "IID", "FAT", "MOT", "SEX", "PHENO"] ,na_values = [-9,"-9"])
    return fam

def read_bgl(fileloc, simpleQC=True):
    """
    Processes Beagle (phased) file and store it as HLAdat object. This gives the hardcall of the variants.

    Parameters
    ------------
    fileloc: str,
        file location of the beagle (phased) file
    Returns
    ------------
    hladat: HLAdat
        HLAdat object that has dataframe of the genomic data files
    """
    print("----------------")
    print("READING IN DATA")
    print("----------------")
    df = pd.read_csv(fileloc, sep=r"\s+", header=0, index_col=1)#.drop(columns=["I"], axis=1)
    markers = df.columns[0]
    df = df[df[markers]=="M"]  #pylint: disable=E1136
    df = df.drop(markers, axis=1)

    df.index.name = "SNP"

    df["SNP"] = df.index
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand")

    df = df.drop(columns=["SNP"], axis=1)
    hladat = HLAdata(df, "hardcall")
    if simpleQC:
        print("----------------------------------------------------")
        print("PERFORMING SIMPLE QC: droppping 1% allele frequency")
        print("----------------------------------------------------")
        hladat.qualitycontrol()

    return hladat

def read_gprobs(fileloc, simpleQC=True):
    """
    Processes Beagle probability (phased) file, transform it into dosage file and store it as HLAdat object. Dosage is the probabilistic gene copy information.

    Parameters
    ------------
    fileloc: str,
        file location of the beagle probability (phased) file
    Returns
    ------------
    hladat: HLAdat
        HLAdat object that has dataframe of the genomic data files
    """
    print("----------------")
    print("READING IN DATA")
    print("----------------")
    df = pd.read_csv(fileloc, sep=r"\s+", header=0, index_col=0)
    namecheck = df.index.name
    assert namecheck == "marker", "ERROR: File appears to be modified. If this is a SNP2HLA output, please use the dosage output with `read_dosage(dosagefileloc, phasedfileloc)` instead."

    ## gets information from variant ID
    df.index.name = "SNP" #pylint: disable=E1101
    df["SNP"] = df.index #pylint: disable=E1101,E1137
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand") #pylint: disable=E1101,E1137
    df = df.drop(columns=["SNP"], axis=1) #pylint: disable=E1101

    hladat = HLAdata(df, "softcall")
    print("---------------------")
    print("CONVERTING TO DOSAGE")
    print("---------------------")
    hladat.convertDosage()

    if simpleQC:
        print("----------------------------------------------------")
        print("PERFORMING SIMPLE QC: droppping 1% allele frequency")
        print("----------------------------------------------------")
        hladat.qualitycontrol()

    return hladat

def read_dosage(dosagefileloc, phasedfileloc, simpleQC=True):
    """
    Processes dosage file and store it as HLAdat object. Dosage is the probabilistic gene copy information.

    Parameters
    ------------
    dosagefileloc: str,
        file location of the dosage file
    phasefileloc: str
        file location of the phase file, this is needed for append sample IDs to the dosage file.
    Returns
    ------------
    hladat: HLAdat
        HLAdat object that has dataframe of the genomic data files
    """
    sampleIDs = getSampleIDs(phasedfileloc)
    header = ["alleleA", "alleleB"]
    header.extend(sampleIDs)
    print("----------------")
    print("READING IN DATA")
    print("----------------")
    df = pd.read_csv(dosagefileloc, sep=r"\s+", header=None, index_col=0)
    df.columns = header

    ## gets information from variant ID
    df.index.name = "SNP" #pylint: disable=E1101
    df["SNP"] = df.index #pylint: disable=E1101,E1137
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand") #pylint: disable=E1101,E1137
    df = df.drop(columns=["SNP"], axis=1) #pylint: disable=E1101

    hladat = HLAdata(df, "softcall")
    ## drops ["alleleA", "alleleB"] off from "data" as this is now stored in "info" of the HLAData object
    hladat.SNP["data"].drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)
    hladat.HLA["data"].drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)
    hladat.AA["data"].drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)

    if simpleQC:
        print("----------------------------------------------------")
        print("PERFORMING SIMPLE QC: droppping 1% allele frequency")
        print("----------------------------------------------------")
        hladat.qualitycontrol()

    return hladat

def getSampleIDs(phasedfileloc):
    """
    Extracts sample ID from phased file

    """
    with open(phasedfileloc, "r") as f:
        sampIDs = f.readline().split()[2:]
        sampIDs = np.array(sampIDs)

        idCount2 = len(sampIDs)
        ix = [i for i in range(1,idCount2,2)]

    return list(sampIDs[ix])

def breakitup(variantID):
    """
    Function called by processBGL() to break variant IDs to different columns for sorting purpose

    Parameters
    ------------
    variantID: str,
        variant ID from genotype files
    Returns
    ------------
    idname,variantype,genename,aapos,genepos : str
        idname - cleaned up id
        variantype - SNPS, HLA or AA
        genename - name of Gene if it is an amino acid variant
        aapos - amino acid position number
        genepos - genomic coordinate
    """
    tmpID = variantID.replace("*", "_")
    tmpID = tmpID.replace("SNPS_", "SNP_")
    tokens = tmpID.split("_")
    if len(tokens) > 1:
        idname = ("_").join(tokens[:4])
        while len(tokens)<4:
            tokens.append(np.nan)
        variantype = tokens[0]
        genename = tokens[1]
        aapos = tokens[2]
        genepos = tokens[3]
    else:
        idname = variantID
        if variantID.startswith("rs"):
            variantype = "SNP"
        else:
            variantype = variantID
        genename = np.nan
        aapos = np.nan
        genepos = np.nan

    return idname,variantype,genename,aapos,genepos
