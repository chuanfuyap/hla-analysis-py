import pandas as pd 
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
import statsmodels.formula.api as smf 

def breakitup(x):
    """
    Function called by processBGL() to break variant IDs to different columns for sorting purpose
    
    Parameters
    ------------
    x: str,
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
    tokens = x.split("_")
    if len(tokens) > 1:
        idname = ("_").join(tokens[:4])
        while len(tokens)<4:
            tokens.append(np.nan)
        variantype = tokens[0]
        genename = tokens[1]        
        aapos = tokens[2]
        genepos = tokens[3]
    else:
        idname = x
        if x.startswith("rs"):
            variantype = "SNPS"
        else:
            variantype = x
        genename = np.nan
        aapos = np.nan
        genepos = np.nan

    return idname,variantype,genename,aapos,genepos

def makehaplodf(aa_df):
    """
    Generates haplotype matrix from the 
    
    Parameters
    ------------
    aa_df: pandas DataFrame,
        processed Beagle file in dataframe format containing just a single gene with multiple amino acid at a given position
    Returns
    ------------
    df: pandas DataFrame
        processed beagle file ready for haplotype matrix generation 
    """
    df = aa_df.copy()
    df = df.drop(columns=['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS'], axis=1).T
    
    df["haplo"] = df.apply(lambda x : "".join(x), axis=1)
    df = df.reset_index()
    df["index"] = df["index"].apply(lambda x : x.split(".")[0])
    df = df.groupby(["index", "haplo"]).size().unstack(fill_value=0)
    
    try:
        df.index = df.index.astype("int")
    except:
        pass
   
    return df.sort_index()

def processBglAA(bglfileloc):
    """
    Processes Beagle (phased) file to be ready for haplotype matrix
    
    dataprocessing code adapted from [here](https://github.com/immunogenomics/HLA-TAPAS/blob/master/HLAassoc/run_omnibus_test_WS.R)
    Parameters
    ------------
    bglfileloc: str,
        file location of the beagle (phased) file
    Returns
    ------------
    df: pandas DataFrame
        processed beagle file ready for haplotype matrix generation 
    """
    df = pd.read_csv(bglfileloc, sep="\s+", header=0, skiprows=[0,2,3,4], index_col=1).drop(columns=["I"], axis=1)
    df.index.name = "SNP"
    
    df["SNP"] = df.index
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand")
    
    df = df.drop(columns=["SNP"], axis=1)
    df = df[df.TYPE=="AA"]
    
    ### now sectioning just a amino acids with >1 amino acids at the same position
    tmpix = np.where(df.groupby("AA_ID").count()["POS"]>1)[0]
    tmpix = df.groupby("AA_ID").count().index[tmpix]
    
    df = df[df.AA_ID.isin(tmpix)]
    
    return df
