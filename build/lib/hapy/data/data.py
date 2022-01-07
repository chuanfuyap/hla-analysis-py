"""
Functions to load and modify haplotype genomics files.

Currently support [Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs. 

TODO: need to turn this into a class/module
"""
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
import statsmodels.formula.api as smf 

from collections import Counter

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

def checkAAblock(aablock):
    ## extracting ambigious amino acids
    aminoacids = [list(i) for i in aablock]
    ### flattening out to make singular list
    aminoacids = [x for i in aminoacids for x in i]
    ## counting the amino acids inside the blocks
    aminoacids = Counter(aminoacids)
    aminoacids = dict(sorted(aminoacids.items(), key=lambda item: item[1], reverse=True))
    ## if there is only 1 most frequent amino acid, the block of amino acid is represented by it, otherwise block stays as a block
    if len(aminoacids)>1:
        aakeys = list(aminoacids.keys())
        if aminoacids[aakeys[0]] != aminoacids[aakeys[1]]:
            aablock = aakeys[0]
    elif len(aminoacids)==1:
        aablock = aablock[0]
    else:
        aablock = ""
    return aablock

def get_aminoacids(idlist, haplotypes):   
    ### extracting the amino acid from variant ID name
    aalist = np.array([i.split("_")[-1] for i in idlist])
    aablocks = []
    
    ### finding the presence marked with T
    for x in haplotypes:        
        presence = []
        haplo = np.array(list(x))
        presence = list(np.nonzero(haplo=="T")[0])
        
        ## extracting amino acid from list based of presence (T) in this given haplotype
        block = list(aalist[presence])
       
        block = checkAAblock(block)
        if block:
            aablocks.append(block)
        
    return aablocks

def makehaplodf(aa_df, basicQC=True):
    """
    Generates haplotype matrix from the 
    
    Parameters
    ------------
    aa_df: pandas DataFrame,
        processed Beagle file in dataframe format containing just a single gene with multiple amino acid at a given position
    basicQC: Boolean
        to perform qc on haplotype matrix generated, dropping haplotype with frequency less than 10% across samples
    Returns
    ------------
    df: pandas DataFrame
        processed beagle file ready for haplotype matrix generation 
    """
    df = aa_df.copy()
    aminoacids = df.index
    
    df = df.drop(columns=['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS'], axis=1).T
    
    df["haplo"] = df.apply(lambda x : "".join(x), axis=1)
    df = df.reset_index()
    df["index"] = df["index"].apply(lambda x : x.split(".")[0])
    df = df.groupby(["index", "haplo"]).size().unstack(fill_value=0)
    
    
    ### THIS IS DONE AFTER BGL FILE QC BECAUSE:
    ### while the variant/allele e.g. AA_A_19_29910588_A might have a lot of "T"(presence) in the samples, 
    ### but when haplotype is formed across the amino acids in the same position, e.g. AATA etc, the haplotype might be low in frequency, so they are dropped
    if basicQC:
        highfreq = df.sum(0)/2/df.shape[0] >0.1
        highfreq = highfreq[highfreq]
        df=df[highfreq.index] 
    
    haplo = df.columns
    aminoacids = get_aminoacids(aminoacids, haplo)

    return df.sort_index(), aminoacids


def processBglAA(bglfileloc):
    """
    Processes Beagle (phased) file to be ready for haplotype matrix
    
    Parts of dataprocessing code for beagle file adapted from [here](https://github.com/immunogenomics/HLA-TAPAS/blob/master/HLAassoc/run_omnibus_test_WS.R)
    Parameters
    ------------
    bglfileloc: str,
        file location of the beagle (phased) file
    Returns
    ------------
    df: pandas DataFrame
        processed beagle file ready for haplotype matrix generation 
    """
    df = pd.read_csv(bglfileloc, sep="\s+", header=0, index_col=1)#.drop(columns=["I"], axis=1)
    markers = df.columns[0]
    df = df[df[markers]=="M"]
    df = df.drop(markers, axis=1)
    
    df.index.name = "SNP"
    
    df["SNP"] = df.index
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1,result_type="expand")
    
    df = df.drop(columns=["SNP"], axis=1)
    df = df[df.TYPE=="AA"]
    
    ### now sectioning just a amino acids with >1 amino acids at the same position
    tmpix = np.where(df.groupby("AA_ID").count()["POS"]>1)[0]
    tmpix = df.groupby("AA_ID").count().index[tmpix]
    
    df = df[df.AA_ID.isin(tmpix)]
    
    for col in df.columns[:-5]:
        df[col] = df[col].map({"P":"T", "A":"A", "T":"T"})
    
    return df


def qualitycontrol(dataframe, samplefilter=0.1, alellefilter=0.1, aminoacid=True ):
    """
    Takes output from processBglAA() and perform quality control, removing samples with <10% alleles present and variants with sample frequency <10%
    
    NOTE: this function drops low frequency alleles first
    
    Parameters
    -----------
    dataframe:
        Beagle dataset to undergo quality control
    samplefilter: float
        sample frequency (% of samples across all alleles)
    allelfilter: float
        allele frequency (% of alleles across all samples)
    aminoacid: boolean
        if dataset is made up of amino acid, if it is other variants, set to False
    
    Returns
    ------------
    df: Pandas DataFrame
        processed dataset
        
    """
    df = dataframe.copy()
    ### taking out aminoacid info
    if aminoacid:
        aainfo = df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']]
    
    df = df[df.columns[:-5]] 
    ### QC allele frequency
    highfreqallele = (df=="T").sum(1)/df.shape[1] > alellefilter
    df = df.loc[highfreqallele]
    
    ### QC sample frequency
    df = df.T
    df["sample"] = [x.split(".")[0] for x in df.index]
    
    highfreqsample = df.groupby("sample").sum().sum(1).apply(lambda x : x.count("T")/len(x)) > samplefilter
    newix = []
    for x in highfreqsample.index:
        newix.append(x)
        newix.append(x+".1")
    df = df.loc[newix]
    df = df.drop("sample", axis=1)
    df = df.T
    
    ### adding back aminoacid info
    if aminoacid:
        aainfo = aainfo.loc[highfreqallele]
        df = pd.concat([df, aainfo], axis=1)
    
    return df