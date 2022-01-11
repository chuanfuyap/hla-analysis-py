"""
Functions built for analysis:

Currently supports:
- Linear model and omnibus test for HLA amino acids with beagle files as input.

"""
__all__ = ["analyseAA"]
from collections import Counter
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.metrics import log_loss

def lrtest(nullmodel, altmodel):
    """
    Likelihood ratio test for 2 linear models from statsmodel

    Parameters
    ------------
    altmodel: fitted linear model from statsmodel
        the full (alternative) model with "extra" variables of interest in the model
    nullmodel: fitted linear model from statsmodel
        the restricted/nested (null) model with "extra" variables of interest removed from the model
    Returns
    ------------
    lr: float
        likelihood ratio

    p: float
        p-value from the significance testing (<0.05 for altmodel to be significantly better)

    Taken from:
    https://scientificallysound.org/2017/08/24/the-likelihood-ratio-test-relevance-and-application/
    https://stackoverflow.com/questions/30541543/how-can-i-perform-a-likelihood-ratio-test-on-a-linear-mixed-effect-model
    Theory explained here:
    https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
    """

    ## Log-likelihood of model
    alt_llf = altmodel.llf
    null_llf = nullmodel.llf
    ## since they are log-transformed, division is subtraction. So this is the ratio
    lr = 2 * (alt_llf - null_llf)
    ## normal formula for this is (-2*log(null/alt)), but since llf is already log-transformed it is the above, and since we put alt model infront, we don't need the negative sign.

    ### degree of freedom
    all_dof = altmodel.df_model
    null_dof = nullmodel.df_model

    dof = all_dof - null_dof

    p = stats.chi2.sf(lr, dof)
    return lr, p

def deviance(ytrue, model):
    """
    computes deviance needed for anova calculation
    """
    ## making prediction in format of predictproba from sklearn.
    ypred = model.predict().reshape(-1,1)
    ypred = np.column_stack([1-ypred,ypred])

    return 2*log_loss(ytrue, ypred, normalize=False)

def anova(nullmodel, altmodel,ytrue, modeltype):
    """
    Anova test between 2 fitter linear model, this test uses F-test from
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html

    Theory here:
    http://pytolearn.csd.auth.gr/d1-hyptest/11/f-distro.html

    modified from https://www.statsmodels.org/stable/_modules/statsmodels/stats/anova.html#anova_lm

    deviance residuals from here:
    https://stackoverflow.com/questions/50975774/calculate-residual-deviance-from-scikit-learn-logistic-regression-model

    Parameters
    ------------
    altmodel: fitted linear model from statsmodel
        the full (alternative) model with "extra" variables of interest in the model
    nullmodel: fitted linear model from statsmodel
        the restricted/nested (null) model with "extra" variables of interest removed from the model
    Returns
    ------------
    test: float
        test statistic
    p: float
        p-value from the significance testing (<0.05 for altmodel to be significantly better)
    """

    ### deviance of residuals for logit (logistic)
    if modeltype=="logit":
        alt_ssr = deviance(ytrue, altmodel)
        null_ssr = deviance(ytrue, nullmodel)

    else: ### else sum of squared error for linear
        alt_ssr = altmodel.ssr
        null_ssr = nullmodel.ssr

    ### degree of freedom from residuals
    alt_df_resid = altmodel.df_resid
    null_df_resid = nullmodel.df_resid

    ### computing fvalue and pvalue
    ssdiff = null_ssr - alt_ssr
    dof = null_df_resid - alt_df_resid

    fvalue = ssdiff/dof/altmodel.scale

    pvalue = stats.f.sf(fvalue, dof, alt_df_resid)

    return fvalue, pvalue

def obt(dataframe, haplotypenumber, model):
    """
    Performs omnibustest as called by `runAnalysis`

    Parameters
    ------------
    dataframe: Pandas DataFrame
        the design matrix, genotype and covariates (X) along with the target/phenotype (y) in one table
    haplotypenumber: int
        number of haplotypes in the design matrix, used to subsection out covariates.
    model: str
        model type based on the phenotype, either 'logit' (binomial/binary) or 'linear' (continuous)
    Returns
    ------------
    test: float
        test statistic
    p: float
        p-value from the significance testing (<0.05 for altmodel to be significantly better)
    """
    abt = dataframe.copy()
    ### -2 is because the abt is usually structured as GENOTYPE in the first columns then last 2 are SEX then PHENOTYPE
    altf = "PHENO ~ C(SEX) +"+"+".join(abt.columns[:-2])
    ### IN CASE OF CONDITIONING IS DONE
    ### haplotypenumber would count up all columns of GENOTYPE, and -2 would remove SEX AND PHENOTYPE, anything in between should be extra covariates desired to be modelled.
    if len(abt.columns[haplotypenumber:-2])>0:
        nullf = "PHENO ~ C(SEX) +"+"+".join(abt.columns[haplotypenumber:-2])
    else:
        nullf = "PHENO ~ C(SEX)"

    if model.lower()=="logit":
        alt_model = smf.glm(formula = str(altf), data = abt, family=sm.families.Binomial()).fit(disp=0)
        null_model = smf.glm(formula = str(nullf), data = abt, family=sm.families.Binomial()).fit(disp=0)
    else:
        alt_model = smf.ols(formula = str(altf), data = abt).fit()
        null_model = smf.ols(formula = str(nullf), data = abt).fit()

    lrstat, lrp = lrtest(null_model, alt_model)
    fstat, fp = anova(null_model, alt_model, abt.PHENO, model)

    coefs = []
    for col in abt.columns[:haplotypenumber]:
        coefs.append(round(alt_model.params[col],3))

    return lrstat, lrp, fstat, fp, coefs

def linear_model(dataframe, model):
    """
    Fit linear model given dataframe (abt) of features (gene copy number/probability) and target (phenotype)
    Parameters
    ------------
    dataframe: Pandas DataFrame
        the design matrix, genotype and covariates (X) along with the target/phenotype (y) in one table
    model: str
        model type based on the phenotype, either 'logit' (binomial/binary) or 'linear' (continuous)
    Returns
    ------------
    pvalue: float
        p-value

    coef: float
        regression coefficient (effect size of the genotype on phenotype)
    """
    abt = dataframe.copy()
    f = "PHENO ~ C(SEX) +"+"+".join(abt.columns[:-2]) ## minus because last 2 columns are sex and pheno

    if model.lower()=="logit":
        model = smf.glm(formula = str(f), data = abt, family=sm.families.Binomial()).fit(disp=0)
    else: ## else it is a linear model
        model = smf.ols(formula = str(f), data = abt).fit()

    pvalue = model.pvalues[abt.columns[0]]
    coef = model.params[abt.columns[0]]

    return pvalue, round(coef, 3)

def getRefAA(haplo, aalist):
    """
    Used within `analyseAA` to know what is the reference AA to be saved in output.
    """
    haplo = np.array(list(haplo))
    #presence = list(np.nonzero(haplo=="T")[0])
    presence = list(np.where((haplo=="P") | (haplo=="T"))[0])

    ### breaking down the amino acids from the idlist
    aminoacids = np.array([i.split("_")[-1] for i in aalist])
    ### matching amino acid with presence marker
    aminoacids = aminoacids[presence]

    aminoacids = checkAAblock(aminoacids)

    return aminoacids

def subsectionFam(dataframe, famfile, datatype):
    """
    Used within runnign of analysis for if the supplied fam file have less samples than the genotype file.
    """
    df = dataframe.copy()
    newix = []

    if datatype == "softcall":
        newix = list(famfile.index)
    elif datatype == "hardcall":
        for x in famfile.index:
            newix.append(x)
            newix.append(x+".1")

    return df[newix]

def obthaplo_hard(aadf):
    """
    function to hide away big chunk of code difference between hardcall/softcall
    """
    ## make haplotype matrix
    haplodf, aalist = makehaplodf(aadf)
    AAcount = haplodf.shape[1]

    ### check if having none of haplotypes is in the column
    missing = "".join(np.repeat("A", aadf.shape[0]))
    if missing in haplodf.columns:
        haplodf = haplodf.drop(missing, axis=1)
        haplocount = haplodf.shape[1]

        refAA = "missing"

    ### dropping most frequent haplotype as reference
    else:
        refix = np.argmax(haplodf.sum())
        refcol = haplodf.columns[refix]
        haplodf = haplodf.drop(refcol, axis=1)
        haplocount = haplodf.shape[1]

        refAA = getRefAA(refcol, aadf.index)

    return haplodf, AAcount, refAA, aalist, haplocount

def obthaplo_soft(aadf):
    """
    function to hide away big chunk of code difference between hardcall/softcall
    """
    ## make haplotype matrix
    haplodf = makehaploprob(aadf)
    aalist = haplodf.columns
    AAcount = len(aalist)

    ### dropping most frequent haplotype as reference
    refix = np.argmax(haplodf.sum())
    refAA = haplodf.columns[refix]
    haplodf = haplodf.drop(refAA, axis=1)
    haplocount = haplodf.shape[1]

    ## renaming columns to prevent function conflict
    haplodf.columns = ["AA_"+cname.replace(".", "dot").replace("*", "asterisk") for cname in haplodf.columns]

    return haplodf, AAcount, refAA, aalist, haplocount

def analyseAA(hladat, famfile, modeltype):
    """
    Goes through all the variants in the given genotype file (dataframe) and build a abt with `famfile` which is then analysed using linear models/omnibus test using the appropriate `modeltype`

    Parameters
    ------------
    dataframe: pandas DataFrame,
        the genotype file containing either copy number or probability (dosage)
    famfile: pandas DataFrame
        the sample information file to include covariates such as sex.
    modeltype: str
        model type based on the phenotype, either 'logit' (binomial/binary) or 'linear' (continuous)
    Returns
    ------------
    output: pandas DataFrame
        the output table containing p-values, coefficients for all the variants tested.
    """
    df = hladat.AA["data"].copy()
    info = hladat.AA["info"].copy()

    fam = famfile[["IID","SEX","PHENO"]].set_index("IID")
    fam.PHENO = fam.PHENO-1 ## minus 1 since PLINK often use 1/2 for phenotype.
    fam = fam.sort_index()

    ### for if famfile has less samples than dataframe
    df = subsectionFam(df, fam, hladat.type)

    aminoacids = info.AA_ID.unique()

    colnames = ["AA_ID", "GENE", "AA_POS", "LR_p", "Anova_p", "multi_Coef", "Uni_p", "Uni_Coef", "Amino_Acids", "Ref_AA"]
    output = pd.DataFrame(columns=colnames)

    df["AA_ID"] = info["AA_ID"]

    for x in aminoacids:
        ### sectioning out singular gene amino acid position and making haplotype matrix
        aadf = df[df.AA_ID==x]
        aainfo = info[info.AA_ID==x]

        if hladat.type == "softcall":
            haplodf, AAcount, refAA, aalist, haplocount = obthaplo_soft(aadf)
        elif hladat.type == "hardcall":
            haplodf, AAcount, refAA, aalist, haplocount = obthaplo_hard(aadf)

        ### building abt
        abt = pd.concat([haplodf, fam], axis=1)

        ### Perform omnibus test if at least 3 amino acids
        if AAcount>2:
            _,lrp, _, anovap, multicoef = obt(abt, haplocount, modeltype)
            multicoef = [str(x) for x in multicoef]
            multicoef = ", ".join(multicoef)

            uni_p = np.nan
            coef = np.nan

        ### Perform univariate test between 2 amino acids
        elif AAcount==2:
            uni_p, coef = linear_model(abt, modeltype)

            lrp = np.nan
            anovap = np.nan
            multicoef = np.nan

        else: ## nothing done
            lrp = np.nan
            anovap = np.nan
            uni_p = np.nan
            refAA = np.nan
            coef = np.nan
            multicoef = np.nan

        aalist = [str(x) for x in aalist]
        aalist = ", ".join(set(aalist))
        output = output.append({"AA_ID":aadf.AA_ID.unique()[0],
                                "GENE":aainfo.GENE.unique()[0],
                                "AA_POS":aainfo.AA_POS.unique()[0],
                                "LR_p": lrp,
                                "Anova_p": anovap,
                                "multi_Coef": multicoef,
                                "Uni_p": uni_p,
                                "Uni_Coef": coef,
                                "Amino_Acids": aalist,
                                "Ref_AA": refAA},
                                ignore_index=True)

        output["LRp_Unip"] = output[["LR_p","Uni_p"]].fillna(0).sum(1).replace(0, np.nan)

    return output

def makehaplodf(aa_df, basicQC=True):
    """
    Generates haplotype matrix from the genotype dataframe

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

    df = df.drop(columns=['AA_ID'], axis=1).T

    df["haplo"] = df.apply(lambda x : "".join(x), axis=1) #pylint: disable=W0108
    df = df.reset_index()
    df["index"] = df["index"].apply(lambda x : x.split(".")[0])
    df = df.groupby(["index", "haplo"]).size().unstack(fill_value=0)

    ### THIS IS DONE AFTER BGL FILE QC BECAUSE:
    ### while the variant/allele e.g. AA_A_19_29910588_A might have a lot of "T"(presence) in the samples,
    ### but when haplotype is formed across the amino acids in the same position, e.g. AATA etc, the haplotype might be low in frequency, so they are dropped
    if basicQC:
        highfreq = df.sum(0)/2/df.shape[0] >0.01
        highfreq = highfreq[highfreq]
        df=df[highfreq.index]

    haplo = df.columns
    aminoacids = get_aminoacids(aminoacids, haplo)

    return df.sort_index(), aminoacids

def makehaploprob(aa_df, basicQC=True):
    """
    Generates haplotype matrix from the dosage genotype dataframe

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

    df = df.drop(columns=['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS'], axis=1)
    ## picking out the amino acids from variant IDs
    df["AA"] = df.index
    df["AA"] = df.AA.apply(lambda x : x.split("_")[-1])
    ## selecting IDs with just 1 amino acid (non-ambigious)
    df["aa_length"] = df.AA.apply(lambda x : len(x)) #pylint: disable=W0108
    df = df[df.aa_length==1]
    ## returning final output
    df = df.set_index("AA").drop("aa_length", axis=1)
    df = df.astype("float")

    df = df.T.sort_index()

    if basicQC:
        highfreq = df.sum(0)/(df.shape[0]*2) > 0.01
        highfreq = highfreq[highfreq]

        df=df[highfreq.index]

    return df.sort_index()

def checkAAblock(aablock):
    """
    Used by `get_aminoacids` to determine what is the amino acid amongst the hardcalls.

    E.g. if there is FY, FS, FK, from the block, it is likely to be amino acid F.
    """
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
    """
    Used by `makehaplodf` to know what are the singular amino acids rather than composite amino acid names.
    """
    ### extracting the amino acid from variant ID name
    aalist = np.array([i.split("_")[-1] for i in idlist])
    aablocks = []

    ### finding the presence marked with T
    for x in haplotypes:
        presence = []
        haplo = np.array(list(x))
        presence = list(np.where((haplo=="P") | (haplo=="T"))[0])
        #presence = list(np.nonzero(haplo=="T")[0])

        ## extracting amino acid from list based of presence (T) in this given haplotype
        block = list(aalist[presence])

        block = checkAAblock(block)
        if block:
            aablocks.append(block)

    return aablocks
