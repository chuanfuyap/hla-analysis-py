"""
Functions built for analysis:

Currently supports:
- Linear model and omnibus test for HLA amino acids with beagle files as input.

TODO: need to turn this into a class/module
"""
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
import statsmodels.formula.api as smf 

from scipy import stats
from sklearn.metrics import log_loss

from collections import Counter

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
    abt = dataframe.copy()
    
    altf = "PHENO ~ C(SEX) +"+"+".join(abt.columns[:-2]) 
    ### IN CASE OF CONDITIONING IS DONE
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
    haplo = np.array(list(haplo))
    presence = list(np.nonzero(haplo=="T")[0])
    
    ### breaking down the amino acids from the idlist
    aminoacids = np.array([i.split("_")[-1] for i in aalist])
    ### matching amino acid with presence marker
    aminoacids = aminoacids[presence]
    
    aminoacids = checkAAblock(aminoacids)
    
    return aminoacids

def subsectionFam(dataframe, famfile, aminoacid=True):
    df = dataframe.copy()
    newix = []
    for x in famfile.index:
        newix.append(x)
        newix.append(x+".1")
    
    if aminoacid:
        newix.extend(['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS'])
        
    return df[newix]

def runOBT(dataframe, famfile, modeltype):
    df = dataframe.copy()    
    
    fam = famfile[["IID","SEX","PHENO"]].set_index("IID")
    fam.PHENO = fam.PHENO-1
    fam = fam.sort_index()
    
    ### for if famfile has less samples than dataframe    
    df = subsectionFam(df, fam)
    
    aminoacids = df.AA_ID.unique()
    
    colnames = ["AA_ID", "GENE", "AA_POS", "LR_p", "Anova_p", "multi_Coef", "Uni_p", "Uni_Coef", "Amino_Acids", "Ref_AA"]
    output = pd.DataFrame(columns=colnames)
    
    for x in aminoacids:
        ### sectioning out singular gene amino acid position and making haplotype matrix
        aadf = df[df.AA_ID==x]
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
                                "GENE":aadf.GENE.unique()[0], 
                                "AA_POS":aadf.AA_POS.unique()[0], 
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