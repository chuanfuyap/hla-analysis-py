"""
Common statistical utilities used across model engines.

Contains:
- Likelihood ratio test for nested models
- Deviance + ANOVA-style F-test logic

These functions are adapted from the legacy stats.py and kept intentionally similar.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import log_loss


def lrtest(nullmodel, altmodel):
    """
    Likelihood ratio test for two nested statsmodels models.

    Parameters
    ----------
    nullmodel:
        Restricted model.
    altmodel:
        Full model.

    Returns
    -------
    lr : float
        LR statistic.
    p : float
        Chi-square p-value with df = df_model(alt) - df_model(null).
    Taken from:
    https://scientificallysound.org/2017/08/24/the-likelihood-ratio-test-relevance-and-application/
    https://stackoverflow.com/questions/30541543/how-can-i-perform-a-likelihood-ratio-test-on-a-linear-mixed-effect-model
    Theory explained here:
    https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
    https://www.itl.nist.gov/div898/handbook/apr/section2/apr233.htm
    """

    # Log-likelihood of model
    alt_llf = altmodel.llf
    null_llf = nullmodel.llf
    # since they are log-transformed, division is subtraction. So this is the ratio
    lr = 2 * (alt_llf - null_llf)
    # normal formula for this is (-2*log(null/alt)), but since llf is already log-transformed it is the above, and since we put alt model infront, we don't need the negative sign.

    # degree of freedom
    all_dof = altmodel.df_model
    null_dof = nullmodel.df_model

    dof = all_dof - null_dof

    p = stats.chi2.sf(lr, dof)

    return lr, p, dof


def deviance(ytrue, model):
    """
    Compute deviance (2 * negative log-likelihood) from a fitted statsmodels GLM.

    Notes
    -----
    This emulates residual deviance for logistic GLMs.
    """
    ypred = model.predict().reshape(-1, 1)
    ypred = np.column_stack([1 - ypred, ypred])
    return 2 * log_loss(ytrue, ypred, normalize=False)


def anova(nullmodel, altmodel, ytrue, modeltype):
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

    # deviance of residuals for logit (logistic)
    if modeltype == "logit":
        alt_ssr = deviance(ytrue, altmodel)
        null_ssr = deviance(ytrue, nullmodel)

    else:  # else sum of squared error for linear
        alt_ssr = altmodel.ssr
        null_ssr = nullmodel.ssr

    # degree of freedom from residuals
    alt_df_resid = altmodel.df_resid
    null_df_resid = nullmodel.df_resid

    # computing fvalue and pvalue
    ssdiff = null_ssr - alt_ssr
    dof = null_df_resid - alt_df_resid

    fvalue = ssdiff/dof/altmodel.scale

    pvalue = stats.f.sf(fvalue, dof, alt_df_resid)

    return fvalue, pvalue


def categorical_term(col: str, dtype) -> str:
    """
    Return a formula term for a column, wrapping categorical columns in C(...).

    This is used for covariates and covariate-block interaction terms.
    """
    if getattr(dtype, "name", "") in ("object", "category", "string"):
        return f"C({col})"
    if not col.isidentifier():
        return f'Q("{col}")'
    return col