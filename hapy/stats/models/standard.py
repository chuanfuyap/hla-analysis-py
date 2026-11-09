"""
Standard association model engine.

Supports:
- Univariate: Y ~ covars + geno
- Omnibus:    compare null (covars only) vs alt (covars + multiple geno columns)
  Returns BOTH LR p-value and ANOVA p-value, per requirement.

Covariates
----------
Covariates are optional. Strings are encoded as categoricals using C(col) in formulas.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .common import lrtest, anova, categorical_term


def _fit(formula: str, abt: pd.DataFrame, model_type: str):
    """Fit either GLM(Binomial) or OLS based on model_type."""
    if model_type == "logit":
        return smf.glm(formula=formula, data=abt, family=sm.families.Binomial()).fit(disp=0)
    return smf.ols(formula=formula, data=abt).fit()


def _fit_info(model, model_type: str) -> str:
    """Describe unusable fitted models; an empty string means checks passed.

    Check both OLS and GLM estimates. GLM additionally exposes convergence.
    These checks do not capture or suppress numerical warnings during fitting.
    """
    issues = []
    if model_type == "logit" and not model.converged:
        issues.append("not converged")
    if not np.isfinite(model.params).all():
        issues.append("non-finite coefficients")
    if not np.isfinite(model.bse).all():
        issues.append("non-finite standard errors")
    if not np.isfinite(model.llf):
        issues.append("non-finite log-likelihood")
    return "; ".join(issues)


def _cov_terms(abt: pd.DataFrame, covar_cols: list[str]) -> list[str]:
    """Convert covariate columns to formula terms with C() wrapping where needed."""
    return [categorical_term(c, abt[c].dtype) for c in covar_cols]


def fit_univariate(abt: pd.DataFrame, geno_col: str, covar_cols: list[str], model_type: str) -> dict:
    """
    Fit a univariate association model and return p/coef/CI and INFO.

    If convergence or finiteness checks fail, return TEST and INFO only.

    Model:
      Y ~ covars + geno_col
    """
    geno_term = categorical_term(geno_col, abt[geno_col].dtype)
    terms = _cov_terms(abt, covar_cols) + [geno_term]
    f = "Y ~ " + " + ".join(terms) if terms else "Y ~ 1"
    m = _fit(f, abt, model_type)

    info = _fit_info(m, model_type)
    if info:
        return {"INFO": info, "TEST": "SINGLE"}

    ci = m.conf_int().loc[geno_term]
    return {
        "P-value": float(m.pvalues[geno_term]),
        "Beta": float(m.params[geno_term]),
        "StdErr": float(m.bse[geno_term]),
        "CI-lower": float(ci.iloc[0]),
        "CI-upper": float(ci.iloc[1]),
        "TEST": "SINGLE",
        "INFO": np.nan,
        "Statistic": float(m.tvalues[geno_term]),
    }


def fit_omnibus(abt: pd.DataFrame, geno_cols: list[str], covar_cols: list[str], model_type: str) -> dict:
    """
    Fit an omnibus association test for multiple genotype columns.

    Null:
      Y ~ covars
    Alt:
      Y ~ covars + geno_cols...

    Returns
    -------
    dict with P-value, LR-stat, DoF, TEST and INFO. If either fit fails
    convergence or finiteness checks, return TEST and INFO only.
    """
    cov_terms = _cov_terms(abt, covar_cols)
    geno_terms = [categorical_term(c, abt[c].dtype) for c in geno_cols]
    alt_terms = cov_terms + geno_terms
    null_terms = cov_terms

    altf = "Y ~ " + " + ".join(alt_terms) if alt_terms else "Y ~ 1"
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    issues = []
    for label, model in (("null model", nul), ("full model", alt)):
        info = _fit_info(model, model_type)
        if info:
            issues.append(f"{label}: {info}")
    if issues:
        return {"INFO": "; ".join(issues), "TEST": "MULTI"}

    lrstat, lrp, dof = lrtest(nul, alt)
    #fstat, fp = anova(nul, alt, abt["Y"], model_type)

    # Only keep genotype terms that exist in model
    # present = [c for c in geno_cols if c in alt.params.index]

    out = {
        "P-value": float(lrp),
        "Statistic": float(lrstat),
        "DoF" : int(dof),
        "TEST": "MULTI",
        "INFO": np.nan,
        #"Anova_p": float(fp),
        #"n_geno": len(present),
        #"geno_terms": ",".join(present) if present else None,
    }

    out["Beta"] = ";".join(
        f"{term.split('_')[1]}={alt.params[term]:.12g}" for term in geno_terms
    )
    out["StdErr"] = ";".join(
        f"{term.split('_')[1]}={alt.bse[term]:.12g}" for term in geno_terms
    )

    # if present:
    #     beta = alt.params.loc[present]
    #     cov = alt.cov_params().loc[present, present]

    #     # flatten beta
    #     for g in present:
    #         out[f"beta_{g}"] = float(beta[g])

    #     # flatten covariance (full matrix, not just upper triangle)
    #     for g1 in present:
    #         for g2 in present:
    #             out[f"cov_{g1}__{g2}"] = float(cov.loc[g1, g2])

    return out