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

from .common import lrtest, anova, categorical_term, _fit_info
from ._null import null_summary
from scipy import stats


def _fit(formula: str, abt: pd.DataFrame, model_type: str):
    """Fit either GLM(Binomial) or OLS based on model_type."""
    if model_type == "logit":
        return smf.glm(formula=formula, data=abt, family=sm.families.Binomial()).fit(disp=0)
    return smf.ols(formula=formula, data=abt).fit()



def _cov_terms(abt: pd.DataFrame, covar_cols: list[str]) -> list[str]:
    """Convert covariate columns to formula terms with C() wrapping where needed."""
    return [categorical_term(c, abt[c].dtype) for c in covar_cols]


def fit_univariate(abt: pd.DataFrame, geno_col: str, covar_cols: list[str], model_type: str, *, _return_model=False) -> dict:
    """
    Fit a univariate association model and return p/coef/CI and INFO.

    If convergence or finiteness checks fail, return TEST and INFO only.
    The private _return_model option keeps the fitted object inside the worker
    for validated reverse contrasts; the default returns the existing dict.

    Model:
      Y ~ covars + geno_col
    """
    geno_term = categorical_term(geno_col, abt[geno_col].dtype)
    terms = _cov_terms(abt, covar_cols) + [geno_term]
    f = "Y ~ " + " + ".join(terms) if terms else "Y ~ 1"
    m = _fit(f, abt, model_type)

    info = _fit_info(m, model_type)
    if info:
        out = {"INFO": info, "TEST": "SINGLE"}
        return (out, m) if _return_model else out

    ci = m.conf_int().loc[geno_term]
    out = {
        "P-value": float(m.pvalues[geno_term]),
        "Beta": float(m.params[geno_term]),
        "StdErr": float(m.bse[geno_term]),
        "CI-lower": float(ci.iloc[0]),
        "CI-upper": float(ci.iloc[1]),
        "TEST": "SINGLE",
        "INFO": np.nan,
        "Statistic": float(m.tvalues[geno_term]),
    }

    return (out, m) if _return_model else out


def fit_standard_null(abt, covar_cols, model_type):
    """Fit the baseline once and retain its likelihood, actual rank and diagnostics."""
    if abt.empty:
        return None
    terms = _cov_terms(abt, covar_cols)
    formula = "Y ~ " + " + ".join(terms) if terms else "Y ~ 1"
    baseline = abt[[*covar_cols, "Y"]]
    model = _fit(formula, baseline, model_type)
    info = _fit_info(model, model_type)
    return null_summary(baseline, model.llf, getattr(model, "df_model", np.nan), model_type,
                        (formula, tuple(covar_cols)), info)


def fit_omnibus(abt: pd.DataFrame, geno_cols: list[str], covar_cols: list[str], model_type: str, *, null_cache=None) -> dict:
    """
    Fit an omnibus association test for multiple genotype columns.

    Null:
      Y ~ covars
    Alt:
      Y ~ covars + geno_cols...

    Returns
    -------
    dict with P-value, Statistic, DoF, TEST and INFO. If either fit fails
    convergence or finiteness checks, return TEST and INFO only.
    """
    abt = abt.dropna(subset=["Y", *covar_cols, *geno_cols])
    cov_terms = _cov_terms(abt, covar_cols)
    geno_terms = [categorical_term(c, abt[c].dtype) for c in geno_cols]
    alt_terms = cov_terms + geno_terms
    null_terms = cov_terms

    altf = "Y ~ " + " + ".join(alt_terms) if alt_terms else "Y ~ 1"
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    alt = _fit(altf, abt, model_type)
    baseline = abt[[*covar_cols, "Y"]]
    reuse = null_cache is not None and null_cache.matches(
        baseline, model_type, (nullf, tuple(covar_cols)))
    null = null_cache if reuse else fit_standard_null(abt, covar_cols, model_type)
    null_info = null.info

    issues = []
    for label, info in (("null model", null_info), ("full model", _fit_info(alt, model_type))):
        if info:
            issues.append(f"{label}: {info}")
    if issues:
        return {"INFO": "; ".join(issues), "TEST": "MULTI"}

    null_llf = null.log_likelihood
    null_df = null.df_model
    lrstat = 2 * (alt.llf - null_llf)
    dof = alt.df_model - null_df
    lrp = stats.chi2.sf(lrstat, dof)
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