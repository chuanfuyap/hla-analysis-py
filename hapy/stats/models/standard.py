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
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .common import lrtest, anova, categorical_term


def _fit(formula: str, abt: pd.DataFrame, model_type: str):
    """Fit either GLM(Binomial) or OLS based on model_type."""
    if model_type == "logit":
        return smf.glm(formula=formula, data=abt, family=sm.families.Binomial()).fit(disp=0)
    return smf.ols(formula=formula, data=abt).fit()


def _cov_terms(abt: pd.DataFrame, covar_cols: list[str]) -> list[str]:
    """Convert covariate columns to formula terms with C() wrapping where needed."""
    return [categorical_term(c, abt[c].dtype) for c in covar_cols]


def fit_univariate(abt: pd.DataFrame, geno_col: str, covar_cols: list[str], model_type: str) -> dict:
    """
    Fit a univariate association model and return p/coef/CI.

    Model:
      Y ~ covars + geno_col
    """
    terms = _cov_terms(abt, covar_cols) + [geno_col]
    f = "Y ~ " + " + ".join(terms) if terms else "Y ~ 1"
    m = _fit(f, abt, model_type)

    ci = m.conf_int().loc[geno_col]
    return {
        "Uni_p": float(m.pvalues[geno_col]),
        "Uni_Coef": round(float(m.params[geno_col]), 3),
        "CI_0.025": float(ci[0]),
        "CI_0.975": float(ci[1]),
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
    dict with LR_p, Anova_p, multi_Coef
    """
    cov_terms = _cov_terms(abt, covar_cols)
    alt_terms = cov_terms + geno_cols
    null_terms = cov_terms

    altf = "Y ~ " + " + ".join(alt_terms) if alt_terms else "Y ~ 1"
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    _lrstat, lrp = lrtest(nul, alt)
    _fstat, fp = anova(nul, alt, abt["Y"], model_type)

    # Only keep genotype terms that exist in model
    present = [c for c in geno_cols if c in alt.params.index]

    out = {
        "LR_p": float(lrp),
        "Anova_p": float(fp),
        "n_geno": len(present),
        "geno_terms": ",".join(present) if present else None,
    }

    if present:
        beta = alt.params.loc[present]
        cov = alt.cov_params().loc[present, present]

        # flatten beta
        for g in present:
            out[f"beta_{g}"] = float(beta[g])

        # flatten covariance (full matrix, not just upper triangle)
        for g1 in present:
            for g2 in present:
                out[f"cov_{g1}__{g2}"] = float(cov.loc[g1, g2])

    return out