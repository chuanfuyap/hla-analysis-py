"""
Interaction model engine.

This module provides two interaction testing helpers used by the interaction runner:

1) Pairwise interaction (single A column × single B column)
   - Tests an interaction term a:b in a linear or logistic regression.
   - Can return either:
     (a) Wald test p-value for the interaction coefficient only (fast; default), or
     (b) likelihood-ratio / ANOVA-style nested model comparison (slower; optional).

2) Block omnibus interaction (partner column × multi-column block)
   - Intended for AA-position encodings where a "variant" expands into many columns.
   - Fits a model with main effects for partner + all block columns, and compares:
       null: Y ~ covars + partner + block_cols
        alt: Y ~ covars + partner + block_cols + (partner:block_cols)
   - Returns P-value, Statistic, DoF and INFO for the omnibus interaction test.

Notes on categorical variables
------------------------------
We use statsmodels formula terms and automatically wrap non-numeric columns in C(...)
via common.categorical_term(). This means:

- If a_col or b_col (or baseline covariates) are categorical, statsmodels will expand
  them into multiple dummy variables.
- In that case, the "interaction" can expand into multiple coefficients.
- The pairwise helper is primarily intended for numeric dosage-like columns; it will
  still work with categorical columns in some cases, but term naming/expansion can be
  more complex.

Returned keys are intentionally "CSV-friendly" (scalars + strings).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .common import lrtest, anova, categorical_term, _fit_info


def _failed_result(info: str, *, omnibus: bool = False) -> dict:
    """Retain missing statistics and a reason when an interaction fit is unusable."""
    columns = ("P-value", "Statistic", "DoF") if omnibus else (
        "P-value", "Statistic", "Beta", "StdErr", "CI-lower", "CI-upper",
        "Variant_p", "Variant_Beta", "Partner_p", "Partner_Beta",
    )
    result = dict.fromkeys(columns, np.nan)
    result["INFO"] = info
    return result


def _fit(formula: str, abt: pd.DataFrame, model_type: str):
    """
    Fit a statsmodels model from a Patsy formula.

    Parameters
    ----------
    formula:
        Patsy/Statsmodels formula, e.g. "Y ~ age + sex + G + E + G:E".
    abt:
        Analysis-by-table. Must contain Y and all referenced columns.
    model_type:
        "linear" or "logit".

    Returns
    -------
    statsmodels results object
    """
    if model_type == "logit":
        return smf.glm(formula=formula, data=abt, family=sm.families.Binomial()).fit(disp=0)
    return smf.ols(formula=formula, data=abt).fit()


def _term(abt: pd.DataFrame, col: str) -> str:
    """
    Convert a column name into a formula term.

    Uses categorical_term() to wrap categoricals with C(col).
    """
    return categorical_term(col, abt[col].dtype)


def _terms(abt: pd.DataFrame, cols: list[str]) -> list[str]:
    """Vectorized _term()."""
    return [_term(abt, c) for c in cols]


def fit_pairwise_interaction(
    abt: pd.DataFrame,
    a_col: str,
    b_col: str,
    baseline_covars: list[str],
    model_type: str,
    *,
    compute_lrt: bool = False,
    _return_model: bool = False,
) -> dict:
    """
    Fit a pairwise interaction model and return interaction statistics.

    Failed convergence/finiteness checks return missing statistics and an INFO
    reason. Successful fits have INFO set to NaN.

    Model
    -----
    Let covars be baseline_covars.

    Base (main-effects) model:
        Y ~ covars + a + b

    Full (interaction) model:
        Y ~ covars + a + b + a:b

    Parameters
    ----------
    abt:
        Analysis-by-table containing:
        - "Y" outcome
        - columns for a_col, b_col
        - baseline covariate columns
    a_col, b_col:
        Column names inside abt representing the two interacting predictors.
    baseline_covars:
        List of covariate column names to include in both base and full models.
    model_type:
        "linear" (OLS) or "logit" (GLM Binomial).
    compute_lrt:
        If False (default), return only the Wald test p-value for the interaction
        coefficient from the full model (fast; fits only the full model).

        If True, also fit the base model and compute nested-model comparison p-values:
        - LR_p via lrtest()
        - Anova_p via anova()

    Returns
    -------
    dict
        P-value, Statistic, Beta, StdErr, CI-lower and CI-upper describe the
        interaction coefficient. Variant_p/Variant_Beta and Partner_p/Partner_Beta
        are main-effect estimates from the full model. INFO is NaN when fit
        checks pass; otherwise estimates are missing and INFO gives a reason.
        compute_lrt=True additionally returns LR_p and Anova_p comparisons.
    """
    # Private worker-only access permits exact contrast/covariance transforms.
    cov_terms = _terms(abt, baseline_covars)
    a_t = _term(abt, a_col)
    b_t = _term(abt, b_col)

    base_terms = cov_terms + [a_t, b_t]
    basef = "Y ~ " + " + ".join(base_terms) if base_terms else "Y ~ 1"
    altf = basef + f" + {a_t}:{b_t}"

    alt = _fit(altf, abt, model_type)

    iterm = f"{a_t}:{b_t}"
    info = _fit_info(alt, model_type)
    if info:
        out = _failed_result(info)
        return (out, alt) if _return_model else out
    if compute_lrt:
        nul = _fit(basef, abt, model_type)
        info = _fit_info(nul, model_type)
        if info:
            out = _failed_result(f"null model: {info}")
            return (out, alt) if _return_model else out

    # NOTE: This assumes the interaction term appears as a single coefficient.
    # For categorical expansions, statsmodels may create multiple interaction params.
    # In that case, this can KeyError; the runner typically uses numeric dosage columns.
    ci = alt.conf_int().loc[iterm]

    out = {
        "P-value": float(alt.pvalues[iterm]),
        "Statistic": float(alt.tvalues[iterm]),
        "Beta": float(alt.params[iterm]),
        "StdErr": float(alt.bse[iterm]),
        "CI-lower": float(ci.iloc[0]),
        "CI-upper": float(ci.iloc[1]),
        "Variant_p": float(alt.pvalues[a_t]),
        "Variant_Beta": float(alt.params[a_t]),
        "Partner_p": float(alt.pvalues[b_t]),
        "Partner_Beta": float(alt.params[b_t]),
        "INFO": np.nan,
    }

    if compute_lrt:
        _lrstat, lrp, _dof = lrtest(nul, alt)
        _fstat, fp = anova(nul, alt, abt["Y"], model_type)
        out["LR_p"] = float(lrp)
        out["Anova_p"] = float(fp)

    return (out, alt) if _return_model else out


def fit_block_omnibus_interaction(
    abt: pd.DataFrame,
    partner_col: str,
    block_cols: list[str],
    baseline_covars: list[str],
    model_type: str,
) -> dict:
    """
    Fit an omnibus interaction model: partner × block.

    Both null and full models must pass convergence/finiteness checks. Failed
    checks return missing statistics and an INFO reason identifying the model.

    This is used when one "variant" is represented by multiple columns (a block),
    e.g., AA position k-1 encoding. Instead of reporting many individual interaction
    coefficients, we report an omnibus nested-model p-value for the entire block.

    Model
    -----
    null: Y ~ covars + partner + block_cols
     alt: Y ~ covars + partner + block_cols + (partner:block_cols)

    Parameters
    ----------
    abt:
        Analysis-by-table containing Y, partner_col, block_cols, and baseline covariates.
    partner_col:
        Column name for the partner predictor.
    block_cols:
        List of column names representing the block.
    baseline_covars:
        Covariates included in both null and alt models.
    model_type:
        "linear" or "logit".

    Returns
    -------
    dict
        P-value, Statistic, DoF and INFO. Failed checks return missing
        statistics and a reason; successful fits have INFO set to NaN.
    """
    cov_terms = _terms(abt, baseline_covars)
    partner_t = _term(abt, partner_col)
    block_terms = _terms(abt, block_cols)

    null_terms = cov_terms + [partner_t] + block_terms
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    interaction_terms = [f"{partner_t}:{bt}" for bt in block_terms]
    altf = nullf + (" + " + " + ".join(interaction_terms) if interaction_terms else "")

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    issues = []
    for label, model in (("null model", nul), ("full model", alt)):
        info = _fit_info(model, model_type)
        if info:
            issues.append(f"{label}: {info}")
    if issues:
        return _failed_result("; ".join(issues), omnibus=True)

    lrstat, lrp, dof = lrtest(nul, alt)
    #fstat, fp = anova(nul, alt, abt["Y"], model_type)

    # Find the interaction coefficients actually present in the fitted model.
    # This is safer than assuming the raw formula strings match alt.params.index exactly.
    present: list[str] = []
    for name in alt.params.index:
        if ":" not in name:
            continue
        has_partner = partner_t in name
        has_block = any(bt in name for bt in block_terms)
        if has_partner and has_block:
            present.append(name)

    out: dict = {
        "P-value": float(lrp),
        "Statistic": float(lrstat),
        "DoF" : int(dof),
        "INFO": np.nan,
    #    "Anova_p": float(fp),
    #    "n_I": len(present),
    #    "I_terms": ",".join(present) if present else None,
    }

    # if present:
    #     beta = alt.params.loc[present]
    #     cov = alt.cov_params().loc[present, present]

    #     for t in present:
    #         out[f"beta_{t}"] = float(beta[t])

    #     for t1 in present:
    #         for t2 in present:
    #             out[f"cov_{t1}__{t2}"] = float(cov.loc[t1, t2])

    return out