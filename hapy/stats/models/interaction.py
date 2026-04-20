"""
Interaction model engine.

This module provides two interaction testing helpers used by the interaction runner:

1) Pairwise interaction (single A column × single B column)
   - Tests an interaction term a:b in a linear or logistic regression.
   - Can return either:
     (a) Wald test p-value for the interaction coefficient only (fast; default), or
     (b) likelihood-ratio / ANOVA-style nested model comparison (slower; optional).

2) Block omnibus interaction (anchor column × multi-column block)
   - Intended for AA-position encodings where a "variant" expands into many columns.
   - Fits a model with main effects for anchor + all block columns, and compares:
       null: Y ~ covars + anchor + block_cols
        alt: Y ~ covars + anchor + block_cols + (anchor:block_cols)
   - Returns omnibus p-values (LR_p, Anova_p), and optionally some coefficient summaries.

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

from .common import lrtest, anova, categorical_term


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
) -> dict:
    """
    Fit a pairwise interaction model and return interaction statistics.

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
    dict with keys
        Always:
        - I_term: string term used for the interaction (e.g., "A:B" or "C(A):B")
        - I_p: Wald p-value for the interaction coefficient in the full model
        - I_coef: interaction coefficient estimate
        - I_CI_0.025 / I_CI_0.975: confidence interval bounds for I_coef

        Additionally if compute_lrt=True:
        - LR_p: likelihood ratio test p-value comparing base vs full
        - Anova_p: ANOVA-style p-value comparing base vs full

        If compute_lrt=False:
        - LR_p, Anova_p are returned as NaN (keeps schema stable; can be dropped later).
    """
    cov_terms = _terms(abt, baseline_covars)
    a_t = _term(abt, a_col)
    b_t = _term(abt, b_col)

    base_terms = cov_terms + [a_t, b_t]
    basef = "Y ~ " + " + ".join(base_terms) if base_terms else "Y ~ 1"
    altf = basef + f" + {a_t}:{b_t}"

    alt = _fit(altf, abt, model_type)

    iterm = f"{a_t}:{b_t}"

    # NOTE: This assumes the interaction term appears as a single coefficient.
    # For categorical expansions, statsmodels may create multiple interaction params.
    # In that case, this can KeyError; the runner typically uses numeric dosage columns.
    ci = alt.conf_int().loc[iterm]

    out = {
        "I_term": iterm,
        "I_p": float(alt.pvalues[iterm]),
        "I_coef": float(alt.params[iterm]),
        "I_CI_0.025": float(ci[0]),
        "I_CI_0.975": float(ci[1]),
    }

    if compute_lrt:
        nul = _fit(basef, abt, model_type)
        _lrstat, lrp = lrtest(nul, alt)
        _fstat, fp = anova(nul, alt, abt["Y"], model_type)
        out["LR_p"] = float(lrp)
        out["Anova_p"] = float(fp)
    else:
        out["LR_p"] = np.nan
        out["Anova_p"] = np.nan

    return out


def fit_block_omnibus_interaction(
    abt: pd.DataFrame,
    anchor_col: str,
    block_cols: list[str],
    baseline_covars: list[str],
    model_type: str,
) -> dict:
    """
    Fit an omnibus interaction model: anchor × block.

    This is used when one "variant" is represented by multiple columns (a block),
    e.g., AA position k-1 encoding. Instead of reporting many individual interaction
    coefficients, we report an omnibus nested-model p-value for the entire block.

    Model
    -----
    null: Y ~ covars + anchor + block_cols
     alt: Y ~ covars + anchor + block_cols + (anchor:block_cols)

    Parameters
    ----------
    abt:
        Analysis-by-table containing Y, anchor_col, block_cols, and baseline covariates.
    anchor_col:
        Column name for the anchor predictor.
    block_cols:
        List of column names representing the block.
    baseline_covars:
        Covariates included in both null and alt models.
    model_type:
        "linear" or "logit".

    Returns
    -------
    dict with keys
        - LR_p: likelihood ratio test p-value comparing null vs alt
        - Anova_p: ANOVA-style p-value comparing null vs alt
        - n_I: number of interaction coefficients actually present in the fitted model
        - I_terms: comma-separated list of present interaction coefficient names
        - beta_<term>: coefficient for each present interaction term
        - cov_<t1>__<t2>: covariance elements between present interaction terms
          (can be many columns; intended for debugging/advanced downstream usage)
    """
    cov_terms = _terms(abt, baseline_covars)
    anchor_t = _term(abt, anchor_col)
    block_terms = _terms(abt, block_cols)

    null_terms = cov_terms + [anchor_t] + block_terms
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    interaction_terms = [f"{anchor_t}:{bt}" for bt in block_terms]
    altf = nullf + (" + " + " + ".join(interaction_terms) if interaction_terms else "")

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    lrstat, lrp, dof = lrtest(nul, alt)
    #fstat, fp = anova(nul, alt, abt["Y"], model_type)

    # Find the interaction coefficients actually present in the fitted model.
    # This is safer than assuming the raw formula strings match alt.params.index exactly.
    present: list[str] = []
    for name in alt.params.index:
        if ":" not in name:
            continue
        has_anchor = anchor_t in name
        has_block = any(bt in name for bt in block_terms)
        if has_anchor and has_block:
            present.append(name)

    out: dict = {
        "LR_p": float(lrp),
        "LR_stat": float(lrstat),
        "DoF" : int(dof)
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