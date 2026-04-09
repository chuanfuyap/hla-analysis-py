"""
Interaction model engine.
"""

from __future__ import annotations
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .common import lrtest, anova, categorical_term


def _fit(formula: str, abt: pd.DataFrame, model_type: str):
    if model_type == "logit":
        return smf.glm(formula=formula, data=abt, family=sm.families.Binomial()).fit(disp=0)
    return smf.ols(formula=formula, data=abt).fit()


def _term(abt: pd.DataFrame, col: str) -> str:
    return categorical_term(col, abt[col].dtype)


def _terms(abt: pd.DataFrame, cols: list[str]) -> list[str]:
    return [_term(abt, c) for c in cols]


def fit_pairwise_interaction(
    abt: pd.DataFrame,
    a_col: str,
    b_col: str,
    baseline_covars: list[str],
    model_type: str,
) -> dict:
    cov_terms = _terms(abt, baseline_covars)
    a_t = _term(abt, a_col)
    b_t = _term(abt, b_col)

    null_terms = cov_terms + [a_t, b_t]
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"
    altf = nullf + f" + {a_t}:{b_t}"

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    iterm = f"{a_t}:{b_t}"
    ci = alt.conf_int().loc[iterm]

    _lrstat, lrp = lrtest(nul, alt)
    _fstat, fp = anova(nul, alt, abt["Y"], model_type)

    return {
        "I_term": iterm,
        "I_p": float(alt.pvalues[iterm]),
        "I_coef": float(alt.params[iterm]),
        "I_CI_0.025": float(ci[0]),
        "I_CI_0.975": float(ci[1]),
        "LR_p": float(lrp),
        "Anova_p": float(fp),
    }


def fit_block_omnibus_interaction(
    abt: pd.DataFrame,
    anchor_col: str,
    block_cols: list[str],
    baseline_covars: list[str],
    model_type: str,
) -> dict:
    cov_terms = _terms(abt, baseline_covars)
    anchor_t = _term(abt, anchor_col)
    block_terms = _terms(abt, block_cols)

    null_terms = cov_terms + [anchor_t] + block_terms
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    interaction_terms = [f"{anchor_t}:{bt}" for bt in block_terms]
    altf = nullf + (" + " + " + ".join(interaction_terms) if interaction_terms else "")

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    _lrstat, lrp = lrtest(nul, alt)
    _fstat, fp = anova(nul, alt, abt["Y"], model_type)

    # Find the interaction coefficients actually present in the fitted model.
    # This is safer than assuming the raw formula strings match alt.params.index exactly.
    present = []
    for name in alt.params.index:
        if ":" not in name:
            continue
        has_anchor = anchor_t in name
        has_block = any(bt in name for bt in block_terms)
        if has_anchor and has_block:
            present.append(name)

    out = {
        "LR_p": float(lrp),
        "Anova_p": float(fp),
        "n_I": len(present),
        "I_terms": ",".join(present) if present else None,
    }

    if present:
        beta = alt.params.loc[present]
        cov = alt.cov_params().loc[present, present]

        for t in present:
            out[f"beta_{t}"] = float(beta[t])

        for t1 in present:
            for t2 in present:
                out[f"cov_{t1}__{t2}"] = float(cov.loc[t1, t2])

    return out

def fit_block_omnibus_interaction(
    abt: pd.DataFrame,
    anchor_col: str,
    block_cols: list[str],
    baseline_covars: list[str],
    model_type: str,
) -> dict:
    cov_terms = _terms(abt, baseline_covars)
    anchor_t = _term(abt, anchor_col)
    block_terms = _terms(abt, block_cols)

    null_terms = cov_terms + [anchor_t] + block_terms
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"

    interaction_terms = [f"{anchor_t}:{bt}" for bt in block_terms]
    altf = nullf + (" + " + " + ".join(interaction_terms) if interaction_terms else "")

    alt = _fit(altf, abt, model_type)
    nul = _fit(nullf, abt, model_type)

    _lrstat, lrp = lrtest(nul, alt)
    _fstat, fp = anova(nul, alt, abt["Y"], model_type)

    coef_parts = []
    for t in interaction_terms:
        if t in alt.params.index:
            coef_parts.append(f"{t}={float(alt.params[t]):.6g}")
    coef_str = ", ".join(coef_parts) if coef_parts else ""

    return {
        "LR_p": float(lrp),
        "Anova_p": float(fp),
        "I_terms": ", ".join(interaction_terms),
        "I_block_coefs": coef_str,
    }