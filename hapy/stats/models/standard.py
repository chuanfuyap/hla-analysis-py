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
from numbers import Real
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError

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


def fit_omnibus(abt: pd.DataFrame, geno_cols: list[str], covar_cols: list[str], model_type: str, *, null_cache=None, term_labels=None) -> dict:
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
    if dof <= 0:
        return {"INFO": "no additional information after conditioning", "TEST": "MULTI"}
    if lrstat < -1e-8:
        return {"INFO": "full model likelihood below null model", "TEST": "MULTI"}
    lrstat = max(0.0, lrstat)
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

    labels = term_labels or {}
    out["Beta"] = ";".join(
        f"{labels.get(column, column.removeprefix('G_'))}={alt.params[term]:.12g}"
        for column, term in zip(geno_cols, geno_terms)
    )
    out["StdErr"] = ";".join(
        f"{labels.get(column, column.removeprefix('G_'))}={alt.bse[term]:.12g}"
        for column, term in zip(geno_cols, geno_terms)
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

def fit_motif_omnibus(abt: pd.DataFrame, reduced_cols: list[str], expanded_cols: list[str],
                      covar_cols: list[str], model_type: str, *, null_cache=None,
                      check_rank: bool = False, term_labels=None) -> dict:
    """Compare covariates + expanded motifs against covariates + existing motifs.

    Null:
      Y ~ covars + reduced_cols... (e.g. VR motif)
    Alt:
      Y ~ covars + expanded_cols... (e.g. VRK motif)

    Both models use identical complete-case samples. The motif builder
    verifies that expanded groups reconstruct the existing groups and drops
    a compatible reference. Additional rank checking is optional.
    """
    abt = abt.dropna(subset=["Y", *covar_cols, *reduced_cols, *expanded_cols])
    if abt.empty:
        return {"INFO": "no complete-case samples", "TEST": "MOTIF"}
    cov_terms = _cov_terms(abt, covar_cols)
    reduced_terms = _cov_terms(abt, reduced_cols)
    expanded_terms = _cov_terms(abt, expanded_cols)
    null_terms = cov_terms + reduced_terms
    alt_terms = cov_terms + expanded_terms
    
    nullf = "Y ~ " + " + ".join(null_terms) if null_terms else "Y ~ 1"
    altf = "Y ~ " + " + ".join(alt_terms) if alt_terms else "Y ~ 1"
    if check_rank:
        from patsy import dmatrix
        baseline = np.asarray(dmatrix(nullf.split("~", 1)[1], abt))
        full = np.asarray(dmatrix(altf.split("~", 1)[1], abt))
        if np.linalg.matrix_rank(full) <= np.linalg.matrix_rank(baseline):
            return {"INFO": "no additional information after motif expansion", "TEST": "MOTIF"}
    alt = _fit(altf, abt, model_type)
    baseline_cols = covar_cols + reduced_cols
    baseline = abt[[*baseline_cols, "Y"]]
    reuse = null_cache is not None and null_cache.matches(
        baseline, model_type, (nullf, tuple(baseline_cols)))
    null = null_cache if reuse else fit_standard_null(abt, baseline_cols, model_type)
    issues = []
    for label, info in (("null model", null.info), ("full model", _fit_info(alt, model_type))):
        if info:
            issues.append(f"{label}: {info}")
    if issues:
        return {"INFO": "; ".join(issues), "TEST": "MOTIF"}
    lrstat = 2 * (alt.llf - null.log_likelihood)
    dof = alt.df_model - null.df_model
    if dof <= 0:
        return {"INFO": "no additional information after motif expansion", "TEST": "MOTIF"}
    if lrstat < -1e-8:
        return {"INFO": "full model likelihood below null model", "TEST": "MOTIF"}
    lrstat = max(0., lrstat)
    labels = term_labels or {}
    return {
        "P-value": float(stats.chi2.sf(lrstat, dof)), "Statistic": float(lrstat),
        "DoF": int(dof), "TEST": "MOTIF", "INFO": np.nan,
        "Beta": ";".join(f"{labels.get(c, c)}={alt.params[t]:.12g}" for c, t in zip(expanded_cols, expanded_terms)),
        "StdErr": ";".join(f"{labels.get(c, c)}={alt.bse[t]:.12g}" for c, t in zip(expanded_cols, expanded_terms)),
    }


def fit_position_comparison(abt, a_cols, b_cols, covar_cols, model_type, *, min_delta_bic=2.0,
                            criterion="bic", conditional_threshold=0.01):
    """Compare two groups of amino acid predictors using BIC and conditional tests.

    Models:
      A:  Y ~ covars + a_cols... (e.g. pos11)
      B:  Y ~ covars + b_cols... (e.g. pos13)
      AB: Y ~ covars + a_cols... + b_cols... (e.g. pos11 + pos13)

    Fit A, B and A+B with shared covariates and complete-case samples.
    Return BIC and conditional recommendations, test results and INFO.
    """
    return _fit_comparison(abt, a_cols, b_cols, a_cols + b_cols, covar_cols,
                           model_type, min_delta_bic=min_delta_bic,
                           criterion=criterion, conditional_threshold=conditional_threshold)


def fit_motif_comparison(abt, a_cols, b_cols, joint_cols, covar_cols, model_type,
                         *, min_delta_bic=2.0, joint_issue="", criterion="bic", conditional_threshold=0.01):
    """Compare alternative motifs against their combined position motif.

    Models:
      A:  Y ~ covars + a_cols... (e.g. pos11-71)
      B:  Y ~ covars + b_cols... (e.g. pos11-74)
      AB: Y ~ covars + joint_cols... (e.g. pos11-71-74)

    Return BIC and conditional recommendations, test results and INFO.
    All models use identical complete-case samples.
    """
    return _fit_comparison(abt, a_cols, b_cols, joint_cols, covar_cols,
                           model_type, min_delta_bic=min_delta_bic,
                           check_nested=True, joint_issue=joint_issue, criterion=criterion,
                           conditional_threshold=conditional_threshold)


def _fit_comparison(abt, a_cols, b_cols, joint_cols, covar_cols, model_type,
                    *, min_delta_bic, check_nested=False, joint_issue="",
                    criterion="bic", conditional_threshold=0.01):
    """Fit and summarise three models using one sample set."""
    _validate_comparison_options(criterion, conditional_threshold)
    if model_type not in ("linear", "logit"):
        raise ValueError("model_type must be 'linear' or 'logit'.")
    if not np.isfinite(min_delta_bic) or min_delta_bic < 0:
        raise ValueError("min_delta_bic must be finite and nonnegative.")
    if not a_cols or not b_cols:
        raise ValueError("No eligible residue predictors.")

    # Separate predictor groups prevent accidental conditioning on a tested column.
    columns = ["Y", *covar_cols, *a_cols, *b_cols]
    if len(set(columns)) != len(columns) or not abt.columns.is_unique:
        raise ValueError("Model columns must be unique and predictor groups must not overlap.")

    # Drop missing rows once so every comparison uses identical samples.
    extra = [c for c in joint_cols if c not in columns]
    abt = abt.loc[:, columns + extra].dropna()
    n = len(abt)
    if n == 0:
        raise ValueError("No complete-case samples.")

    # Construct model formulas
    cov_terms = _cov_terms(abt, covar_cols)
    a_terms = _cov_terms(abt, a_cols)
    b_terms = _cov_terms(abt, b_cols)

    def formula(terms):
        return "Y ~ " + " + ".join(terms) if terms else "Y ~ 1"

    formulas = {
        "A": formula(cov_terms + a_terms),
        "B": formula(cov_terms + b_terms),
        "AB": formula(cov_terms + _cov_terms(abt, joint_cols)),
    }

    # Fit models
    models = {}

    for label, f in formulas.items():
        if label == "AB" and joint_issue:
            models[label] = None
            continue
        try:
            model = _fit(f, abt, model_type)

            # Check model rank
            exog = model.model.exog
            if np.linalg.matrix_rank(exog) < exog.shape[1]:
                raise ValueError("rank-deficient design matrix")

            info = _fit_info(model, model_type)
            if info:
                raise ValueError(info)

            if label == "AB" and check_nested:
                # Conditional tests require both alternative design spaces to be nested.
                for reduced in models.values():
                    design = reduced.model.exog
                    recovered = exog @ np.linalg.lstsq(exog, design, rcond=None)[0]
                    if not np.allclose(recovered, design, rtol=1e-8, atol=1e-8):
                        raise ValueError("combined motif model does not contain both alternatives")
            models[label] = model

        except (ValueError, np.linalg.LinAlgError, PerfectSeparationError) as exc:
            if label != "AB":
                raise ValueError(f"Cannot fit {label}: {exc}") from exc

            # Joint model may be unidentifiable because of LD
            models[label] = None
            joint_issue = str(exc)

    a = models["A"]
    b = models["B"]
    ab = models["AB"]

    # Calculate BIC
    def bic(model):
        if model is None:
            return np.nan
        k = model.df_model + 1
        return float(-2 * model.llf + k * np.log(n))

    bic_a = bic(a)
    bic_b = bic(b)
    bic_ab = bic(ab)

    # Reciprocal conditional association tests
    p_b_given_a = np.nan
    p_a_given_b = np.nan
    df_b_given_a = np.nan
    df_a_given_b = np.nan

    if ab is not None:
        df_b_given_a = int(round(ab.df_model - a.df_model))
        df_a_given_b = int(round(ab.df_model - b.df_model))

        if model_type == "linear":
            if df_b_given_a > 0:
                _, p_b_given_a, _ = ab.compare_f_test(a)
            if df_a_given_b > 0:
                _, p_a_given_b, _ = ab.compare_f_test(b)

        else:
            if df_b_given_a > 0:
                _, p_b_given_a, _ = lrtest(a, ab)
            if df_a_given_b > 0:
                _, p_a_given_b, _ = lrtest(b, ab)

    # Model selection
    best_single = "A" if bic_a <= bic_b else "B"
    best_single_bic = min(bic_a, bic_b)

    delta_joint = best_single_bic - bic_ab

    if np.isfinite(delta_joint) and delta_joint >= min_delta_bic:
        bic_recommendation = "A+B"
    elif bic_a == bic_b or abs(bic_a - bic_b) < min_delta_bic:
        bic_recommendation = "A/B unresolved"
    else:
        bic_recommendation = best_single

    conditional_recommendation = _conditional_recommendation(
        p_b_given_a, p_a_given_b, df_b_given_a, df_a_given_b, conditional_threshold)
    recommendation = bic_recommendation if criterion == "bic" else conditional_recommendation

    return {
        "N_used": n,
        "BIC_A": bic_a,
        "BIC_B": bic_b,
        "BIC_AB": bic_ab,
        "Delta_BIC_B_minus_A": bic_b - bic_a,
        "Delta_BIC_joint_improvement": delta_joint,
        "P_B_given_A": p_b_given_a,
        "P_A_given_B": p_a_given_b,
        "DoF_B_given_A": df_b_given_a,
        "DoF_A_given_B": df_a_given_b,
        "BIC_RECOMMENDATION": bic_recommendation,
        "CONDITIONAL_RECOMMENDATION": conditional_recommendation,
        "RECOMMENDATION": recommendation,
        "INFO": joint_issue,
    }


def _validate_comparison_options(criterion, conditional_threshold):
    """Check how model recommendations will be selected."""
    if criterion not in ("bic", "conditional"):
        raise ValueError("criterion must be 'bic' or 'conditional'.")
    if (isinstance(conditional_threshold, bool) or not isinstance(conditional_threshold, Real)
            or not np.isfinite(conditional_threshold) or not 0 < conditional_threshold < 1):
        raise ValueError("conditional_threshold must be finite and strictly between 0 and 1.")


def _conditional_recommendation(p_b_given_a, p_a_given_b, df_b_given_a, df_a_given_b, threshold):
    """Choose positions using the two reciprocal conditional tests."""
    # Missing tests are not evidence that a position has no additional effect.
    if (not np.isfinite([p_b_given_a, p_a_given_b, df_b_given_a, df_a_given_b]).all()
            or df_b_given_a <= 0 or df_a_given_b <= 0
            or not 0 <= p_b_given_a <= 1 or not 0 <= p_a_given_b <= 1):
        return "A/B unresolved"
    if p_b_given_a < threshold:
        return "A+B" if p_a_given_b < threshold else "B"
    return "A" if p_a_given_b < threshold else "A/B unresolved"
