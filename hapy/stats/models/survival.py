"""
Survival model engine using lifelines.CoxPHFitter.

- Fits Cox proportional hazards model.
- Returns hazard ratio (exp(coef)), p-value, CI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceError


def fit_cox_model(cox_df: pd.DataFrame):
    """Select interval-based fitting when start/stop columns are present."""
    if {"start", "stop"}.issubset(cox_df.columns):
        return CoxTimeVaryingFitter().fit(
            cox_df, id_col="sample_id", start_col="start", stop_col="stop", event_col="event")
    return CoxPHFitter().fit(
        cox_df.drop(columns=["sample_id"]), duration_col="time", event_col="event")


def _empty_cox_result(info) -> dict:
    """Keep result fields missing when no usable Cox fit is available."""
    result = dict.fromkeys(("P-value", "HR", "StdErr", "CI-lower", "CI-upper"), np.nan)
    result["INFO"] = info
    return result


def _cox_fit_info(model) -> str:
    """Check returned Cox estimates; lifelines has no public convergence flag.

    Warning-only convergence problems remain visible as lifelines warnings.
    These explicit checks do not capture or suppress warnings.
    """
    issues = []
    if not np.isfinite(model.params_).all():
        issues.append("non-finite coefficients")
    if not np.isfinite(model.standard_errors_).all():
        issues.append("non-finite standard errors")
    if not np.isfinite(model.log_likelihood_):
        issues.append("non-finite log-likelihood")
    return "; ".join(issues)


def _checked_cox_fit(cox_df):
    """Report expected numerical fit failures without hiding input/programming errors."""
    try:
        model = fit_cox_model(cox_df)
    except ConvergenceError:
        return None, "not converged"
    except np.linalg.LinAlgError:
        return None, "singular matrix"
    except FloatingPointError:
        return None, "numerical fitting error"
    return model, _cox_fit_info(model)


def fit_cox_univariate(cox_df: pd.DataFrame, feature: str) -> dict:
    """Fit a Cox model and return HR, coefficient SE, HR CI, P-value and INFO.

    Failed fitting/finite-value checks return missing estimates and a reason.
    INFO is NaN for fits passing these checks. Warnings are not captured.
    """
    cph, info = _checked_cox_fit(cox_df)
    if info:
        return _empty_cox_result(info)
    if feature not in cph.summary.index:
        return _empty_cox_result("feature not estimated")

    s = cph.summary.loc[feature]
    return {
        "P-value": float(s["p"]),
        "HR": float(s["exp(coef)"]),
        "StdErr": float(s["se(coef)"]),
        "CI-lower": float(s["exp(coef) lower 95%"]),
        "CI-upper": float(s["exp(coef) upper 95%"]),
        "INFO": np.nan,
    }


def fit_cox_omnibus(cox_df: pd.DataFrame, geno_cols: list[str]) -> dict:
    """Check full and null Cox fits before testing the genotype block jointly."""
    full, full_info = _checked_cox_fit(cox_df)
    null_cols = [c for c in cox_df.columns if c not in set(geno_cols)]
    null, null_info = _checked_cox_fit(cox_df[null_cols])
    issues = []
    for label, info in (("null model", null_info), ("full model", full_info)):
        if info:
            issues.append(f"{label}: {info}")
    if issues:
        return _empty_cox_result("; ".join(issues))
    result = _empty_cox_result(np.nan)
    result["P-value"] = lrtest_cox(
        float(full.log_likelihood_), float(null.log_likelihood_), dof=len(geno_cols),
    )
    return result


def lrtest_cox(alt_ll: float, null_ll: float, dof: int) -> float:
    """
    LR test p-value for Cox models from partial log-likelihoods.
    """
    from scipy import stats
    lr = 2.0 * (alt_ll - null_ll)
    return float(stats.chi2.sf(lr, dof))