"""
Survival model engine using lifelines.CoxPHFitter.

- Fits Cox proportional hazards model.
- Returns hazard ratio (exp(coef)), p-value, CI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter


def fit_cox_model(cox_df: pd.DataFrame):
    """Select interval-based fitting when start/stop columns are present."""
    if {"start", "stop"}.issubset(cox_df.columns):
        return CoxTimeVaryingFitter().fit(
            cox_df, id_col="sample_id", start_col="start", stop_col="stop", event_col="event")
    return CoxPHFitter().fit(
        cox_df.drop(columns=["sample_id"]), duration_col="time", event_col="event")


def fit_cox_univariate(cox_df: pd.DataFrame, feature: str) -> dict:
    """
    Fit CoxPH with all columns in cox_df (excluding sample_id) and extract stats for `feature`.

    cox_df columns:
      sample_id, time, event, <features...>

    Returns
    -------
    dict with p, HR, CI_0.025, CI_0.975
    """
    cph = fit_cox_model(cox_df)

    if feature not in cph.summary.index:
        # can happen if feature got dropped (collinear) or name mismatch
        return {"P-value": np.nan, "HR": np.nan, "StdErr": np.nan, "CI-lower": np.nan, "CI-upper": np.nan}

    s = cph.summary.loc[feature]
    return {
        "P-value": float(s["p"]),
        "HR": float(s["exp(coef)"]),
        "StdErr": float(s["se(coef)"]),
        "CI-lower": float(s["exp(coef) lower 95%"]),
        "CI-upper": float(s["exp(coef) upper 95%"]),
    }


def lrtest_cox(alt_ll: float, null_ll: float, dof: int) -> float:
    """
    LR test p-value for Cox models from partial log-likelihoods.
    """
    from scipy import stats
    lr = 2.0 * (alt_ll - null_ll)
    return float(stats.chi2.sf(lr, dof))