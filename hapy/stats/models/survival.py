"""
Survival model engine using lifelines.CoxPHFitter.

- Fits Cox proportional hazards model.
- Returns hazard ratio (exp(coef)), p-value, CI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_univariate(cox_df: pd.DataFrame, feature: str) -> dict:
    """
    Fit CoxPH with all columns in cox_df (excluding sample_id) and extract stats for `feature`.

    cox_df columns:
      sample_id, time, event, <features...>

    Returns
    -------
    dict with p, HR, CI_0.025, CI_0.975
    """
    abt = cox_df.drop(columns=["sample_id"]).copy()

    cph = CoxPHFitter()
    cph = cph.fit(abt, duration_col="time", event_col="event")

    if feature not in cph.summary.index:
        # can happen if feature got dropped (collinear) or name mismatch
        return {"p": np.nan, "HR": np.nan, "CI_0.025": np.nan, "CI_0.975": np.nan}

    s = cph.summary.loc[feature]
    return {
        "p": float(s["p"]),
        "HR": float(s["exp(coef)"]),
        "StdErr": float(s["se(coef)"]),
        "CI_0.025": float(s["exp(coef) lower 95%"]),
        "CI_0.975": float(s["exp(coef) upper 95%"]),
    }


def lrtest_cox(alt_ll: float, null_ll: float, dof: int) -> float:
    """
    LR test p-value for Cox models from partial log-likelihoods.
    """
    from scipy import stats
    lr = 2.0 * (alt_ll - null_ll)
    return float(stats.chi2.sf(lr, dof))