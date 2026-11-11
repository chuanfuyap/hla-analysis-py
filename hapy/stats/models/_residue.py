"""Exact diploid contrast reversal shared by the existing association runners."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial


def can_reverse(abt, current, opposite, model, model_type, baseline_covars=()):
    """Require identical fitted rows, complementary dosages and a full-rank intercept design.

    The caller supplies the unchanged ABT/covariates of the original fit; no
    result is inferred across different complete-case samples or fitted models.
    """
    if model is None or not current.index.equals(abt.index) or not opposite.index.equals(abt.index):
        return False
    if not pd.Index(model.model.data.row_labels).equals(abt.index):
        return False
    baseline = ["Y", *baseline_covars]
    if not model.model.data.frame.loc[abt.index, baseline].equals(abt[baseline]):
        return False
    supported = ((model_type == "linear" and isinstance(model.model, OLS)) or
                 (model_type == "logit" and isinstance(model.model, GLM)
                  and isinstance(model.model.family, Binomial)))
    x = model.model.exog
    return bool(supported and "Intercept" in model.params.index
                and np.linalg.matrix_rank(x) == x.shape[1]
                and np.isfinite(current).all() and np.isfinite(opposite).all()
                and np.allclose(current + opposite, 2.0, rtol=0, atol=1e-8))


def reverse_standard(result):
    out = dict(result)
    for key in ("Beta", "Statistic"):
        if key in out:
            out[key] = -out[key]
    if "CI-lower" in out and "CI-upper" in out:
        out["CI-lower"], out["CI-upper"] = -out["CI-upper"], -out["CI-lower"]
    return out


def reverse_interaction(result, model, a_col, b_col, reverse_a=False, reverse_b=False):
    """Transform all reported coefficients and main-effect P-values using covariance.

    Only numeric, single-column, uncentred pairwise terms are eligible. The
    runner validates dosage complementarity and identical modelling rows first.
    """
    names = list(model.params.index)
    interaction = f"{a_col}:{b_col}"
    if not all(c in names for c in (a_col, b_col, interaction, "Intercept")):
        return None
    a, b, i, intercept = [names.index(c) for c in (a_col, b_col, interaction, "Intercept")]
    transform = np.eye(len(names))
    for side in ("a", "b"):
        if not (reverse_a if side == "a" else reverse_b):
            continue
        selected, other = (a, b) if side == "a" else (b, a)
        step = np.eye(len(names))
        step[intercept, selected] = 2.0
        step[selected, selected] = -1.0
        step[other, i] = 2.0
        step[i, i] = -1.0
        transform = step @ transform
    beta = transform @ np.asarray(model.params)
    cov = transform @ np.asarray(model.cov_params()) @ transform.T
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    distribution = stats.t(df=model.df_resid) if model.use_t else stats.norm
    p = 2 * distribution.sf(np.abs(beta / se))
    width = distribution.ppf(.975) * se[i]
    out = dict(result)
    out.update({"Beta": float(beta[i]), "StdErr": float(se[i]),
                "Statistic": float(beta[i]/se[i]), "P-value": float(p[i]),
                "CI-lower": float(beta[i]-width), "CI-upper": float(beta[i]+width),
                "Variant_Beta": float(beta[a]), "Variant_p": float(p[a]),
                "Partner_Beta": float(beta[b]), "Partner_p": float(p[b])})
    return out
