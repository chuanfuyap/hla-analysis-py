from typing import List
import numpy as np
import pandas as pd
from scipy import stats

def _inverse_variance_meta_analysis(betas, ses):
    """
    Perform inverse-variance meta-analysis on a list of effect sizes and standard errors.
    """
    betas = np.array(betas)
    ses = np.array(ses)

    # Validate inputs
    if betas.shape != ses.shape:
        raise ValueError("The lists 'betas' and 'ses' must have the same length.")
    if np.any(~np.isfinite(betas)):
        raise ValueError("All betas must be finite.")
    if np.any(~np.isfinite(ses)) or np.any(ses <= 0):
        raise ValueError("All standard errors must be positive and finite.")

    # Calculate weights (inverse of variance)
    variances = ses ** 2
    weights = 1 / variances
    weight_sum = np.sum(weights)

    # Compute the combined effect size (meta_beta)
    meta_beta = np.sum(weights * betas) / weight_sum

    # Compute the combined variance and standard error (meta_se)
    meta_variance = 1 / weight_sum
    meta_se = np.sqrt(meta_variance)

    # Calculate the z-score
    z_score = meta_beta / meta_se

    # Calculate the two-sided p-value
    p_value = 2 * stats.norm.sf(abs(z_score))
    log_p =  (stats.norm.logsf(abs(z_score)) + np.log(2)) / np.log(10)
    return meta_beta, meta_se, p_value, log_p, z_score*z_score


# One input table represents one independent cohort. Sample counts are optional
# metadata; when present they must be finite and nonnegative (missing is allowed).
_SAMPLE_TOTALS = {"N_total": "Total N", "N_case": "Total Case", "N_control": "Total Control"}
_SAMPLE_COLUMNS = ("N_used", "N", "N_total", "N_case", "N_control")


def _prepare_studies(studies, required, variant_col):
    frames = []
    for index, study in enumerate(studies):
        if not isinstance(study, pd.DataFrame):
            raise TypeError(f"Study {index + 1} must be a pandas DataFrame.")
        for column in required:
            if column not in study:
                raise ValueError(f"Column '{column}' not found in study {index + 1}.")
        frame = study.dropna(subset=[variant_col]).copy()
        if frame[variant_col].duplicated().any():
            raise ValueError(f"Study {index + 1} contains duplicate variant IDs.")
        for column in _SAMPLE_COLUMNS:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
                values = frame[column].dropna().to_numpy(dtype=float)
                if np.any(~np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"Study {index + 1}: {column} must be finite and nonnegative.")
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=required))


def _total_columns(combined):
    return {source: target for source, target in _SAMPLE_TOTALS.items() if source in combined}


def _sample_totals(group, columns):
    # Do not report a partial sum as a complete total when a cohort lacks counts.
    return {target: group[source].sum(min_count=len(group))
            for source, target in columns.items()}


def inv_var_meta_studies(studies: List[pd.DataFrame], beta: str, se: str,
                         variant_col: str, prefix: str) -> pd.DataFrame:
    """Fixed-effect inverse-variance meta-analysis of aligned cohort effects.

    Each input table represents an independent cohort and must contain at most
    one row per variant. Results require at least two contributing cohorts.
    Sample totals are calculated per variant; the coefficient test is two-sided.
    """
    combined = _prepare_studies(studies, [variant_col, beta, se], variant_col)
    totals = _total_columns(combined)
    columns = [variant_col, "N studies", f"{prefix}_BETA", f"{prefix}_SE",
               f"{prefix}_chisquared", f"{prefix}_P", f"{prefix}_log10P", *totals.values()]
    rows = []
    for variant, group in combined.groupby(variant_col, sort=False):
        if len(group) < 2:
            continue
        betas = pd.to_numeric(group[beta], errors="raise").to_numpy(dtype=float)
        ses = pd.to_numeric(group[se], errors="raise").to_numpy(dtype=float)
        meta_beta, meta_se, p_value, log_p, chisquared = _inverse_variance_meta_analysis(betas, ses)
        rows.append({
            variant_col: variant, "N studies": len(group),
            f"{prefix}_BETA": meta_beta, f"{prefix}_SE": meta_se,
            f"{prefix}_chisquared": chisquared, f"{prefix}_P": p_value,
            f"{prefix}_log10P": -log_p, **_sample_totals(group, totals),
        })
    return pd.DataFrame(rows, columns=columns)


def _stouffer_weights(combined):
    weights = pd.Series(1.0, index=combined.index)
    binary = pd.Series(False, index=combined.index)
    if "N_case" in combined and "N_control" in combined:
        cases, controls = combined["N_case"], combined["N_control"]
        binary = cases.gt(0) & controls.gt(0)
        effective_n = 4.0 / (1.0 / cases.loc[binary] + 1.0 / controls.loc[binary])
        weights.loc[binary] = np.sqrt(effective_n)

    # Prefer the actual modelling sample size. N_total also supports HAPY
    # outputs where N_used was omitted because all samples were used.
    sample_n = pd.Series(np.nan, index=combined.index)
    for column in ("N_used", "N", "N_total"):
        if column in combined:
            available = sample_n.isna() & combined[column].gt(0)
            sample_n.loc[available] = combined.loc[available, column]
    quantitative = ~binary & sample_n.notna()
    weights.loc[quantitative] = np.sqrt(sample_n.loc[quantitative])
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Sample sizes produce non-finite Stouffer weights.")
    return weights


def stouffers_meta_studies(studies: List[pd.DataFrame], pval: str,
                           variant_col: str, prefix: str) -> pd.DataFrame:
    """Combine independent-cohort omnibus p-values with weighted Stouffer.

    The evidence transform is z_i = norm.isf(p_i), followed by
    Z = sum(w_i*z_i) / sqrt(sum(w_i**2)) and P = norm.sf(Z). This does not
    impose an effect direction on the cohort tests. Weights are sqrt(N_eff)
    for valid case/control counts, otherwise sqrt(N_used), sqrt(N), or
    sqrt(N_total), in that order. Rows with no usable counts use weight 1.
    Weights must be chosen independently of the observed p-values.

    Each input table represents one independent cohort with at most one row per
    variant. Missing variant IDs or p-values are omitted; invalid p-values raise
    an error. Inputs require 0 < p <= 1 and are not clipped. An exact p=1 maps
    to -infinity, giving combined P=1, consistent with the Stouffer limit.
    At least two contributing cohorts are required for each reported variant.

    Output: N studies, <prefix>_Z, <prefix>_P and <prefix>_log10P (-log10 P),
    plus available per-variant cohort totals. Incomplete sample totals are NaN.
    Input tables are not modified.
    """
    combined = _prepare_studies(studies, [variant_col, pval], variant_col)
    totals = _total_columns(combined)
    columns = [variant_col, "N studies", f"{prefix}_Z", f"{prefix}_P",
               f"{prefix}_log10P", *totals.values()]
    combined[pval] = pd.to_numeric(combined[pval], errors="raise")
    combined = combined.dropna(subset=[pval]).copy()
    pvalues = combined[pval].to_numpy(dtype=float)
    if np.any(~np.isfinite(pvalues)):
        raise ValueError("All p-values must be finite.")
    if np.any((pvalues <= 0) | (pvalues > 1)):
        raise ValueError("All p-values must be in the interval (0, 1].")
    combined["_z"] = stats.norm.isf(pvalues)
    combined["_w"] = _stouffer_weights(combined)
    rows = []
    for variant, group in combined.groupby(variant_col, sort=False):
        if len(group) < 2:
            continue
        weights = group["_w"].to_numpy(dtype=float)
        # Scaling preserves relative weights and avoids squaring huge counts.
        weights = weights / weights.max()
        z_meta = float(np.dot(weights, group["_z"].to_numpy(dtype=float))
                       / np.linalg.norm(weights))
        p_meta = float(stats.norm.sf(z_meta))
        log_p = float(stats.norm.logsf(z_meta) / np.log(10.0))
        rows.append({
            variant_col: variant, "N studies": len(group),
            f"{prefix}_Z": z_meta, f"{prefix}_P": p_meta,
            f"{prefix}_log10P": -log_p, **_sample_totals(group, totals),
        })
    return pd.DataFrame(rows, columns=columns)
