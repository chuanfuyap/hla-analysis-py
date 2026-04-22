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


def inv_var_meta_studies(studies: List[pd.DataFrame], beta: str, se: str, variant_col: str, prefix: str) -> pd.DataFrame:
    """
    Computes inverse-variance weighted meta-analysis for all available variants in the input dataframes.

    Parameters:
        studies: list of sumstats in dataframes
        beta: column name for effect size
        se: column name for standard error
        variant_col: column name for variants
        prefix: prefix for output column names

    Returns:
        DataFrame with meta-analysis results.
    """
    # Concatenate all studies into one DataFrame
    combined = pd.concat(studies, ignore_index=True)
    has_binary_cols = ("N_case" in combined.columns) and ("N_control" in combined.columns)
    
    # Validate required columns
    required_cols = [variant_col, beta, se]
    for col in required_cols:
        if col not in combined.columns:
            raise ValueError(f"Column '{col}' not found in the input dataframes.")
    
    # Group by variant
    grouped = combined.groupby(variant_col) 
    total_n = combined["N_total"].unique().sum()
    total_case, total_control = None, None
    if has_binary_cols:
        total_case = combined["N_case"].unique().sum()
        total_control = combined["N_control"].unique().sum()

    # We'll filter groups to only those with more than one study
    # Then apply meta-analysis to those groups
    def meta_func(df):
        if len(df) < 2:
            return pd.Series({
                "N studies": np.nan,
                f"{prefix}_BETA": np.nan,
                f"{prefix}_SE": np.nan,
                f"{prefix}_chisquared": np.nan, 
                f"{prefix}_P": np.nan,
                f"{prefix}_log10P": np.nan
            })
        
        betas = df[beta].values
        ses = df[se].values

        meta_beta, meta_se, p_value, log_p, chisquared = _inverse_variance_meta_analysis(betas, ses)
        return pd.Series({
            "N studies": len(betas),
            f"{prefix}_BETA": meta_beta,
            f"{prefix}_SE": meta_se,
            f"{prefix}_chisquared": chisquared,
            f"{prefix}_P": p_value,
            f"{prefix}_log10P": -log_p
        })
    
    # Apply the meta_func to each group
    results = grouped.apply(meta_func).reset_index()
    results["Total N"] = total_n
    if has_binary_cols:
        results['Total Case'] = total_case
        results["Total Control"] = total_control
    results = results.dropna(subset=["N studies"])

    # Drop rows that do not have at least two studies
    results = results.dropna(subset=["N studies"])

    return results


def stouffers_meta_studies(
    studies: List[pd.DataFrame],
    pval: str,
    variant_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Combine per-study p-values across studies using weighted Stouffer's method.

    Weighting rule
    --------------
    For each row:
    - if both N_case and N_control are present and non-missing, use sqrt(N_eff),
      where N_eff = 4 / (1/N_case + 1/N_control)
    - else if N is present and non-missing, use sqrt(N)
    - else use equal weight = 1

    This version is appropriate for non-directional p-values, such as omnibus
    LR-test or ANOVA p-values.

    Parameters
    ----------
    studies : list[pd.DataFrame]
        List of per-study result tables.
    pval : str
        Name of the column containing the per-study p-value.
    variant_col : str
        Name of the column identifying the tested variant / marker / block.
    prefix : str
        Prefix used for output column names.

    Returns
    -------
    pd.DataFrame
        One row per variant, containing:
        - number of studies combined
        - meta-analytic Z-statistic
        - meta-analytic p-value
        - -log10(p)
    """

    # Stack all studies into one table
    combined = pd.concat(studies, ignore_index=True)

    # Check essential columns
    required_cols = [variant_col, pval]
    for col in required_cols:
        if col not in combined.columns:
            raise ValueError(f"Column '{col}' not found in the input dataframes.")

    # Work on a copy so we do not modify the original input tables
    combined = combined.copy()

    # Convert p-values to numeric and drop missing rows
    combined[pval] = pd.to_numeric(combined[pval], errors="coerce")
    combined = combined.dropna(subset=[variant_col, pval])

    # Validate p-values
    if np.any(~np.isfinite(combined[pval])):
        raise ValueError("All p-values must be finite.")
    if np.any((combined[pval] <= 0) | (combined[pval] > 1)):
        raise ValueError("All p-values must be in the interval (0, 1].")

    # Convert omnibus p-values to unsigned Z-scores
    # We use a two-sided mapping because omnibus tests are non-directional
    eps = 1e-300
    pvals = combined[pval].clip(lower=eps).astype(float)
    combined["_z"] = stats.norm.isf(pvals / 2.0)

    # Initialise equal weights
    combined["_w"] = 1.0

    # Detect whether sample-size columns exist
    has_binary_cols = ("N_case" in combined.columns) and ("N_control" in combined.columns)
    has_quant_col = "N" in combined.columns

    # Convert sample-size columns to numeric if present
    if has_binary_cols:
        combined["N_case"] = pd.to_numeric(combined["N_case"], errors="coerce")
        combined["N_control"] = pd.to_numeric(combined["N_control"], errors="coerce")

    if has_quant_col:
        combined["N"] = pd.to_numeric(combined["N"], errors="coerce")

    # Binary trait weighting takes priority when both columns are available
    if has_binary_cols:
        valid_binary = (
            combined["N_case"].notna()
            & combined["N_control"].notna()
            & (combined["N_case"] > 0)
            & (combined["N_control"] > 0)
        )

        # Effective sample size for case-control data
        combined.loc[valid_binary, "_N_eff"] = 4.0 / (
            (1.0 / combined.loc[valid_binary, "N_case"])
            + (1.0 / combined.loc[valid_binary, "N_control"])
        )
        combined.loc[valid_binary, "_w"] = np.sqrt(combined.loc[valid_binary, "_N_eff"])

    # For rows without valid binary sample sizes, try quantitative N
    if has_quant_col:
        valid_quant = (
            combined["N"].notna()
            & (combined["N"] > 0)
            & (combined["_w"] == 1.0)
        )
        combined.loc[valid_quant, "_w"] = np.sqrt(combined.loc[valid_quant, "N"])

    # Group by tested unit
    grouped = combined.groupby(variant_col, sort=False)

    total_n = combined["N_total"].unique().sum()
    total_case, total_control = None, None
    if has_binary_cols:
        total_case = combined["N_case"].unique().sum()
        total_control = combined["N_control"].unique().sum()
    
    def stouffer_func(df: pd.DataFrame) -> pd.Series:
        """
        Combine p-values for one variant / block across studies.
        """
        z_i = df["_z"].to_numpy(dtype=float)
        w_i = df["_w"].to_numpy(dtype=float)
        k = len(z_i)

        # Require at least two studies
        if k < 2:
            return pd.Series({
                "N studies": np.nan,
                f"{prefix}_Z": np.nan,
                f"{prefix}_P": np.nan,
                f"{prefix}_log10P": np.nan,
            })

        # Weighted Stouffer combination
        z_meta = np.sum(w_i * z_i) / np.sqrt(np.sum(w_i ** 2))
        p_meta = 2.0 * stats.norm.sf(abs(z_meta))
        log_p =  (stats.norm.logsf(abs(z_meta)) + np.log(2)) / np.log(10)

        return pd.Series({
            "N studies": k,
            f"{prefix}_Z": z_meta,
            f"{prefix}_P": p_meta,
            f"{prefix}_log10P": -log_p,
        })

    results = grouped.apply(stouffer_func).reset_index()
    results["Total N"] = total_n
    if has_binary_cols:
        results['Total Case'] = total_case
        results["Total Control"] = total_control
    results = results.dropna(subset=["N studies"])

    return results