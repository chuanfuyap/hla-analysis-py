"""
Public API for hapy.stats.

This module provides stable entrypoints for:
- standard association tests (linear/logit): analyse(...)
- interaction tests (linear/logit): interaction(...)
- survival tests (Cox PH): survival(...)

All functions accept pandas DataFrames and return pandas DataFrames suitable for CSV export.
"""

from __future__ import annotations
import pandas as pd

from .types import StandardConfig, InteractionConfig, SurvivalConfig
from .runners.standard_runner import run_standard
from .runners.interaction_runner import run_interaction
from .runners.survival_runner import run_survival

from .adapters.aa import AAAdapter
from .adapters.hla import HLAAdapter
from .adapters.snp import SNPAdapter

_KIND_MAP = {"AA": AAAdapter, "HLA": HLAAdapter, "SNP": SNPAdapter}


def analyse(
    hladat,
    config: StandardConfig,
    *,
    kind: str,
    famfile: pd.DataFrame=None,
    covar=None,
    y=None,
    variant_filter=None,
    condition_on: str | None = None,
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 100,
) -> pd.DataFrame:
    """
    Run standard (single-block) association testing for AA/HLA/SNP.

    Parameters
    ----------
    hladat:
        Your HLAdat object containing genotype blocks (AA/HLA/SNP) and corresponding info.
    famfile:
        PLINK Fam file imported with read_famfile:
        - IID: sample identifier
        - PHENO: phenotype (if y is not provided). Binary PLINK-style {1,2} is auto-recoded to {0,1}.
    config:
        StandardConfig specifying model type ("linear" or "logit") and parallel settings.
    kind:
        Which genotype block to analyse. Must be one of: "AA", "HLA", "SNP".
    covar:
        Optional covariate table:
        - pandas DataFrame indexed by IID, or
        - str path to a CSV with IID as its index (index_col=0).
        Covariates may be numeric and/or categorical strings.
    y:
        Optional phenotype override:
        - None: use famfile.PHENO
        - pandas Series indexed by IID
        - array-like aligned to famfile row order
    variant_filter:
        Optional callable(ctx)->bool that can drop variants before fitting.
        Context schema is documented in hapy.stats.filters.
    condition_on:
        Optional variant input for conditional analysis
    verbose:
        If True, prints progress messages with flush=True.
    use_progress_bar:
        If True, uses tqdm progress bar if tqdm is installed; otherwise falls back to periodic prints.
    print_every:
        If tqdm is not used, print a progress update every N completed variants.

    Returns
    -------
    pandas.DataFrame
        One row per tested variant (position/allele depending on kind), including:
        - variant meta fields (GENE/POS/etc when available)
        - QC fields: N_total, N_missing_Y, N_dropped_other, N_used, N_case, N_control
        - frequency fields: MAF / HLA_AF+counts / AA_AF_by_col_str
        - model fields: Uni_p/Uni_Coef/CI... and/or LR_p/Anova_p for omnibus tests

    Notes
    -----
    Frequencies (MAF/AF/counts) are computed on the FINAL modelling sample set (after dropping missingness).
    """
    if kind not in _KIND_MAP:
        raise ValueError(f"Unknown kind={kind}. Must be one of {sorted(_KIND_MAP.keys())}")
    
    if not hasattr(hladat, kind):
        raise ValueError(f"{kind} not loaded. Please read the file type into HLAdat object.")

    adapter = _KIND_MAP[kind]
    return run_standard(
        adapter,
        hladat,
        config,
        famfile,
        covar=covar,
        y=y,
        variant_filter=variant_filter,
        condition_on=condition_on,
        verbose=verbose,
        use_progress_bar=use_progress_bar,
        print_every=print_every,
    )

def interaction(
    hladat,
    config: InteractionConfig,
    *,
    a_kind: str,
    b_kind: str,
    famfile: pd.DataFrame=None,
    covar=None,
    y=None,
    pair_filter=None,
    cov_block_cols: list[str] | None = None,
    baseline_covar_cols: list[str] | None = None,
    block_b_df: pd.DataFrame | None = None,  # <-- NEW (for b_kind="DF")
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 500,
) -> pd.DataFrame:
    """
    Run interaction testing between two blocks.

    Supported blocks
    ----------------
    - a_kind: "AA" | "HLA" | "SNP"
    - b_kind: "AA" | "HLA" | "SNP" | "COV" | "DF"

    Where:
    - b_kind="COV" uses covariate columns as block B features (cov_block_cols required)
    - b_kind="DF"  uses an external IID-indexed dataframe as block B features (block_b_df required)

    Interaction modes
    -----------------
    - mode="pairwise":
        Tests each feature column from A against each feature column from B:
          Base: Y ~ baseline_covars + A + B
          Alt:  Base + A:B
        Returns signed interaction coefficient I_coef.

    - mode="one_vs_block_omnibus":
        Tests an omnibus interaction between an anchor feature and an AA multi-column block:
          Null: Y ~ baseline_covars + anchor + AA_block
          Alt:  Null + sum(anchor:AA_block_j)
        IMPORTANT RULE: This mode is only supported when AA is one of the blocks
        (regardless of whether AA is on A or B). The anchor side may be genotypes, COV, or DF.

    Parameters
    ----------
    hladat, famfile, covar, y:
        Same meaning as analyse(...).
    config:
        InteractionConfig with model_type and mode, plus parallel settings.
    a_kind, b_kind:
        Block types.
    pair_filter:
        Optional callable(ctx)->bool for dropping specific A/B pairs or columns.
        Context schema is documented in hapy.stats.filters.
    cov_block_cols:
        Required if b_kind="COV". These covariate columns are treated as block B features.
    baseline_covar_cols:
        Optional list of covariate columns to always include as baseline adjustment.
        If omitted:
          - when b_kind not in {"COV"}: baseline = all covar columns
          - when b_kind == "COV": baseline = all covar columns EXCEPT cov_block_cols
        (For b_kind="DF", baseline defaults to all covar columns.)
    block_b_df:
        Required if b_kind="DF".
        IID-indexed dataframe (index must match famfile IID / modelling IID set).
        Columns are treated as block B features (each column interacts with each A column in pairwise mode).
    verbose, use_progress_bar, print_every:
        Progress reporting controls.

    Returns
    -------
    pandas.DataFrame
        Pairwise mode: one row per (A_col, B_col) test.
        Omnibus mode: one row per (anchor_col, AA_block_variant) test.
    """
    if not hasattr(hladat, a_kind):
        raise ValueError(f"{a_kind} not loaded. Please read the file type into HLAdat object.")
    
    if a_kind not in _KIND_MAP:
        raise ValueError(f"Unknown a_kind={a_kind}. Must be one of {sorted(_KIND_MAP.keys())}")

    allowed_b = sorted(list(_KIND_MAP.keys()) + ["COV", "DF"])
    if b_kind not in allowed_b:
        raise ValueError(f"Unknown b_kind={b_kind}. Must be one of {allowed_b}")

    if b_kind == "COV":
        if cov_block_cols is None:
            raise ValueError("b_kind='COV' requires cov_block_cols.")
        if block_b_df is not None:
            raise ValueError("block_b_df is only used when b_kind='DF'.")

    if b_kind == "DF":
        if block_b_df is None:
            raise ValueError("b_kind='DF' requires block_b_df.")
        if cov_block_cols is not None:
            raise ValueError("cov_block_cols is only used when b_kind='COV'.")

    adapter_a = _KIND_MAP[a_kind]
    adapter_b = _KIND_MAP[b_kind] if b_kind in _KIND_MAP else None  # None for COV/DF

    return run_interaction(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        hladat=hladat,
        famfile=famfile,
        config=config,
        a_kind=a_kind,
        b_kind=b_kind,
        covar=covar,
        y=y,
        pair_filter=pair_filter,
        cov_block_cols=cov_block_cols,
        baseline_covar_cols=baseline_covar_cols,
        block_b_df=block_b_df,  # <-- pass through
        verbose=verbose,
        use_progress_bar=use_progress_bar,
        print_every=print_every,
    )


def survival(
    hladat,
    famfile: pd.DataFrame,
    config: SurvivalConfig,
    *,
    kind: str,
    event_time: pd.DataFrame,
    covar=None,
    dynamic_covar: pd.DataFrame | None = None,
    variant_filter=None,
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 100,
) -> pd.DataFrame:
    """
    Run Cox proportional hazards survival analysis for AA/HLA/SNP.

    Parameters
    ----------
    hladat:
        HLAdat genotype object.
    famfile:
        Sample table containing at least IID. (PHENO is not used by survival.)
    config:
        SurvivalConfig specifying parallel settings.
    kind:
        "AA" | "HLA" | "SNP"
    event_time:
        DataFrame containing survival endpoints with columns:
        - sample_id: matches famfile.IID
        - time: duration
        - event: 1 if event occurred, 0 if censored
        This table can contain multiple rows per sample (long format).
    covar:
        Optional static covariates indexed by IID (or CSV path).
        Categorical covariates are one-hot encoded automatically (drop_first=True).
    dynamic_covar:
        Optional time-varying covariates DataFrame with columns including:
        - sample_id
        - time
        plus covariate columns.
        These are merged into the survival table on ["sample_id","time"].
    variant_filter:
        Optional callable(ctx)->bool to skip variants.
    verbose, use_progress_bar, print_every:
        Progress reporting controls.

    Returns
    -------
    pandas.DataFrame
        One row per variant with:
        - QC: N_total, N_dropped_na, N_used, N_event, N_censored
        - frequency summaries computed on final modelling rows
        - survival stats: p, HR, CI_0.025, CI_0.975
        - omnibus LR_p for multi-column blocks (e.g., AA positions) where applicable

    Notes
    -----
    lifelines does not support statsmodels formula encoding; categorical covariates are one-hot encoded
    before fitting.
    """
    if kind not in _KIND_MAP:
        raise ValueError(f"Unknown kind={kind}. Must be one of {sorted(_KIND_MAP.keys())}")
    
    if not hasattr(hladat, kind):
        raise ValueError(f"{kind} not loaded. Please read the file type into HLAdat object.")

    adapter = _KIND_MAP[kind]
    return run_survival(
        adapter,
        hladat,
        famfile,
        config,
        kind=kind,
        event_time=event_time,
        covar=covar,
        dynamic_covar=dynamic_covar,
        variant_filter=variant_filter,
        verbose=verbose,
        use_progress_bar=use_progress_bar,
        print_every=print_every,
    )