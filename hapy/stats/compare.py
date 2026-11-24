"""Compare two HLA amino acid positions using BIC and conditional tests."""

import numpy as np
import pandas as pd

from .adapters.aa import AAAdapter
from .models.standard import fit_position_comparison
from .preprocess import prepare_y, prepare_covar, make_model_table
from .types import StandardConfig


def compare_positions(
    hladat,
    config: StandardConfig,
    position_a: str,
    position_b: str,
    *,
    famfile=None,
    covar=None,
    y=None,
    min_delta_bic=2.0,
) -> pd.DataFrame:
    """Compare two amino acid positions using BIC and conditional tests.

    Parameters
    ----------
    hladat : HLAdata
        HLA data with AA genotypes loaded.
    config : StandardConfig
        Configuration with model_type="linear" or "logit".
    position_a : str
        Variant ID from hladat.AA.info["AA_ID"] for position A.
    position_b : str
        Variant ID from hladat.AA.info["AA_ID"] for position B.
    famfile : pandas.DataFrame, optional
        Sample information containing IID and PHENO when y is not supplied.
    covar : pandas.DataFrame, optional
        Covariates indexed by IID or containing an IID column.
    y : pandas.Series or array-like, optional
        Phenotype indexed by IID, or aligned to famfile row order.
        Without famfile, y must be a Series indexed by IID.
    min_delta_bic : float, optional
        Minimum BIC difference to prefer one model. Defaults to 2.

    Returns
    -------
    pandas.DataFrame
        One row with BIC for A, B and A+B, reciprocal conditional P-values
        and degrees of freedom, sample count, recommendation and INFO.

    Notes
    -----
    All models use the same covariates and complete-case samples.
    Lower BIC is preferred. Conditional tests use F-tests for linear models
    and likelihood-ratio tests for logistic models.
    If the joint model cannot be fitted, its results are NaN and INFO
    describes the issue. The individual models are still compared.
    """

    if not isinstance(config, StandardConfig) or config.model_type not in ("linear", "logit"):
        raise ValueError("config must be StandardConfig with model_type='linear' or 'logit'.")

    if not np.isfinite(min_delta_bic) or min_delta_bic < 0:
        raise ValueError("min_delta_bic must be finite and nonnegative.")

    if position_a == position_b:
        raise ValueError("Select two different amino acid positions.")

    if not hasattr(hladat, "AA"):
        raise ValueError("AA not loaded.")

    available = set(AAAdapter.iter_variants(hladat))
    for position in (position_a, position_b):
        if position not in available or str(position).startswith("MOTIF_"):
            raise ValueError(f"Invalid AA position: {position}")

    # Prepare phenotype and covariates
    phenotype = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, phenotype.index)
    covar_cols = list(covdf.columns) if covdf is not None else []

    # Build amino acid genotype matrices
    geno_a, _ = AAAdapter.build_geno(hladat, position_a, phenotype.index)
    geno_b, _ = AAAdapter.build_geno(hladat, position_b, phenotype.index)

    geno_a = geno_a[
        list(geno_a.attrs.get("test_columns", geno_a.columns))
    ].copy()

    geno_b = geno_b[
        list(geno_b.attrs.get("test_columns", geno_b.columns))
    ].copy()

    if geno_a.empty or geno_b.empty:
        raise ValueError("No eligible residue predictors.")

    # Rename to prevent overlapping residue column names
    geno_a.columns = [f"__A_{i}" for i in range(geno_a.shape[1])]
    geno_b.columns = [f"__B_{i}" for i in range(geno_b.shape[1])]

    a_cols = list(geno_a.columns)
    b_cols = list(geno_b.columns)

    if set(covar_cols).intersection(a_cols + b_cols + ["Y"]):
        raise ValueError("Covariates conflict with reserved columns.")

    # All models must use the same samples
    abt, _ = make_model_table(
        pd.concat([geno_a, geno_b], axis=1),
        phenotype,
        covdf,
    )

    result = fit_position_comparison(
        abt, a_cols, b_cols, covar_cols, config.model_type,
        min_delta_bic=min_delta_bic,
    )
    return pd.DataFrame([{
        "POSITION_A": position_a,
        "POSITION_B": position_b,
        **result,
    }])
