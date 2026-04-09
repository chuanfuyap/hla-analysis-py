"""
Preprocessing utilities for analyses.

Standard / interaction analyses:
- prepare_y: obtain phenotype series Y aligned to famfile IIDs
- prepare_covar: align covariates to IIDs
- make_model_table: join geno + covar + Y; drop missingness; return QC counts

Survival analyses:
- build_survival_table: merge event_time + geno + covar + optional dynamic covar;
  drop missingness; one-hot encode categoricals for lifelines.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def fam_index(famfile: pd.DataFrame) -> pd.Index:
    """
    Extract IID index from famfile.

    Parameters
    ----------
    famfile:
        Must contain column 'IID'.

    Returns
    -------
    pd.Index
        IID values as strings.
    """
    return pd.Index(famfile["IID"].astype(str), name="IID")


def _maybe_recode_binary_12_to_01(ys: pd.Series) -> pd.Series:
    """
    Recode PLINK-style binary phenotype {1,2} to {0,1} if detected.

    Parameters
    ----------
    ys:
        Phenotype series.

    Returns
    -------
    pd.Series
        Recoded series if values are exactly {1,2}; otherwise unchanged.
    """
    vals = set(pd.unique(ys.dropna()))
    if vals == {1, 2}:
        return ys - 1
    return ys


def prepare_y(famfile: pd.DataFrame, y=None) -> pd.Series:
    """
    Prepare phenotype series Y aligned to famfile IIDs.

    Parameters
    ----------
    famfile:
        DataFrame containing 'IID' and 'PHENO' (if y is None).
    y:
        - None: uses famfile['PHENO']
        - pd.Series: reindexed to famfile IIDs
        - array-like: assumed aligned to famfile row order

    Returns
    -------
    pd.Series
        Named 'Y' indexed by IID.

    Notes
    -----
    Missing values are allowed; missing Y samples will be dropped during modelling.
    """
    ix = fam_index(famfile)

    if y is None:
        fam = famfile[["IID", "PHENO"]].copy()
        fam["IID"] = fam["IID"].astype(str)
        fam = fam.set_index("IID")
        ys = fam["PHENO"].reindex(ix)
        ys = _maybe_recode_binary_12_to_01(ys)
        ys.name = "Y"
        return ys

    if isinstance(y, pd.Series):
        ys = y.copy()
        ys.index = ys.index.astype(str)
        ys = ys.reindex(ix)
        ys = _maybe_recode_binary_12_to_01(ys)
        ys.name = "Y"
        return ys

    arr = np.asarray(y)
    if arr.shape[0] != len(ix):
        raise ValueError(f"Provided y has length {arr.shape[0]} but famfile has {len(ix)} IIDs.")
    ys = pd.Series(arr, index=ix, name="Y")
    ys = _maybe_recode_binary_12_to_01(ys)
    return ys


def prepare_covar(covar, index: pd.Index) -> pd.DataFrame | None:
    """
    Load/align covariates to a target sample index.

    Parameters
    ----------
    covar:
        - None
        - pd.DataFrame indexed by IID
        - str path to CSV (index_col=0)
    index:
        Desired index (IIDs) to align to.

    Returns
    -------
    DataFrame or None
        Covariate frame aligned to `index`.

    Notes
    -----
    Categorical covariates (object/category) are not encoded here; encoding is handled later:
    - standard/interaction: via statsmodels formulas using C(col)
    - survival: via pd.get_dummies in build_survival_table
    """
    if covar is None:
        return None

    if isinstance(covar, str):
        cov = pd.read_csv(covar, index_col=0)
    else:
        cov = covar.copy()

    cov.index = cov.index.astype(str)
    return cov.reindex(index)


def subset_samples_beagle_orientation(data: pd.DataFrame, sample_ids: pd.Index, call_type: str) -> pd.DataFrame:
    """
    Subset a Beagle-oriented genotype DataFrame to the analysis sample IDs.

    Parameters
    ----------
    data:
        Beagle-like genotype frame with variants as rows and sample IDs as columns.
    sample_ids:
        IID list/index (strings).
    call_type:
        - "softcall": expected columns are IIDs
        - "hardcall": expected columns include IID and IID.1

    Returns
    -------
    pd.DataFrame
        Subset DataFrame containing only the required columns.
    """
    df = data.copy()

    if call_type == "softcall":
        cols = list(map(str, sample_ids))
    elif call_type == "hardcall":
        cols = []
        for x in sample_ids:
            cols.append(str(x))
            cols.append(f"{x}.1")
    else:
        raise ValueError(f"Unknown call_type={call_type}")

    return df[cols]


def _case_control_counts(abt: pd.DataFrame) -> dict:
    """
    Compute case/control counts if Y is binary on the FINAL modelling rows.

    Returns
    -------
    dict with N_case, N_control
    """
    yy = abt["Y"]
    vals = set(pd.unique(yy))
    if vals.issubset({0, 1}) and len(vals) > 0:
        return {"N_case": int((yy == 1).sum()), "N_control": int((yy == 0).sum())}
    return {"N_case": np.nan, "N_control": np.nan}


def make_model_table(geno_df: pd.DataFrame, y: pd.Series, cov: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Construct the modelling table for standard/interaction analyses.

    Parameters
    ----------
    geno_df:
        Genotype/features DataFrame indexed by IID.
    y:
        Phenotype series indexed by IID (name 'Y').
    cov:
        Optional covariate DataFrame indexed by IID.

    Returns
    -------
    abt:
        Final modelling DataFrame with columns geno + cov + Y.
    qc:
        Dict with:
          - N_total: total rows before dropping
          - N_missing_Y: count of missing Y before dropping
          - N_dropped_other: count of rows dropped due to any NA after dropping missing Y
          - N_used: final rows used
          - N_case/N_control: if binary on final rows
    """
    parts = [geno_df, y.to_frame()]
    if cov is not None:
        parts.insert(1, cov)

    abt = pd.concat(parts, axis=1)

    n_total = int(abt.shape[0])
    n_missing_y = int(abt["Y"].isna().sum())

    abt = abt.dropna(subset=["Y"])
    n_dropped_other = int(abt.isna().any(axis=1).sum())
    abt = abt.dropna(axis=0)

    qc = {
        "N_total": n_total,
        "N_missing_Y": n_missing_y,
        "N_dropped_other": n_dropped_other,
        "N_used": int(abt.shape[0]),
        **_case_control_counts(abt),
    }
    return abt, qc


def build_survival_table(
    geno_df: pd.DataFrame,
    event_time: pd.DataFrame,
    cov: pd.DataFrame | None = None,
    dynamic_covar: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the modelling table for Cox PH survival analysis.

    Parameters
    ----------
    geno_df:
        Feature matrix indexed by IID.
    event_time:
        Must contain columns: sample_id, time, event.
        May be long format (multiple rows per sample).
    cov:
        Optional static covariates indexed by IID.
    dynamic_covar:
        Optional time-varying covariates with columns including sample_id and time.

    Returns
    -------
    cox_df:
        DataFrame containing:
          - sample_id, time, event
          - one-hot encoded covariates/features (drop_first=True)
    qc:
        Dict with:
          - N_total (rows after merges, before dropping NA)
          - N_dropped_na
          - N_used
          - N_event
          - N_censored
    """
    et = event_time.copy()
    for col in ("sample_id", "time", "event"):
        if col not in et.columns:
            raise ValueError(f"event_time must contain column '{col}'")
    et["sample_id"] = et["sample_id"].astype(str)

    base = geno_df.copy()
    base.index = base.index.astype(str)

    if cov is not None:
        cov2 = cov.copy()
        cov2.index = cov2.index.astype(str)
        base = pd.concat([base, cov2], axis=1)

    base = base.reset_index().rename(columns={"index": "sample_id"})
    merged = pd.merge(et, base, on="sample_id", how="inner")

    if dynamic_covar is not None:
        dyn = dynamic_covar.copy()
        if "sample_id" not in dyn.columns or "time" not in dyn.columns:
            raise ValueError("dynamic_covar must contain columns ['sample_id','time', ...]")
        dyn["sample_id"] = dyn["sample_id"].astype(str)
        merged = pd.merge(merged, dyn, on=["sample_id", "time"], how="left")

    n_total = int(merged.shape[0])
    n_dropped_na = int(merged.isna().any(axis=1).sum())
    merged = merged.dropna(axis=0)

    keep = merged[["sample_id", "time", "event"]]
    X = merged.drop(columns=["sample_id", "time", "event"])

    # lifelines expects numeric matrix; one-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    cox_df = pd.concat([keep, X], axis=1)

    n_used = int(cox_df.shape[0])
    n_event = int(cox_df["event"].astype(int).sum())
    n_censored = n_used - n_event

    qc = {
        "N_total": n_total,
        "N_dropped_na": n_dropped_na,
        "N_used": n_used,
        "N_event": n_event,
        "N_censored": n_censored,
    }
    return cox_df, qc