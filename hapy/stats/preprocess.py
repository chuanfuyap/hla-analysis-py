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
from os import PathLike

import numpy as np
import pandas as pd


def sample_index(values, *, source: str = "Samples") -> pd.Index:
    """Canonical IID labels; ambiguous or missing sample identifiers are errors."""
    ix = pd.Index(values)
    if ix.isna().any():
        raise ValueError(f"{source} contains missing sample IDs.")
    ix = ix.astype(str)
    if (ix.str.strip() == "").any():
        raise ValueError(f"{source} contains empty sample IDs.")
    if ix.has_duplicates:
        raise ValueError(f"{source} contains duplicate sample IDs after conversion to strings.")
    return ix.rename("IID")


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
    return sample_index(famfile["IID"], source="famfile")


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
    if famfile is not None:
        ix = fam_index(famfile)
    elif isinstance(y, pd.Series):
        ix = sample_index(y.index, source="Phenotype")
    else:
        raise ValueError("Without famfile, y must be a pandas Series indexed by IID.")

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
        ys.index = sample_index(ys.index, source="Phenotype")
        ys = ys.reindex(ix)
        ys = _maybe_recode_binary_12_to_01(ys)
        ys.name = "Y"
        return ys

    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("Provided y must be one-dimensional.")
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

    index = sample_index(index, source="Target samples")
    if isinstance(covar, (str, PathLike)):
        # Read the ID column as text before pandas can discard leading zeros.
        cov = pd.read_csv(covar, index_col=0, converters={0: str})
    else:
        cov = covar.copy()

    cov.index = sample_index(cov.index, source="Covariates")
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

    tmp_ids = df.columns.intersection(cols)

    return df[tmp_ids]


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
    geno_df = geno_df.copy()
    geno_df.index = sample_index(geno_df.index, source="Genotypes")
    y = y.copy()
    y.index = sample_index(y.index, source="Phenotype")
    y.name = "Y"
    parts = [geno_df, y.to_frame()]
    if cov is not None:
        cov = cov.copy()
        cov.index = sample_index(cov.index, source="Covariates")
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
    """Build static Cox rows, or start/stop intervals for dynamic covariates.

    event_time has one endpoint per sample: sample_id, time (>0), event (0/1).
    Dynamic rows describe state changes at nonnegative time, including baseline
    values at zero. Missing update cells mean no change. Changes at/after the
    endpoint are ignored. Incomplete subjects are excluded in their entirety;
    QC sample counts count people, not intervals.
    """
    from lifelines.utils import to_long_format, add_covariate_to_timeline

    et = event_time.copy()
    required = ["sample_id", "time", "event"]
    if not set(required).issubset(et.columns):
        raise ValueError("event_time must contain sample_id, time, event.")
    if set(et.columns) != set(required):
        raise ValueError("Put baseline predictors in covar, not event_time.")
    et["sample_id"] = sample_index(et["sample_id"], source="Survival endpoints").to_numpy()
    et["time"] = pd.to_numeric(et["time"], errors="raise")
    if not np.isfinite(et["time"]).all() or (et["time"] <= 0).any():
        raise ValueError("Survival endpoint times must be finite and positive.")
    if not et["event"].isin([0, 1]).all():
        raise ValueError("Survival event must be 0 or 1.")

    base = geno_df.copy()
    base.index = sample_index(base.index, source="Genotypes")
    if cov is not None:
        cov = cov.copy()
        cov.index = sample_index(cov.index, source="Covariates")
        base = pd.concat([base, cov], axis=1)
    reserved = {"sample_id", "time", "event", "start", "stop"}
    if not base.columns.is_unique or reserved.intersection(base.columns):
        raise ValueError("Survival predictor names must be unique and cannot use reserved time/ID columns.")
    base.index.name = "sample_id"
    merged = et.merge(base.reset_index(), on="sample_id", how="inner", validate="one_to_one")
    n_total = len(merged)
    merged = merged.dropna().copy()

    dynamic_cols = []
    if dynamic_covar is not None:
        dyn = dynamic_covar.copy()
        if not {"sample_id", "time"}.issubset(dyn.columns):
            raise ValueError("dynamic_covar must contain sample_id and time.")
        dynamic_cols = [c for c in dyn.columns if c not in {"sample_id", "time"}]
        if not dynamic_cols or not dyn.columns.is_unique:
            raise ValueError("dynamic_covar must have uniquely named covariate columns.")
        if (set(dynamic_cols) & (reserved | set(base.columns))):
            raise ValueError("Dynamic covariate names must not overlap static predictors or reserved columns.")
        # Repeated IDs are expected here, but missing/blank labels are invalid.
        sample_index(pd.unique(dyn["sample_id"]), source="Dynamic covariates")
        dyn["sample_id"] = dyn["sample_id"].astype(str)
        dyn["time"] = pd.to_numeric(dyn["time"], errors="raise")
        if not np.isfinite(dyn["time"]).all() or (dyn["time"] < 0).any():
            raise ValueError("Dynamic change times must be finite and nonnegative.")
        if dyn.duplicated(["sample_id", "time"]).any():
            raise ValueError("Duplicate dynamic sample_id/time updates are ambiguous.")
        baseline = dyn.loc[dyn["time"] == 0].set_index("sample_id")
        complete = baseline[dynamic_cols].notna().all(axis=1)
        merged = merged[merged["sample_id"].isin(baseline.index[complete])].copy()
        endpoints = merged.set_index("sample_id")["time"]
        dyn = dyn[dyn["sample_id"].isin(endpoints.index)].copy()
        dyn = dyn[dyn["time"] < dyn["sample_id"].map(endpoints)]
        intervals = to_long_format(merged, duration_col="time")
        if not merged.empty:
            for col in dynamic_cols:
                updates = dyn[["sample_id", "time", col]].dropna(subset=[col])
                intervals = add_covariate_to_timeline(
                    intervals, updates, id_col="sample_id", duration_col="time", event_col="event")
        else:
            for col in dynamic_cols:
                intervals[col] = pd.Series(dtype=dyn[col].dtype)
        table = intervals
        keep_cols = ["sample_id", "start", "stop", "event"]
    else:
        table = merged
        keep_cols = ["sample_id", "time", "event"]

    features = pd.get_dummies(table.drop(columns=keep_cols), drop_first=True)
    cox_df = pd.concat([table[keep_cols], features], axis=1)
    n_used = len(merged)
    n_event = int(merged["event"].sum())
    qc = {"N_total": int(n_total), "N_dropped_na": int(n_total - n_used),
          "N_used": int(n_used), "N_event": n_event, "N_censored": int(n_used - n_event)}
    if dynamic_covar is not None:
        qc["N_intervals"] = int(len(cox_df))
    return cox_df, qc
