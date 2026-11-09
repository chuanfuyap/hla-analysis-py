"""
Survival runner: Cox PH regression per variant (batched).
P
Note
----
lifelines does not support statsmodels formula C() terms, so categorical columns are one-hot encoded
in preprocess.build_survival_table().
"""

from __future__ import annotations

import time
from functools import partial
import numpy as np
import pandas as pd


from ..parallel import parallel_imap_batched
from ..progress import ProgressPrinter, format_seconds
from ..preprocess import prepare_covar, build_survival_table
from ..models.survival import fit_cox_univariate, fit_cox_model, lrtest_cox


def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    if df.N_missing_Y.nunique()==1 and df.N_missing_Y.unique()[0]==0:
        df = df.drop(columns = ['N_missing_Y'])
    if df.N_dropped_other.nunique()==1 and df.N_dropped_other.unique()[0]==0:
        df = df.drop(columns = ['N_dropped_other'])
    if df.N_used.unique()[0]==df.N_total.unique()[0]:
        df = df.drop(columns = ['N_used'])

    df = df.dropna(axis=1, how="all")

    return df

def _af_from_col(s: pd.Series) -> float:
    """Allele frequency estimate from dosage/copy number."""
    return float(s.mean() / 2.0)


def _maf_from_col(s: pd.Series) -> float:
    """Minor allele frequency estimate from dosage/copy number."""
    p = _af_from_col(s)
    return min(p, 1.0 - p)


def _run_survival_one_variant(variant_id: str, *, state: dict) -> dict:
    adapter = state["adapter"]
    hladat = state["hladat"]
    iid_index = state["iid_index"]
    covdf = state["covdf"]
    event_time = state["event_time"]
    dynamic_covar = state["dynamic_covar"]
    variant_filter = state["variant_filter"]

    geno_df, meta = adapter.build_geno(hladat, variant_id, sample_index=iid_index)
    geno_cols = list(geno_df.columns)

    if variant_filter is not None:
        ctx = {"analysis": "survival", "kind": adapter.KIND, "variant_id": variant_id, "meta": meta, "geno_cols": geno_cols}
        if not variant_filter(ctx):
            return {"SKIPPED": True, **meta, "VARIANT": variant_id}

    cox_df, qc = build_survival_table(geno_df, event_time, cov=covdf, dynamic_covar=dynamic_covar)

    row = dict(meta)
    row["VARIANT"] = variant_id
    row.update(qc)

    if qc["N_used"] == 0 or len(geno_cols) == 0:
        row.update({"P-value": np.nan, "HR": np.nan,"StdErr":np.nan, "CI-lower": np.nan, "CI-upper": np.nan})
        return row

    # Count each person once, regardless of how many intervals they contribute.
    frequency_df = cox_df.drop_duplicates("sample_id")

    # Frequencies for single-column tests on final modelling samples.
    if len(geno_cols) == 1 and geno_cols[0] in cox_df.columns:
        if adapter.KIND == "SNP":
            row["MAF"] = _maf_from_col(frequency_df[geno_cols[0]])
        elif adapter.KIND == "HLA":
            row["HLA_AF"] = _af_from_col(frequency_df[geno_cols[0]])

    # Option A stats
    if len(geno_cols) == 1:
        row.update(fit_cox_univariate(cox_df, geno_cols[0]))  # fills p/HR/CI
        return row

    # Multi-column: omnibus LR only
    abt = cox_df.copy()

    alt = fit_cox_model(abt)
    alt_ll = float(alt.log_likelihood_)

    null_cols = [c for c in abt.columns if c not in set(geno_cols)]
    nul = fit_cox_model(abt[null_cols])
    null_ll = float(nul.log_likelihood_)

    row["P-value"] = lrtest_cox(alt_ll, null_ll, dof=len(geno_cols))
    row.update({"HR": np.nan,"StdErr":np.nan, "CI-lower": np.nan, "CI-upper": np.nan})
    return row


def run_survival(
    adapter,
    hladat,
    famfile: pd.DataFrame,
    config,
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
    t0 = time.perf_counter()

    iid_index = pd.Index(famfile["IID"].astype(str), name="IID")
    covdf = prepare_covar(covar, iid_index)
    variant_ids = list(adapter.iter_variants(hladat))
    if variant_filter is not None:
        block = getattr(hladat, kind)
        info_df = block.info
        filtered = []
        for vid in variant_ids:
            if "AA_ID" in info_df.columns:
                row = info_df[info_df["AA_ID"] == vid]
            else:
                row = info_df[info_df.index == vid]
            meta = row.iloc[0].to_dict() if len(row) > 0 else {}
            ctx = {"analysis": "survival", "kind": kind, "variant_id": vid, "meta": meta, "geno_cols": []}
            if variant_filter(ctx):
                filtered.append(vid)
        variant_ids = filtered

    if verbose:
        print("----------------", flush=True)
        print(f"STARTING SURVIVAL ANALYSES on:\t{kind}", flush=True)
        print(f"VARIANTS SCHEDULED FOR ANALYSIS:\t{len(variant_ids)}", flush=True)
        print(f"parallel: n_jobs={config.n_jobs}, backend={config.backend}, batch_size={config.batch_size}, chunksize={config.chunksize}", flush=True)
        print("----------------", flush=True)

    state = {
        "adapter": adapter,
        "hladat": hladat,
        "iid_index": iid_index,
        "covdf": covdf,
        "event_time": event_time,
        "dynamic_covar": dynamic_covar,
        "variant_filter": variant_filter,
    }


    prog = ProgressPrinter(total=len(variant_ids), desc=f"survival[{kind}]", use_tqdm=use_progress_bar, print_every=print_every) if verbose else None

    rows: list[dict] = []
    for r in parallel_imap_batched(
        partial(_run_survival_one_variant, state=state),
        variant_ids,
        config.n_jobs,
        config.backend,
        batch_size=config.batch_size,
        max_in_flight=config.chunksize,
    ):
        rows.append(r)
        if prog is not None:
            prog.update(1)

    if prog is not None:
        prog.close()

    out = pd.DataFrame(rows)

    if verbose:
        print(f"survival[{kind}] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

    return out