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
from ..models.survival import fit_cox_univariate, fit_cox_omnibus, _empty_cox_result
from ..models import survival as survival_models
from ..adapters._samples import null_sample_ids


def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Preserve the established Cox output fields and keep INFO last.
    df = dataframe.drop(columns=["N_total"], errors="ignore")
    if not df.empty and "INFO" not in df:
        df["INFO"] = np.nan
    if "INFO" in df:
        df = df[[c for c in df.columns if c != "INFO"] + ["INFO"]]
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
            return {"SKIPPED": True, **meta, "VARIANT": variant_id,
                    **_empty_cox_result("excluded by variant filter")}

    cox_df, qc = build_survival_table(geno_df, event_time, cov=covdf, dynamic_covar=dynamic_covar)

    row = dict(meta)
    row["VARIANT"] = variant_id
    row.update(qc)

    if len(geno_cols) == 0:
        row.update(_empty_cox_result("insufficient variation"))
        return row
    if qc["N_used"] == 0:
        row.update(_empty_cox_result("no complete-case samples"))
        return row
    if qc["N_event"] == 0:
        row.update(_empty_cox_result("no observed events"))
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

    # Multi-column: check both Cox models before the omnibus LR test.
    row.update(fit_cox_omnibus(cox_df, geno_cols, null_cache=state.get("null_cache")))
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

    genetic_ids = null_sample_ids(adapter, hladat, iid_index, variant_ids)
    null_cache = None
    if genetic_ids is not None and len(genetic_ids):
        baseline, baseline_qc = build_survival_table(
            pd.DataFrame(index=genetic_ids), event_time,
            cov=covdf.reindex(genetic_ids) if covdf is not None else None,
            dynamic_covar=dynamic_covar,
        )
        if baseline_qc["N_used"] and baseline_qc["N_event"]:
            null_cache = survival_models.fit_cox_null(baseline)

    state = {
        "adapter": adapter,
        "hladat": hladat,
        "iid_index": iid_index,
        "covdf": covdf,
        "event_time": event_time,
        "dynamic_covar": dynamic_covar,
        "variant_filter": variant_filter,
        "null_cache": null_cache,
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

    out = _clean_output(pd.DataFrame(rows))

    if verbose:
        print(f"survival[{kind}] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

    return out