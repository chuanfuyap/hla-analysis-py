"""
Standard runner for linear/logit association.

This runner:
1) prepares Y and covariates
2) iterates over variants (via adapter)
3) builds a modelling table (abt) and QC counts
4) computes frequency summaries on FINAL modelling rows
5) fits univariate or omnibus model
6) reports progress + timing

Outputs are CSV-friendly scalars + string summaries.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .logger import get_runner_logger

from ..parallel import parallel_imap_batched
from ..progress import ProgressPrinter, format_seconds
from ..preprocess import prepare_y, prepare_covar, make_model_table
from ..models.standard import fit_univariate, fit_omnibus


# -------------------------------------------------------------------
# Worker-global state
# -------------------------------------------------------------------

_STANDARD_STATE: dict | None = None

def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe
    if df.N_missing_Y.nunique()==1 and df.N_missing_Y.unique()[0]==0:
        df = df.drop(columns = ['N_missing_Y'])
    if df.N_dropped_other.nunique()==1 and df.N_dropped_other.unique()[0]==0:
        df = df.drop(columns = ['N_dropped_other'])
    if df.N_used.unique()[0]==df.N_total.unique()[0]:
        df = df.drop(columns = ['N_used'])

    df = df.dropna(axis=1, how="all")

    return df

def _init_standard_worker(state: dict) -> None:
    """
    Initialise per-worker shared state.

    In sequential mode we call it directly from run_standard().
    In parallel/process mode you must also run this once per worker process.
    """
    global _STANDARD_STATE
    _STANDARD_STATE = state


def _af_from_col(s: pd.Series) -> float:
    """Compute allele frequency estimate p = mean(dosage)/2."""
    return float(s.mean() / 2.0)


def _maf_from_col(s: pd.Series) -> float:
    """Compute MAF = min(p, 1-p) where p = mean(dosage)/2."""
    p = _af_from_col(s)
    return min(p, 1.0 - p)


def _run_standard_one_variant(variant_id: str) -> dict:
    """Module-scope per-variant worker, safe for process pools."""
    try:
        if _STANDARD_STATE is None:
            raise RuntimeError("Standard worker state has not been initialised.")

        adapter = _STANDARD_STATE["adapter"]
        hladat = _STANDARD_STATE["hladat"]
        yser = _STANDARD_STATE["yser"]
        covdf = _STANDARD_STATE["covdf"]
        covar_cols = _STANDARD_STATE["covar_cols"]
        model_type = _STANDARD_STATE["model_type"]
        variant_filter = _STANDARD_STATE["variant_filter"]
        condition_on = _STANDARD_STATE["condition_on"]

        geno_df, meta = adapter.build_geno(hladat, variant_id, sample_index=yser.index)
        geno_cols = list(geno_df.columns)

        covdf_local = covdf.copy() if covdf is not None else None
        covar_cols_local = list(covar_cols)

        if condition_on is not None:
            if condition_on == variant_id:
                return {
                    **meta,
                    "VARIANT": variant_id,
                    "COND_VARIANT": condition_on,
                    "COND_KIND": adapter.KIND,
                    "SKIPPED": True,
                    "Skip_Reason": "tested variant equals conditioning variant",
                }

            cond_df, _ = adapter.build_geno(hladat, condition_on, sample_index=yser.index)

            # Safety: prevent overlap between tested and conditioning columns
            cond_df = cond_df.copy()
            cond_df.columns = [f"COND_{c}" for c in cond_df.columns]

            if covdf_local is None:
                covdf_local = cond_df
            else:
                covdf_local = pd.concat([covdf_local, cond_df], axis=1)

            covar_cols_local.extend(list(cond_df.columns))

        if variant_filter is not None:
            ctx = {
                "analysis": "standard",
                "kind": adapter.KIND,
                "variant_id": variant_id,
                "meta": meta,
                "geno_cols": geno_cols,
            }
            if not variant_filter(ctx):
                return {"SKIPPED": True, **meta, "VARIANT": variant_id}

        abt, qc = make_model_table(geno_df, yser, covdf_local)

        row = dict(meta)
        row["VARIANT"] = variant_id
        row["COND_VARIANT"] = condition_on if condition_on is not None else np.nan
        row.update(qc)

        # Only set defaults that are potentially relevant; leave others absent.
        # We'll clean columns at the end (drop all-empty).
        if adapter.KIND == "SNP":
            row.setdefault("MAF", np.nan)
            row.setdefault("MAF_by_col_str", np.nan)
        elif adapter.KIND == "HLA":
            row.setdefault("HLA_AF", np.nan)
            row.setdefault("HLA_freqs_str", np.nan)
        elif adapter.KIND == "AA":
            row.setdefault("AA_AF_by_col_str", np.nan)

        # Always set model output keys so downstream code can rely on them
        row.setdefault("LR_p", np.nan)
        row.setdefault("Anova_p", np.nan)
        row.setdefault("multi_Coef", np.nan)
        row.setdefault("Uni_p", np.nan)
        row.setdefault("Uni_Coef", np.nan)
        row.setdefault("Uni_StdErr", np.nan)
        row.setdefault("CI_0.025", np.nan)
        row.setdefault("CI_0.975", np.nan)

        if qc["N_used"] == 0 or len(geno_cols) == 0:
            return row

        # frequencies on FINAL modelling rows
        if adapter.KIND == "SNP":
            maf_by = {c: _maf_from_col(abt[c]) for c in geno_cols}
            if len(geno_cols) == 1:
                row["MAF"] = maf_by[geno_cols[0]]
            else:
                row["MAF_by_col_str"] = ";".join([f"{k}={v:.6g}" for k, v in maf_by.items()])

        elif adapter.KIND == "HLA":
            if len(geno_cols) == 1:
                row["HLA_AF"] = _af_from_col(abt[geno_cols[0]])
            else:
                pieces = []
                for c in geno_cols:
                    pieces.append(f"{c}:af={_af_from_col(abt[c]):.6g}")
                row["HLA_freqs_str"] = ";".join(pieces) if pieces else np.nan

        # elif adapter.KIND == "AA":
        #     af_by = {c: _af_from_col(abt[c]) for c in geno_cols}
        #     row["AA_AF_by_col_str"] = ";".join([f"{k}={v:.6g}" for k, v in af_by.items()]) if af_by else np.nan

        # model fits (single-col -> univariate; multi-col -> omnibus)
        if len(geno_cols) == 1:
            row.update(fit_univariate(abt, geno_cols[0], covar_cols_local, model_type))
            # LR_p/Anova_p/multi_Coef remain NaN
        else:
            row.update(fit_omnibus(abt, geno_cols, covar_cols_local, model_type))
            # Uni_* remain NaN

        return row
    except Exception:
        logger = get_runner_logger()
        logger.exception(
            "standard runner FAILED on variant_id=%r (kind=%s)",
            variant_id,
            _STANDARD_STATE["adapter"].KIND if _STANDARD_STATE else "UNKNOWN",
        )
        raise

def run_standard(
    adapter,
    hladat,
    config,
    famfile: pd.DataFrame=None,
    covar=None,
    y=None,
    variant_filter=None,
    condition_on: str | None = None,
    *,
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 100,
) -> pd.DataFrame:
    """Run standard association testing for a single block kind."""
    t0 = time.perf_counter()

    yser = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, yser.index)
    covar_cols = list(covdf.columns) if covdf is not None else []

    variant_ids = list(adapter.iter_variants(hladat))
    if variant_filter is not None:
        block = getattr(hladat, adapter.KIND)
        info_df = block.info
        filtered = []
        for vid in variant_ids:
            if "AA_ID" in info_df.columns:
                row = info_df[info_df["AA_ID"] == vid]
            else:
                row = info_df[info_df.index == vid]
            meta = row.iloc[0].to_dict() if len(row) > 0 else {}
            ctx = {"analysis": "standard", "kind": adapter.KIND, "variant_id": vid, "meta": meta, "geno_cols": []}
            if variant_filter(ctx):
                filtered.append(vid)
        variant_ids = filtered

    if verbose:
        print("----------------", flush=True)
        print(f"STARTING ANALYSES on:\t{adapter.KIND}, model={config.model_type}", flush=True)
        print(f"VARIANTS SCHEDULED FOR ANALYSIS:\t{len(variant_ids)}", flush=True)
        print(
            f"parallel: n_jobs={config.n_jobs}, backend={config.backend}, "
            f"batch_size={getattr(config, 'batch_size', 32)}, chunksize={getattr(config, 'chunksize', None)}",
            flush=True,
        )
        print("----------------", flush=True)
        print()

    state = {
        "adapter": adapter,
        "hladat": hladat,
        "yser": yser,
        "covdf": covdf,
        "covar_cols": covar_cols,
        "model_type": config.model_type,
        "variant_filter": variant_filter,
        "condition_on": condition_on,
    }

    # sequential initialization (also used by thread backend)
    _init_standard_worker(state)

    prog = (
        ProgressPrinter(
            total=len(variant_ids),
            desc=f"standard[{adapter.KIND}]",
            use_tqdm=use_progress_bar,
            print_every=print_every,
        )
        if verbose
        else None
    )

    rows: list[dict] = []
    for r in parallel_imap_batched(
        _run_standard_one_variant,
        variant_ids,
        config.n_jobs,
        config.backend,
        batch_size=getattr(config, "batch_size", 32),
        max_in_flight=getattr(config, "chunksize", None),  # chunksize == max in-flight batches
    ):
        rows.append(r)
        if prog is not None:
            prog.update(1)

    if prog is not None:
        prog.close()

    out = pd.DataFrame(rows)

    # Coalesce omnibus/univariate p into one convenience column
    if adapter.KIND=="AA":
        if "LR_p" in out.columns and "Uni_p" in out.columns:
            out["LRp_Unip"] = out["LR_p"].combine_first(out["Uni_p"])

    # Clean: convert empty strings (if any) to NaN, then drop all-empty columns
    for c in ("MAF_by_col_str", "HLA_freqs_str", "AA_AF_by_col_str"):
        if c in out.columns:
            out[c] = out[c].replace("", np.nan)
    out = _clean_output(out)

    if verbose:
        print(f"standard[{adapter.KIND}] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

    return out