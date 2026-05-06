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

import re
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .logger import get_runner_logger
from ..parallel import parallel_imap_batched
from ..progress import ProgressPrinter, format_seconds
from ..preprocess import prepare_y, prepare_covar, make_model_table
from ..models.standard import fit_univariate, fit_omnibus


_STANDARD_STATE: dict | None = None


def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe
    if "N_missing_Y" in df.columns and df.N_missing_Y.nunique() == 1 and df.N_missing_Y.unique()[0] == 0:
        df = df.drop(columns=["N_missing_Y"])
    if "N_COND_VARIANTS" in df.columns and df.N_COND_VARIANTS.nunique() == 1 and df.N_COND_VARIANTS.unique()[0] == 0:
        df = df.drop(columns=["N_COND_VARIANTS"])
    if "N_dropped_other" in df.columns and df.N_dropped_other.nunique() == 1 and df.N_dropped_other.unique()[0] == 0:
        df = df.drop(columns=["N_dropped_other"])
    if "N_used" in df.columns and "N_total" in df.columns:
        if df.N_used.nunique() == 1 and df.N_total.nunique() == 1:
            if df.N_used.unique()[0] == df.N_total.unique()[0]:
                df = df.drop(columns=["N_used"])
    df = df.dropna(axis=1, how="all")
    return df


def _init_standard_worker(state: dict) -> None:
    global _STANDARD_STATE
    _STANDARD_STATE = state


def _af_from_col(s: pd.Series) -> float:
    return float(s.mean() / 2.0)


def _maf_from_col(s: pd.Series) -> float:
    p = _af_from_col(s)
    return min(p, 1.0 - p)


def _normalize_condition_on(condition_on) -> list[str]:
    if condition_on is None:
        return []

    if isinstance(condition_on, str):
        vals = [condition_on]
    elif isinstance(condition_on, Sequence):
        vals = list(condition_on)
    else:
        raise TypeError("condition_on must be None, a string, or a sequence of strings.")

    out: list[str] = []
    seen: set[str] = set()

    for v in vals:
        if not isinstance(v, str):
            raise TypeError("All condition_on values must be strings.")
        vv = v.strip()
        if not vv:
            raise ValueError("condition_on cannot contain empty strings.")
        if vv not in seen:
            out.append(vv)
            seen.add(vv)

    return out


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")


def _validate_conditioning_variants(adapter, hladat, condition_on_list: list[str]) -> None:
    if not condition_on_list:
        return

    valid_ids = {str(v) for v in adapter.iter_variants(hladat)}
    missing = [v for v in condition_on_list if v not in valid_ids]
    if missing:
        raise ValueError(
            f"The following conditioning variant(s) were not found in the {adapter.KIND} block: {missing}"
        )


def _run_standard_one_variant(variant_id: str) -> dict:
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
        condition_on_list = _STANDARD_STATE["condition_on_list"]

        # Main tested genotype
        geno_df, meta = adapter.build_geno(hladat, variant_id, sample_index=yser.index)
        geno_cols = list(geno_df.columns)

        # Local covariates for this tested variant
        covdf_local = covdf.copy() if covdf is not None else None
        covar_cols_local = list(covar_cols)

        used_conditions: list[str] = []
        skipped_conditions: list[str] = []

        # Add all requested conditioning variants as extra covariates
        for cond_variant in condition_on_list:
            if cond_variant == variant_id:
                skipped_conditions.append(cond_variant)
                continue

            cond_df, _ = adapter.build_geno(hladat, cond_variant, sample_index=yser.index)
            cond_df = cond_df.copy()

            safe_vid = _safe_name(cond_variant)
            cond_df.columns = [f"COND_{safe_vid}__{c}" for c in cond_df.columns]

            if covdf_local is None:
                covdf_local = cond_df
            else:
                covdf_local = pd.concat([covdf_local, cond_df], axis=1)

            covar_cols_local.extend(cond_df.columns.tolist())
            used_conditions.append(cond_variant)

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
        row["COND_KIND"] = adapter.KIND if used_conditions else np.nan
        row["COND_VARIANTS"] = ";".join(used_conditions) if used_conditions else np.nan
        row["N_COND_VARIANTS"] = len(used_conditions)
        row["SKIPPED_COND_VARIANTS"] = ";".join(skipped_conditions) if skipped_conditions else np.nan
        row.update(qc)

        if adapter.KIND == "SNP":
            row.setdefault("MAF", np.nan)
            row.setdefault("MAF_by_col_str", np.nan)
        elif adapter.KIND == "HLA":
            row.setdefault("HLA_AF", np.nan)
            row.setdefault("HLA_freqs_str", np.nan)
        elif adapter.KIND == "AA":
            row.setdefault("AA_AF_by_col_str", np.nan)

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
                pieces = [f"{c}:af={_af_from_col(abt[c]):.6g}" for c in geno_cols]
                row["HLA_freqs_str"] = ";".join(pieces) if pieces else np.nan

        if len(geno_cols) == 1:
            row.update(fit_univariate(abt, geno_cols[0], covar_cols_local, model_type))
        else:
            row.update(fit_omnibus(abt, geno_cols, covar_cols_local, model_type))

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
    famfile: pd.DataFrame = None,
    covar=None,
    y=None,
    variant_filter=None,
    condition_on=None,
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

    condition_on_list = _normalize_condition_on(condition_on)
    _validate_conditioning_variants(adapter, hladat, condition_on_list)

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
            ctx = {
                "analysis": "standard",
                "kind": adapter.KIND,
                "variant_id": vid,
                "meta": meta,
                "geno_cols": [],
            }
            if variant_filter(ctx):
                filtered.append(vid)
        variant_ids = filtered

    if verbose:
        print("----------------", flush=True)
        print(f"STARTING ANALYSES on:\t{adapter.KIND}, model={config.model_type}", flush=True)
        print(f"VARIANTS SCHEDULED FOR ANALYSIS:\t{len(variant_ids)}", flush=True)
        if condition_on_list:
            print(f"CONDITIONING ON:\t{condition_on_list}", flush=True)
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
        "condition_on_list": condition_on_list,
    }

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
        max_in_flight=getattr(config, "chunksize", None),
    ):
        rows.append(r)
        if prog is not None:
            prog.update(1)

    if prog is not None:
        prog.close()

    out = pd.DataFrame(rows)

    if adapter.KIND == "AA":
        if "LR_p" in out.columns and "Uni_p" in out.columns:
            out["LRp_Unip"] = out["LR_p"].combine_first(out["Uni_p"])

    for c in ("MAF_by_col_str", "HLA_freqs_str", "AA_AF_by_col_str", "COND_VARIANTS", "SKIPPED_COND_VARIANTS"):
        if c in out.columns:
            out[c] = out[c].replace("", np.nan)

    out = _clean_output(out)

    if verbose:
        print(f"standard[{adapter.KIND}] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

    return out