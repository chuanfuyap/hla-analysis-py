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
from functools import partial
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .logger import get_runner_logger
from ..parallel import parallel_imap_batched
from ..progress import ProgressPrinter, format_seconds
from ..preprocess import prepare_y, prepare_covar, make_model_table, build_model_predictors, attach_model_phenotype, sample_index
from ..models.standard import fit_univariate, fit_omnibus




def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe
    if "N_missing_Y" in df.columns and df.N_missing_Y.nunique() == 1 and df.N_missing_Y.unique()[0] == 0:
        df = df.drop(columns=["N_missing_Y"])
    if "N_COND_VARIANTS" in df.columns and df.N_COND_VARIANTS.nunique() == 1 and df.N_COND_VARIANTS.unique()[0] == 0:
        df = df.drop(columns=["N_COND_VARIANTS"])
    if "N_dropped_other" in df.columns and df.N_dropped_other.nunique() == 1 and df.N_dropped_other.unique()[0] == 0:
        df = df.drop(columns=["N_dropped_other"])
    df = df.dropna(axis=1, how="all")
    return df


def _ensure_aa_output_info(out):
    """Keep INFO and missing P-values available even when every test skips."""
    if 'INFO' not in out:
        out['INFO'] = ''
    if out.INFO.eq('insufficient variation').any():
        if "P-value" not in out:
            out[ "P-value"] = np.nan
    return out


def _report_aa_skips(tables):
    variants = set()
    for table in tables:
        if 'INFO' in table:
            variants.update(table.loc[table.INFO.eq('insufficient variation'), 'VARIANT'])
    if variants:
        plural = 'position' if len(variants) == 1 else 'positions'
        print(f'{len(variants)} AA {plural} skipped because insufficient variation remained '
              'after subsetting to analysis samples. See INFO.', flush=True)


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


def _fit_prepared_row(abt, qc, meta, variant_id, adapter, geno_cols,
                      covar_cols_local, model_type, used_conditions, skipped_conditions):
    """Fit one phenotype."""
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

    #row.setdefault("LR_p", np.nan)
    #row.setdefault("Anova_p", np.nan)
    row.setdefault("P-value", np.nan)
    row.setdefault("TEST", np.nan)
    #row.setdefault("multi_Coef", np.nan)
    #row.setdefault("Uni_p", np.nan)
    row.setdefault("Beta", np.nan)
    row.setdefault("StdErr", np.nan)
    row.setdefault("CI-lower", np.nan)
    row.setdefault("CI-upper", np.nan)

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


def _run_standard_one_variant(variant_id: str, *, state: dict) -> dict:
    try:
        adapter = state["adapter"]
        hladat = state["hladat"]
        yser = state["yser"]
        covdf = state["covdf"]
        covar_cols = state["covar_cols"]
        model_type = state["model_type"]
        variant_filter = state["variant_filter"]
        condition_on_list = state["condition_on_list"]

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
            if cond_df.shape[1] == 0:
                raise ValueError(f"Conditioning variant {cond_variant!r} has no eligible genotype columns.")
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
                return {**meta, "VARIANT": variant_id, "INFO": "excluded by variant filter"}

        abt, qc = make_model_table(geno_df, yser, covdf_local)

        return _fit_prepared_row(abt, qc, meta, variant_id, adapter, geno_cols,
                                 covar_cols_local, model_type, used_conditions, skipped_conditions)

    except Exception:
        logger = get_runner_logger()
        logger.exception(
            "standard runner FAILED on variant_id=%r (kind=%s)",
            variant_id,
            state["adapter"].KIND if state else "UNKNOWN",
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

    if condition_on_list:
        condition_set = set(condition_on_list)
        variant_ids = [vid for vid in variant_ids if vid not in condition_set]

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
        partial(_run_standard_one_variant, state=state),
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

    for c in ("MAF_by_col_str", "HLA_freqs_str", "AA_AF_by_col_str", "COND_VARIANTS", "SKIPPED_COND_VARIANTS"):
        if c in out.columns:
            out[c] = out[c].replace("", np.nan)

    out = _clean_output(out)
    if adapter.KIND == 'AA':
        out = _ensure_aa_output_info(out)

    if verbose:
        if adapter.KIND == 'AA':
            _report_aa_skips([out])
        print(f"standard[{adapter.KIND}] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

    return out


def _run_many_one_variant(variant_id, *, state):
    adapter = state['adapter']
    try:
        geno, meta = adapter.build_geno(state['hladat'], variant_id, sample_index=state['sample_index'])
    except Exception as exc:
        exc.add_note(f"While preparing genotype columns for variant {variant_id!r}.")
        raise
    geno_cols = list(geno.columns)
    predictors = build_model_predictors(geno, state['covdf'])
    results = {}
    for name, y in state['phenotypes'].items():
        try:
            abt, qc = attach_model_phenotype(predictors, y)
            results[name] = _fit_prepared_row(
                abt, qc, meta, variant_id, adapter, geno_cols, state['covar_cols'],
                state['model_type'], state['condition_on_list'], [])
        except Exception as exc:
            exc.add_note(f"While analysing phenotype {name!r}, variant {variant_id!r}.")
            raise
    return results


def _many_saving_message(verbose):
    if verbose:
        print('You can save multiple output files with:\n\n'
              'for name, result in results.items():\n'
              '    result.to_csv(f"output.{name}.txt", sep="\\t", index=False)', flush=True)


def run_standard_many(adapter, hladat, config, phenotypes, famfile=None, covar=None,
                      condition_on=None, *, variant_filter=None, verbose=True, use_progress_bar=False, print_every=100):
    """One worker pool; prepare each variant once and fit all phenotypes locally."""
    t0 = time.perf_counter()

    if not isinstance(phenotypes, pd.DataFrame):
        raise TypeError("phenotypes must be a pandas DataFrame indexed by sample IID.")
    if phenotypes.empty:
        raise ValueError("phenotypes must contain at least one sample and one phenotype.")
    if not phenotypes.columns.is_unique:
        raise ValueError("Phenotype column names must be unique.")
    if any(not isinstance(name, str) or not name.strip() for name in phenotypes.columns):
        raise ValueError("Phenotype column names must be nonblank strings.")
    sample_index(phenotypes.index, source="Phenotypes")

    ys = {name: prepare_y(famfile, y=phenotypes[name]) for name in phenotypes}
    index = next(iter(ys.values())).index
    covdf = prepare_covar(covar, index)
    covar_cols = list(covdf.columns) if covdf is not None else []
    conditions = _normalize_condition_on(condition_on)
    _validate_conditioning_variants(adapter, hladat, conditions)
    # Conditioning positions are not tested, so the same adjustment is reusable.
    for variant in conditions:
        adjustment, _ = adapter.build_geno(hladat, variant, sample_index=index)
        if adjustment.shape[1] == 0:
            raise ValueError(f"Conditioning variant {variant!r} has no eligible genotype columns.")
        adjustment = adjustment.copy()
        adjustment.columns = [f'COND_{_safe_name(variant)}__{c}' for c in adjustment.columns]
        covdf = adjustment if covdf is None else pd.concat([covdf, adjustment], axis=1)
        covar_cols.extend(adjustment.columns)

    variants = [v for v in adapter.iter_variants(hladat) if v not in set(conditions)]
    if variant_filter is not None:
        info_df = getattr(hladat, adapter.KIND).info
        selected = []
        for variant_id in variants:
            rows = info_df[info_df['AA_ID'] == variant_id] if 'AA_ID' in info_df else info_df.loc[info_df.index == variant_id]
            meta = rows.iloc[0].to_dict() if len(rows) else {}
            ctx = dict(analysis='standard', kind=adapter.KIND,
                       variant_id=variant_id, meta=meta, geno_cols=[])
            if variant_filter(ctx):
                selected.append(variant_id)
        variants = selected
    state = dict(adapter=adapter, hladat=hladat, sample_index=index, covdf=covdf,
                 covar_cols=covar_cols, phenotypes=ys, model_type=config.model_type,
                 condition_on_list=conditions)
    if verbose:
        print("----------------", flush=True)
        print(f"STARTING ANALYSES on:\t{adapter.KIND}, model={config.model_type}", flush=True)
        print(f"VARIANTS SCHEDULED FOR ANALYSIS:\t{len(variants)}", flush=True)
        print(f'Analysing {len(ys)} phenotypes across {len(variants)} variants with shared preparation.', flush=True)
        if conditions:
            print(f"CONDITIONING ON:\t{conditions}", flush=True)
        print(
            f"parallel: n_jobs={config.n_jobs}, backend={config.backend}, "
            f"batch_size={getattr(config, 'batch_size', 32)}, chunksize={getattr(config, 'chunksize', None)}",
            flush=True,
        )
        print("----------------", flush=True)
        print()
    progress = ProgressPrinter(total=len(variants), desc=f'standard_many[{adapter.KIND}]',
        use_tqdm=use_progress_bar, print_every=print_every) if verbose else None
    rows = {name: [] for name in ys}

    for batch in parallel_imap_batched(partial(_run_many_one_variant, state=state), variants,
            config.n_jobs, config.backend, batch_size=getattr(config, 'batch_size', 32),
            max_in_flight=getattr(config, 'chunksize', None)):
        for name, row in batch.items():
            rows[name].append(row)
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    results = {}
    for name, values in rows.items():
        out = pd.DataFrame(values)
        if adapter.KIND == 'AA' and 'LR_p' in out and 'Uni_p' in out:
            out['LRp_Unip'] = out['LR_p'].combine_first(out['Uni_p'])
        for col in ['MAF_by_col_str', 'HLA_freqs_str', 'AA_AF_by_col_str', 'COND_VARIANTS', 'SKIPPED_COND_VARIANTS']:
            if col in out:
                out[col] = out[col].replace('', np.nan)
        results[name] = _clean_output(out)
        if adapter.KIND == 'AA':
            results[name] = _ensure_aa_output_info(results[name])

    if verbose:
        if adapter.KIND == 'AA':
            _report_aa_skips(results.values())
        print(f"standard[{adapter.KIND}] done in {format_seconds(time.perf_counter() - t0)}", flush=True)
    _many_saving_message(verbose)
    return results
