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
from ..preprocess import prepare_y, prepare_covar, make_model_table, build_model_predictors, attach_model_phenotype, sample_index, _case_control_counts
from ..models.standard import fit_univariate, fit_omnibus
from ..models import standard as standard_models
from ..adapters._samples import null_sample_ids
from ..adapters.aa import AAAdapter, aa_frequency
from ..models._residue import can_reverse, reverse_standard
from statsmodels.tools.sm_exceptions import PerfectSeparationError




def _order_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Put annotations/tests first and modelling sample counts/INFO last."""
    first = [
        "VARIANT", "GENE", "AA_POS", "Amino_Acids", "Ref_AA", "AAcount", "MAF","HLA_AF","AA_AF",
        "TEST", "P-value", "Statistic", "Beta", "StdErr", "CI-lower", "CI-upper", "DoF",
    ]
    last = ["N_used", "N_case", "N_control", "INFO"]
    first = [c for c in first if c in dataframe.columns]
    last = [c for c in last if c in dataframe.columns]
    middle = [c for c in dataframe.columns if c not in first + last]
    return dataframe[first + middle + last]


def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.drop(columns=["N_total"], errors="ignore")
    if "N_missing_Y" in df.columns and df.N_missing_Y.nunique() == 1 and df.N_missing_Y.unique()[0] == 0:
        df = df.drop(columns=["N_missing_Y"])
    if "N_COND_VARIANTS" in df.columns and df.N_COND_VARIANTS.nunique() == 1 and df.N_COND_VARIANTS.unique()[0] == 0:
        df = df.drop(columns=["N_COND_VARIANTS"])
    if "N_dropped_other" in df.columns and df.N_dropped_other.nunique() == 1 and df.N_dropped_other.unique()[0] == 0:
        df = df.drop(columns=["N_dropped_other"])
    # Preserve missing statistics when fitted models fail diagnostic checks.
    keep = ["INFO"]
    if "INFO" in df and df["INFO"].fillna("").ne("").any():
        keep.extend(c for c in ("P-value", "Beta", "StdErr", "CI-lower", "CI-upper")
                    if c in df)
    df = df.drop(columns=[c for c in df if c not in keep and df[c].isna().all()])
    return _order_output(df)


def _ensure_aa_output_info(out):
    """Keep INFO and missing P-values available even when every test skips."""
    if 'INFO' not in out:
        out['INFO'] = ''
    if out.INFO.eq('insufficient variation').any():
        if "P-value" not in out:
            out[ "P-value"] = np.nan
    return _order_output(out)


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


def _prepare_conditions(adapter, hladat, index, covdf, conditions):
    """Construct the same adjustment columns once, independently of tested variants."""
    covar_cols = list(covdf.columns) if covdf is not None else []
    for variant in conditions:
        adjustment, _ = adapter.build_geno(hladat, variant, sample_index=index)
        if adjustment.shape[1] == 0:
            raise ValueError(f"Conditioning variant {variant!r} has no eligible genotype columns.")
        adjustment = adjustment.copy()
        adjustment.columns = [f'COND_{_safe_name(variant)}__{c}' for c in adjustment.columns]
        covdf = adjustment if covdf is None else pd.concat([covdf, adjustment], axis=1)
        covar_cols.extend(adjustment.columns)
    return covdf, covar_cols


def _prepare_shared_null(ids, y, covdf, covar_cols, model_type):
    if ids is None or len(ids) == 0:
        return None
    baseline, _ = make_model_table(pd.DataFrame(index=ids), y.reindex(ids),
                                   covdf.reindex(ids) if covdf is not None else None)
    try:
        return standard_models.fit_standard_null(baseline, covar_cols, model_type)
    except (PerfectSeparationError, np.linalg.LinAlgError, FloatingPointError, ValueError, ZeroDivisionError):
        # Optional pre-fitting must not change exception behaviour for tests that
        # turn out to be univariate/skipped. An actual omnibus refits locally,
        # retaining the original error reporting if its baseline cannot fit.
        return None


def _fit_prepared_row(abt, qc, meta, variant_id, adapter, geno_cols,
                      covar_cols_local, model_type, used_conditions, skipped_conditions, null_cache=None, aa_geno=None, residue=False, ovr_source=None):
    """Fit one phenotype."""
    row = dict(meta)
    row["VARIANT"] = variant_id
    row["COND_KIND"] = adapter.KIND if used_conditions else np.nan
    row["COND_VARIANTS"] = ";".join(used_conditions) if used_conditions else np.nan
    row["N_COND_VARIANTS"] = len(used_conditions)
    row["SKIPPED_COND_VARIANTS"] = ";".join(skipped_conditions) if skipped_conditions else np.nan
    row.update(qc)
    row.setdefault("INFO", "")

    if adapter.KIND == "SNP":
        row.setdefault("MAF", np.nan)
    elif adapter.KIND == "HLA":
        row.setdefault("HLA_AF", np.nan)

    row.setdefault("P-value", np.nan)
    row.setdefault("TEST", np.nan)
    row.setdefault("Statistic", np.nan)
    row.setdefault("Beta", np.nan)
    row.setdefault("StdErr", np.nan)
    row.setdefault("CI-lower", np.nan)
    row.setdefault("CI-upper", np.nan)

    if aa_geno is not None:
        row["AA_AF"] = aa_frequency(aa_geno, abt.index)

    if len(geno_cols) == 0 or (qc["N_used"] == 0 and not residue):
        return row

    if len(geno_cols) == 1:
        if adapter.KIND == "SNP":
            row["MAF"] = _maf_from_col(abt[geno_cols[0]])
        elif adapter.KIND == "HLA":
            row["HLA_AF"] = _af_from_col(abt[geno_cols[0]])

    model = None
    if qc["N_used"]:
        if len(geno_cols) == 1:
            if residue:
                fitted, model = fit_univariate(abt, geno_cols[0], covar_cols_local, model_type, _return_model=True)
                row.update(fitted)
            else:
                row.update(fit_univariate(abt, geno_cols[0], covar_cols_local, model_type))
        else:
            row.update(fit_omnibus(abt, geno_cols, covar_cols_local, model_type, null_cache=null_cache))
    if not residue or aa_geno is None:
        return row
    rows = [row]
    labels = aa_geno.attrs.get("residue_labels", {})
    # SINGLE already tests its one non-reference residue. MULTI needs every residue.
    extra = [c for c in aa_geno if len(geno_cols) > 1 or c not in geno_cols]
    for col in extra:
        source = ovr_source if ovr_source is not None else abt.assign(**{col: aa_geno[col].reindex(abt.index)})
        ovr_abt = source[[col, *covar_cols_local, "Y"]].dropna()
        dosage = aa_geno[col].reindex(ovr_abt.index)
        ovr = {k: v for k, v in row.items() if k not in {
            "P-value", "Statistic", "Beta", "StdErr", "CI-lower", "CI-upper", "DoF", "TEST", "INFO"}}
        if len(ovr_abt) == 0:
            result = {"INFO": "no complete-case samples"}
        elif (len(geno_cols) == 1 and len(aa_geno.columns) == 2
              and can_reverse(ovr_abt, aa_geno[geno_cols[0]].reindex(ovr_abt.index), dosage, model, model_type, covar_cols_local)
              and pd.notna(row.get("P-value"))):
            result = reverse_standard(row)
        else:
            result = fit_univariate(ovr_abt, col, covar_cols_local, model_type)
        ovr.update(result)
        ovr["Ref_AA"] = f"tested: {labels.get(col, col)}"
        ovr["AA_AF"] = aa_frequency(aa_geno, ovr_abt.index, tested=col)
        ovr["TEST"] = "OVR"
        ovr["N_used"] = len(ovr_abt)
        ovr["N_dropped_other"] = qc["N_total"] - qc["N_missing_Y"] - len(ovr_abt)
        ovr.update(_case_control_counts(ovr_abt))
        rows.append(ovr)
    return rows


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
        all_geno, meta = adapter.build_geno(hladat, variant_id, sample_index=yser.index,
            **({"retain_reference": True} if adapter is AAAdapter else {}))
        aa_geno = all_geno if adapter is AAAdapter else None
        geno_cols = list(all_geno.attrs.get("test_columns", all_geno.columns))
        geno_df = all_geno[geno_cols]

        covdf_local = covdf
        covar_cols_local = covar_cols
        used_conditions = condition_on_list
        skipped_conditions = []

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
        ovr_source = None
        if state.get("residue", False) and aa_geno is not None:
            ovr_source = pd.concat([build_model_predictors(aa_geno, covdf_local), yser.rename("Y")], axis=1)

        return _fit_prepared_row(abt, qc, meta, variant_id, adapter, geno_cols,
                                 covar_cols_local, model_type, used_conditions, skipped_conditions,
                                 null_cache=state.get("null_cache"), aa_geno=aa_geno, residue=state.get("residue", False), ovr_source=ovr_source)

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
    residue: bool = False,
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

    if variant_ids:
        covdf, covar_cols = _prepare_conditions(adapter, hladat, yser.index, covdf, condition_on_list)
    genetic_ids = null_sample_ids(adapter, hladat, yser.index, variant_ids)
    null_cache = _prepare_shared_null(genetic_ids, yser, covdf, covar_cols, config.model_type)

    state = {
        "adapter": adapter,
        "hladat": hladat,
        "yser": yser,
        "covdf": covdf,
        "covar_cols": covar_cols,
        "model_type": config.model_type,
        "variant_filter": variant_filter,
        "condition_on_list": condition_on_list,
        "null_cache": null_cache,
        "residue": residue,
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
        rows.extend(r if isinstance(r, list) else [r])
        if prog is not None:
            prog.update(1)

    if prog is not None:
        prog.close()

    out = pd.DataFrame(rows)

    for c in ("COND_VARIANTS", "SKIPPED_COND_VARIANTS"):
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
        all_geno, meta = adapter.build_geno(state['hladat'], variant_id, sample_index=state['sample_index'],
            **({'retain_reference': True} if adapter is AAAdapter else {}))
        geno = all_geno[list(all_geno.attrs.get('test_columns', all_geno.columns))]
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
                state['model_type'], state['condition_on_list'], [],
                null_cache=state.get('null_caches', {}).get(name),
                aa_geno=all_geno if adapter is AAAdapter else None)
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
    covdf, covar_cols = _prepare_conditions(adapter, hladat, index, covdf, conditions)

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
    genetic_ids = null_sample_ids(adapter, hladat, index, variants)
    null_caches = {name: _prepare_shared_null(genetic_ids, y, covdf, covar_cols, config.model_type)
                   for name, y in ys.items()}
    state = dict(adapter=adapter, hladat=hladat, sample_index=index, covdf=covdf,
                 covar_cols=covar_cols, phenotypes=ys, model_type=config.model_type,
                 condition_on_list=conditions, null_caches=null_caches)
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
        for col in ['COND_VARIANTS', 'SKIPPED_COND_VARIANTS']:
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
