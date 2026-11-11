"""
Interaction runner with progress reporting.

Pairwise:
- prints total task count (A_cols × B_cols over variant pairs, after optional pair_filter)
- updates progress as results complete
- reports genotype AFs or COV predictor means on final modelling rows

Omnibus:
- AA-only block omnibus rule enforced

Conditional A-side adjustment:
- optional same-kind condition_on variants are built from adapter_a
- conditioning columns are appended to the covariate dataframe
- any task whose tested A variant is in condition_on is skipped entirely
"""

from __future__ import annotations

import time
from functools import partial
import numpy as np
import pandas as pd

from ..parallel import parallel_imap_batched
from ..progress import ProgressPrinter, format_seconds
from ..preprocess import prepare_y, prepare_covar, make_model_table, sample_index
from ..models.interaction import fit_pairwise_interaction, fit_block_omnibus_interaction, _failed_result


def _af_safe(series: pd.Series) -> float:
    if not pd.api.types.is_numeric_dtype(series):
        return np.nan
    return float(series.mean() / 2.0)

def _predictor_summary(series: pd.Series, kind: str, prefix: str) -> dict:
    """Report genotype AF, numeric covariate mean or a binary proportion.

    DF partners use the genotype-dosage convention; COV marks covariates.
    Binary categoricals report the second level; other categorical sizes are NaN.
    """
    if kind != "COV":
        return {f"{prefix}_AF": _af_safe(series)}
    if (not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)):
        proportion = np.nan
        levels = pd.Categorical(series).categories
        n = int(series.notna().sum())
        if len(levels) == 2 and n:
            proportion = float(series.eq(levels[1]).sum() / n)
        return {f"{prefix}_proportion": proportion}
    value = np.nan
    if pd.api.types.is_numeric_dtype(series):
        mean = series.mean()
        if pd.notna(mean):
            value = float(mean)
    return {f"{prefix}_mean": value}


def _clean_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.drop(columns=["N_total"], errors="ignore")
    if df.empty:
        return df
    if df.N_missing_Y.nunique()==1 and df.N_missing_Y.unique()[0]==0:
        df = df.drop(columns = ['N_missing_Y'])
    if df.N_dropped_other.nunique()==1 and df.N_dropped_other.unique()[0]==0:
        df = df.drop(columns = ['N_dropped_other'])

    if "INFO" in df:
        df["INFO"] = df["INFO"].replace("", np.nan)
    keep = set()
    if "INFO" in df and df["INFO"].notna().any():
        keep.update(("P-value", "Statistic", "LR_p", "DoF", "Beta", "StdErr",
                     "CI-lower", "CI-upper", "Variant_p", "Variant_Beta",
                     "Partner_p", "Partner_Beta"))
    df = df.drop(columns=[c for c in df if c not in keep and df[c].isna().all()])
    if "INFO" in df:
        df = df[[c for c in df.columns if c != "INFO"] + ["INFO"]]

    return df


def _order_interaction_output(out: pd.DataFrame) -> pd.DataFrame:
    """Order the already named interaction output columns."""
    first = [
        "VARIANT", "GENE", "AA_POS", "Amino_Acids", "Ref_AA", "AAcount","HLA_AF","MAF", "AA_AF",
        "Variant_col", "Partner_col", "Partner_mean", "Partner_proportion", "Partner_AF",
        "P-value", "Statistic", "Beta", "StdErr", "CI-lower", "CI-upper", "DoF",
        "Variant_p", "Variant_Beta", "Partner_p", "Partner_Beta",
    ]
    last = ["N_used", "N_case", "N_control", "INFO"]
    first = [c for c in first if c in out.columns]
    last = [c for c in last if c in out.columns]
    middle = [c for c in out.columns if c not in first + last]
    return out[first + middle + last]


def _normalize_condition_on(condition_on) -> list[str]:
    if condition_on is None:
        return []
    if isinstance(condition_on, str):
        return [condition_on]
    if isinstance(condition_on, (list, tuple, set, pd.Index, np.ndarray)):
        return [str(x) for x in condition_on]
    raise TypeError("condition_on must be None, a string, or a list-like of strings.")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))


def _build_condition_covariates(adapter_a, hladat, variant_ids: list[str], sample_index) -> tuple[pd.DataFrame | None, list[str]]:
    if not variant_ids:
        return None, []

    parts: list[pd.DataFrame] = []
    cond_cols: list[str] = []

    for vid in variant_ids:
        geno_df, _meta = adapter_a.build_geno(hladat, vid, sample_index=sample_index)
        if geno_df is None or geno_df.shape[1] == 0:
            continue

        prefix = f"COND_A_{_safe_name(vid)}"
        renamed = geno_df.copy()
        renamed.columns = [f"{prefix}__{_safe_name(c)}" for c in renamed.columns]
        parts.append(renamed)
        cond_cols.extend(list(renamed.columns))

    if not parts:
        return None, []

    cond_df = pd.concat(parts, axis=1)
    return cond_df, cond_cols


def _run_omnibus_task(task, *, state: dict) -> dict:
    """
    Run one omnibus interaction task.

    A task always tests:
    - one single partner column
    - against one multi-column AA block

    Returns one output row as a dict.
    """
    a_blocks = state["a_blocks"]
    b_blocks = state["b_blocks"]
    yser = state["yser"]
    covdf_model = state["covdf_model"]
    baseline_covars_model = state["baseline_covars_model"]
    kind = state["kind"]
    partner_kind = state["partner_kind"]
    config = state["config"]
    tag = task[0]

    if tag == "A_BLOCK":
        # Task layout:
        # ("A_BLOCK", a_id, block_cols, b_id, partner_col)
        #
        # Here:
        # - A is the AA block
        # - B contributes the partner column
        _, a_id, block_cols, b_id, partner_col = task
        a_df, a_meta = a_blocks[a_id]
        b_df, b_meta = b_blocks[b_id]

        # Build genotype design:
        # - one partner column from B
        # - all AA block columns from A
        geno = pd.concat([b_df[[partner_col]], a_df[block_cols]], axis=1)

        # Join genotype data with phenotype/covariates and compute QC
        abt, qc = make_model_table(geno, yser, covdf_model)

        # Start output row with QC metrics
        row = {}
        row.update(qc)

        # Add metadata describing this interaction test
        row.update(
            {
                **{k: v for k, v in a_meta.items()
                   if k not in {"AAcount", "VARIANT", "INFO"}},
                "VARIANT": a_id,
                "Partner_col": partner_col,
                ("Partner_mean" if partner_kind == "COV" else "Partner_AF"): np.nan,
                **{f"Partner_{k}": v for k, v in b_meta.items()
                   if k not in {"AAcount", "VARIANT"}},
            }
        )

        # If no rows remain after merging / NA filtering, return empty stats
        if abt.shape[0] == 0:
            row.update(_failed_result("no complete-case samples", omnibus=True))
            return row
        row.update(_predictor_summary(abt[partner_col], partner_kind, "Partner"))

        # Fit omnibus interaction model:
        # partner_col x all block_cols jointly
        row.update(fit_block_omnibus_interaction(abt, partner_col, block_cols, baseline_covars_model, config.model_type))
        return row

    # B_BLOCK case
    #
    # Task layout:
    # ("B_BLOCK", a_id, partner_col, b_id, block_cols)
    #
    # Here:
    # - B is the AA block
    # - A contributes the partner column
    _, a_id, partner_col, b_id, block_cols = task
    a_df, a_meta = a_blocks[a_id]
    b_df, b_meta = b_blocks[b_id]

    # Build genotype design:
    # - one partner column from A
    # - all AA block columns from B
    geno = pd.concat([a_df[[partner_col]], b_df[block_cols]], axis=1)

    # Join genotype data with phenotype/covariates and compute QC
    abt, qc = make_model_table(geno, yser, covdf_model)

    # Start output row with QC metrics
    row = {}
    row.update(qc)

    # Add metadata describing this interaction test
    row.update(
        {
            **{k: v for k, v in b_meta.items()
               if k not in {"AAcount", "VARIANT", "INFO"}},
            "VARIANT": b_id,
            "Partner_col": partner_col,
            "Partner_AF": np.nan,
            **{f"Partner_{k}": v for k, v in a_meta.items()
               if k not in {"AAcount", "VARIANT"}},
        }
    )



    # If analysis table is empty, return row with NA stats
    if abt.shape[0] == 0:
        row.update(_failed_result("no complete-case samples", omnibus=True))
        return row
    row.update(_predictor_summary(abt[partner_col], kind, "Partner"))


    # Fit omnibus interaction model:
    # partner_col x all block_cols jointly
    row.update(fit_block_omnibus_interaction(abt, partner_col, block_cols, baseline_covars_model, config.model_type))
    return row


def _run_pairwise_task(task: tuple[str, str, str, str], *, state: dict) -> dict:
    a_blocks = state["a_blocks"]
    b_blocks = state["b_blocks"]
    yser = state["yser"]
    covdf_model = state["covdf_model"]
    baseline_covars_model = state["baseline_covars_model"]
    kind = state["kind"]
    partner_kind = state["partner_kind"]
    config = state["config"]
    a_id, ac, b_id, bc = task
    a_df, a_meta = a_blocks[a_id]
    b_df, b_meta = b_blocks[b_id]

    geno = pd.concat([a_df[[ac]], b_df[[bc]]], axis=1)
    abt, qc = make_model_table(geno, yser, covdf_model)

    row: dict = {}
    row.update(qc)
    row.update(
        {
            **{k: v for k, v in a_meta.items() if k not in {"VARIANT", "INFO"}},
            "VARIANT": a_id,
            **({"Variant_col": ac} if len(a_df.columns) > 1 else {}),
            "Partner_col": bc,
            **{f"Partner_{k}": v for k, v in b_meta.items()
               if k not in {"AAcount", "VARIANT"}},
        }
    )

    if abt.shape[0] == 0:
        row.update(_failed_result("no complete-case samples"))
        return row

    if kind == "HLA":
        row["HLA_AF"] = _af_safe(abt[ac])
    elif kind == "SNP":
        frequency = _af_safe(abt[ac])
        row["MAF"] = min(frequency, 1.0 - frequency)
    row.update(_predictor_summary(abt[bc], partner_kind, "Partner"))

    row.update(fit_pairwise_interaction(abt, ac, bc, baseline_covars_model, config.model_type))
    return row


def _filter_a_variants(hladat, kind, variant_ids, variant_filter):
    """Filter reader metadata once per A variant without constructing genotypes."""
    info = getattr(hladat, kind).info
    if "AA_ID" in info.columns:
        metadata = info.drop_duplicates("AA_ID").set_index("AA_ID", drop=False)
    else:
        metadata = info.loc[~info.index.duplicated()]
    selected = []
    for variant_id in variant_ids:
        meta = metadata.loc[variant_id].to_dict() if variant_id in metadata.index else {}
        ctx = {"analysis": "interaction", "side": "A", "kind": kind,
               "variant_id": variant_id, "meta": meta, "geno_cols": []}
        if variant_filter(ctx):
            selected.append(variant_id)
    return selected


def run_interaction(
    adapter_a,
    adapter_b,
    hladat,
    config,
    kind: str,
    partner_kind: str,
    famfile: pd.DataFrame=None,
    covar=None,
    y=None,
    pair_filter=None,
    cov_block_cols: list[str] | None = None,
    baseline_covar_cols: list[str] | None = None,
    partner_data: pd.DataFrame | None = None,
    condition_on=None,
    *,
    variant_filter=None,
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 500,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    if variant_filter is not None and not callable(variant_filter):
        raise TypeError("variant_filter must be callable or None; use the ID/gene filter helpers.")

    yser = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, yser.index)

    # ----------------------------
    # Validate B block selection
    # ----------------------------
    if partner_kind == "DF":
        if partner_data is None:
            raise ValueError("partner_kind='DF' requires partner_data.")
        if not isinstance(partner_data, pd.DataFrame):
            raise TypeError("partner_data must be a pandas DataFrame.")
        # align by IID index; keep only modelling rows later via make_model_table
        partner_data = partner_data.copy()
        partner_data.index = sample_index(partner_data.index, source="Interaction block")
        partner_data = partner_data.reindex(yser.index)

    # ----------------------------
    # Normalize / validate condition_on
    # ----------------------------
    condition_on_list = _normalize_condition_on(condition_on)
    condition_on_set = set(condition_on_list)


    # baseline covars
    if partner_kind == "COV":
        if covdf is None:
            raise ValueError("partner_kind='COV' requires covar.")
        if cov_block_cols is None:
            raise ValueError("partner_kind='COV' requires cov_block_cols.")
        missing = [c for c in cov_block_cols if c not in covdf.columns]
        if missing:
            raise ValueError(f"cov_block_cols not found: {missing}")
        baseline_covars = (
            list(baseline_covar_cols)
            if baseline_covar_cols is not None
            else [c for c in covdf.columns if c not in set(cov_block_cols)]
        )
    else:
        baseline_covars = []
        if covdf is not None:
            baseline_covars = list(baseline_covar_cols) if baseline_covar_cols is not None else list(covdf.columns)

    # AA-only omnibus rule
    if config.mode == "one_vs_block_omnibus":
        # Omnibus interaction is defined as: (partner_col) x (AA_block_cols)
        # Partner can come from genotypes, covariates, or an external DF block.
        if not (kind == "AA" or partner_kind == "AA"):
            raise ValueError("one_vs_block_omnibus requires AA on A or B.")


    if verbose:
        print("----------------", flush=True)
        print(f"STARTING INTERACTION ANALYSES on:\tmode={config.mode}, model={config.model_type}", flush=True)
        print(f"BLOCKS: A={kind}, B={partner_kind}", flush=True)
        if condition_on_list:
            print(f"conditioning on A-side variants: {condition_on_list}", flush=True)
        print(
            f"parallel: n_jobs={config.n_jobs}, backend={config.backend}, "
            f"batch_size={getattr(config, 'batch_size', 32)}, chunksize={getattr(config, 'chunksize', None)}",
            flush=True,
        )
        print("----------------", flush=True)
        print()

    # ----------------------------
    # Build blocks (memory-heavy)
    # ----------------------------
    a_ids = list(adapter_a.iter_variants(hladat))
    a_id_set = set(a_ids)

    missing_condition_ids = [vid for vid in condition_on_list if vid not in a_id_set]
    if missing_condition_ids:
        raise ValueError(
            f"condition_on contains variant ids not found in A block ({kind}): {missing_condition_ids}"
        )

    # Conditioning blocks are built separately and are never removed by the filter.
    a_ids = [vid for vid in a_ids if vid not in condition_on_set]
    if variant_filter is not None:
        a_ids = _filter_a_variants(hladat, kind, a_ids, variant_filter)
        if verbose:
            print(f"HLA-side variants selected for interaction: {len(a_ids)}", flush=True)
        if not a_ids:
            return pd.DataFrame()

    a_blocks = {vid: adapter_a.build_geno(hladat, vid, sample_index=yser.index) for vid in a_ids}

    if partner_kind == "COV":
        b_blocks = {"COV_BLOCK": (covdf[cov_block_cols].copy(), {"VARIANT": "COV_BLOCK"})}
    elif partner_kind == "DF":

        b_blocks = {"DF_BLOCK": (partner_data.copy(), {"VARIANT": "DF_BLOCK"})}
    else:
        b_ids = list(adapter_b.iter_variants(hladat))
        b_blocks = {vid: adapter_b.build_geno(hladat, vid, sample_index=yser.index) for vid in b_ids}

    cond_df, cond_cols = _build_condition_covariates(
        adapter_a=adapter_a,
        hladat=hladat,
        variant_ids=condition_on_list,
        sample_index=yser.index,
    )

    covdf_model = (
        covdf.drop(columns=cov_block_cols)
        if partner_kind == "COV"
        else covdf
    )

    if cond_df is not None:
        covdf_model = (
            cond_df.copy()
            if covdf_model is None
            else pd.concat([covdf_model, cond_df], axis=1)
        )
        baseline_covars_model = list(baseline_covars) + cond_cols
    else:
        baseline_covars_model = list(baseline_covars)

    worker_state = {
        "a_blocks": a_blocks,
        "b_blocks": b_blocks,
        "yser": yser,
        "covdf_model": covdf_model,
        "baseline_covars_model": baseline_covars_model,
        "kind": kind,
        "partner_kind": partner_kind,
        "config": config,
    }

    # ==============
    # Pairwise mode
    # ==============
    if config.mode == "pairwise":
        tasks: list[tuple[str, str, str, str]] = []
        for a_id, (a_df, a_meta) in a_blocks.items():
            if a_id in condition_on_set:
                continue
            for b_id, (b_df, b_meta) in b_blocks.items():
                for ac in a_df.columns:
                    for bc in b_df.columns:
                        ctx = {
                            "analysis": "interaction",
                            "kind": kind,
                            "partner_kind": partner_kind,
                            "a_id": a_id,
                            "b_id": b_id,
                            "a_meta": a_meta,
                            "b_meta": b_meta,
                            "a_col": ac,
                            "b_col": bc,
                        }
                        if pair_filter is not None and not pair_filter(ctx):
                            continue
                        tasks.append((a_id, ac, b_id, bc))

        if verbose:
            print(f"pairs available to analyse (scheduled tasks): {len(tasks)}", flush=True)


        prog = ProgressPrinter(
            total=len(tasks),
            desc="interaction[pairwise]",
            use_tqdm=use_progress_bar,
            print_every=print_every,
        ) if verbose else None

        rows: list[dict] = []
        for r in parallel_imap_batched(
            partial(_run_pairwise_task, state=worker_state),
            tasks,
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
        out = _clean_output(out)

        if verbose:
            print(f"interaction[pairwise] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

        return _order_interaction_output(out)

    # =============
    # Omnibus mode
    # =============
    # Decide which side should be treated as the AA "block".
    #
    # Idea:
    # - If only A is AA, then A is the AA block.
    # - If only B is AA, then B is the AA block.
    # - If both are AA, or neither is AA, default to treating B as the block side.
    if kind == "AA" and partner_kind != "AA":
        aa_block_side = "A"
    elif partner_kind == "AA" and kind != "AA":
        aa_block_side = "B"
    else:
        aa_block_side = "B"

    # This will store all omnibus interaction tasks to run later.
    # Each task represents:
    # - one partner column (single predictor)
    # - tested against one AA block (multiple columns together)
    tasks_omni = []

    if aa_block_side == "A":
        # A is the AA block, B provides the partner column.
        #
        # For each AA variant in A:
        #   - get all columns in that AA block
        # For each variant in B:
        #   - test each column from B as the partner
        for a_id, (a_df, a_meta) in a_blocks.items():
            if a_id in condition_on_set:
                continue

            block_cols = list(a_df.columns)

            # Skip empty AA blocks
            if not block_cols:
                continue

            for b_id, (b_df, b_meta) in b_blocks.items():
                for partner_col in b_df.columns:
                    # Build a context dict for optional filtering.
                    # This lets pair_filter decide whether this partner/block test
                    # should be kept or skipped.
                    ctx = {
                        "analysis": "interaction",
                        "kind": kind,
                        "partner_kind": partner_kind,
                        "a_id": a_id,
                        "b_id": b_id,
                        "a_meta": a_meta,
                        "b_meta": b_meta,
                        "partner_col": partner_col,
                        "block_cols": block_cols,
                    }

                    # Skip task if user-supplied filter says so
                    if pair_filter is not None and not pair_filter(ctx):
                        continue

                    # Store task in a compact tuple.
                    #
                    # Format for A_BLOCK:
                    # ("A_BLOCK", a_id, block_cols, b_id, partner_col)
                    #
                    # Meaning:
                    # - AA block is on side A
                    # - use a_id block_cols as the omnibus block
                    # - use b_id / partner_col as the single partner variable
                    tasks_omni.append(("A_BLOCK", a_id, block_cols, b_id, partner_col))
    else:
        # B is the AA block, A provides the partner column.
        #
        # For each AA variant in B:
        #   - get all columns in that AA block
        # For each variant in A:
        #   - test each column from A as the partner
        for b_id, (b_df, b_meta) in b_blocks.items():
            block_cols = list(b_df.columns)

            # Skip empty AA blocks
            if not block_cols:
                continue

            for a_id, (a_df, a_meta) in a_blocks.items():
                if a_id in condition_on_set:
                    continue
                for partner_col in a_df.columns:
                    # Build filter context for this candidate partner/block pair
                    ctx = {
                        "analysis": "interaction",
                        "kind": kind,
                        "partner_kind": partner_kind,
                        "a_id": a_id,
                        "b_id": b_id,
                        "a_meta": a_meta,
                        "b_meta": b_meta,
                        "partner_col": partner_col,
                        "block_cols": block_cols,
                    }

                    # Skip if filtered out
                    if pair_filter is not None and not pair_filter(ctx):
                        continue

                    # Store task in compact tuple.
                    #
                    # Format for B_BLOCK:
                    # ("B_BLOCK", a_id, partner_col, b_id, block_cols)
                    #
                    # Meaning:
                    # - AA block is on side B
                    # - use b_id block_cols as the omnibus block
                    # - use a_id / partner_col as the single partner variable
                    tasks_omni.append(("B_BLOCK", a_id, partner_col, b_id, block_cols))

    # Optional progress message showing how many omnibus tests were scheduled
    if verbose:
        print(f"omnibus tasks available to analyse (scheduled): {len(tasks_omni)}", flush=True)




    # Optional progress printer for omnibus interaction jobs
    prog = ProgressPrinter(
        total=len(tasks_omni),
        desc="interaction[omnibus]",
        use_tqdm=use_progress_bar,
        print_every=print_every,
    ) if verbose else None

    rows = []

    # Run all omnibus tasks in parallel, batched for lower scheduling overhead
    for r in parallel_imap_batched(
        partial(_run_omnibus_task, state=worker_state),
        tasks_omni,
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

    # Combine all per-task output rows into a final DataFrame
    out = pd.DataFrame(rows)

    if verbose:
        print(f"interaction[omnibus] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

    # clean: drop all-empty columns; also allow empty strings to be dropped
    for c in ("I_block_coefs",):
        if c in out.columns:
            out[c] = out[c].replace("", np.nan)
    if "P-value" in out.columns:
        out = out.sort_values("P-value")

    out = _clean_output(out)
    return _order_interaction_output(out)
