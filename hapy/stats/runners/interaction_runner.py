"""
Interaction runner with progress reporting.

Pairwise:
- prints total task count (A_cols × B_cols over variant pairs, after optional pair_filter)
- updates progress as results complete
- reports genotype AFs or COV predictor means on final modelling rows

Omnibus:
- AA-only block omnibus rule enforced

Conditional A-side adjustment:
- optional same-kind condition_on_a variants are built from adapter_a
- conditioning columns are appended to the covariate dataframe
- any task whose tested A variant is in condition_on_a is skipped entirely
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
                     "Anchor_p", "Anchor_Beta"))
    df = df.drop(columns=[c for c in df if c not in keep and df[c].isna().all()])
    if "INFO" in df:
        df = df[[c for c in df.columns if c != "INFO"] + ["INFO"]]

    return df


def _order_interaction_output(out: pd.DataFrame) -> pd.DataFrame:
    """Order the already named interaction output columns."""
    first = [
        "VARIANT", "GENE", "AA_POS", "Amino_Acids", "Ref_AA", "AAcount","HLA_AF","MAF", "AA_AF",
        "Variant_col", "Anchor_col", "Anchor_mean", "Anchor_proportion", "Anchor_AF",
        "P-value", "Statistic", "Beta", "StdErr", "CI-lower", "CI-upper", "DoF",
        "Variant_p", "Variant_Beta", "Anchor_p", "Anchor_Beta",
    ]
    last = ["N_used", "N_case", "N_control", "INFO"]
    first = [c for c in first if c in out.columns]
    last = [c for c in last if c in out.columns]
    middle = [c for c in out.columns if c not in first + last]
    return out[first + middle + last]


def _normalize_condition_on(condition_on_a) -> list[str]:
    if condition_on_a is None:
        return []
    if isinstance(condition_on_a, str):
        return [condition_on_a]
    if isinstance(condition_on_a, (list, tuple, set, pd.Index, np.ndarray)):
        return [str(x) for x in condition_on_a]
    raise TypeError("condition_on_a must be None, a string, or a list-like of strings.")


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
    - one single anchor column
    - against one multi-column AA block

    Returns one output row as a dict.
    """
    a_blocks = state["a_blocks"]
    b_blocks = state["b_blocks"]
    yser = state["yser"]
    covdf_model = state["covdf_model"]
    baseline_covars_model = state["baseline_covars_model"]
    a_kind = state["a_kind"]
    b_kind = state["b_kind"]
    config = state["config"]
    tag = task[0]

    if tag == "A_BLOCK":
        # Task layout:
        # ("A_BLOCK", a_id, block_cols, b_id, anchor_col)
        #
        # Here:
        # - A is the AA block
        # - B contributes the anchor column
        _, a_id, block_cols, b_id, anchor_col = task
        a_df, a_meta = a_blocks[a_id]
        b_df, b_meta = b_blocks[b_id]

        # Build genotype design:
        # - one anchor column from B
        # - all AA block columns from A
        geno = pd.concat([b_df[[anchor_col]], a_df[block_cols]], axis=1)

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
                "Anchor_col": anchor_col,
                ("Anchor_mean" if b_kind == "COV" else "Anchor_AF"): np.nan,
                **{f"Anchor_{k}": v for k, v in b_meta.items()
                   if k not in {"AAcount", "VARIANT"}},
            }
        )

        # If no rows remain after merging / NA filtering, return empty stats
        if abt.shape[0] == 0:
            row.update(_failed_result("no complete-case samples", omnibus=True))
            return row
        row.update(_predictor_summary(abt[anchor_col], b_kind, "Anchor"))

        # Fit omnibus interaction model:
        # anchor_col x all block_cols jointly
        row.update(fit_block_omnibus_interaction(abt, anchor_col, block_cols, baseline_covars_model, config.model_type))
        return row

    # B_BLOCK case
    #
    # Task layout:
    # ("B_BLOCK", a_id, anchor_col, b_id, block_cols)
    #
    # Here:
    # - B is the AA block
    # - A contributes the anchor column
    _, a_id, anchor_col, b_id, block_cols = task
    a_df, a_meta = a_blocks[a_id]
    b_df, b_meta = b_blocks[b_id]

    # Build genotype design:
    # - one anchor column from A
    # - all AA block columns from B
    geno = pd.concat([a_df[[anchor_col]], b_df[block_cols]], axis=1)

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
            "Anchor_col": anchor_col,
            "Anchor_AF": np.nan,
            **{f"Anchor_{k}": v for k, v in a_meta.items()
               if k not in {"AAcount", "VARIANT"}},
        }
    )



    # If analysis table is empty, return row with NA stats
    if abt.shape[0] == 0:
        row.update(_failed_result("no complete-case samples", omnibus=True))
        return row
    row.update(_predictor_summary(abt[anchor_col], a_kind, "Anchor"))


    # Fit omnibus interaction model:
    # anchor_col x all block_cols jointly
    row.update(fit_block_omnibus_interaction(abt, anchor_col, block_cols, baseline_covars_model, config.model_type))
    return row


def _run_pairwise_task(task: tuple[str, str, str, str], *, state: dict) -> dict:
    a_blocks = state["a_blocks"]
    b_blocks = state["b_blocks"]
    yser = state["yser"]
    covdf_model = state["covdf_model"]
    baseline_covars_model = state["baseline_covars_model"]
    a_kind = state["a_kind"]
    b_kind = state["b_kind"]
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
            "Anchor_col": bc,
            **{f"Anchor_{k}": v for k, v in b_meta.items()
               if k not in {"AAcount", "VARIANT"}},
        }
    )

    if abt.shape[0] == 0:
        row.update(_failed_result("no complete-case samples"))
        return row

    if a_kind == "HLA":
        row["HLA_AF"] = _af_safe(abt[ac])
    elif a_kind == "SNP":
        frequency = _af_safe(abt[ac])
        row["MAF"] = min(frequency, 1.0 - frequency)
    row.update(_predictor_summary(abt[bc], b_kind, "Anchor"))

    row.update(fit_pairwise_interaction(abt, ac, bc, baseline_covars_model, config.model_type))
    return row


def run_interaction(
    adapter_a,
    adapter_b,
    hladat,
    config,
    a_kind: str,
    b_kind: str,
    famfile: pd.DataFrame=None,
    covar=None,
    y=None,
    pair_filter=None,
    cov_block_cols: list[str] | None = None,
    baseline_covar_cols: list[str] | None = None,
    block_b_df: pd.DataFrame | None = None,
    condition_on_a=None,
    *,
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 500,
) -> pd.DataFrame:
    t0 = time.perf_counter()

    yser = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, yser.index)

    # ----------------------------
    # Validate B block selection
    # ----------------------------
    if b_kind == "DF":
        if block_b_df is None:
            raise ValueError("b_kind='DF' requires block_b_df.")
        if not isinstance(block_b_df, pd.DataFrame):
            raise TypeError("block_b_df must be a pandas DataFrame.")
        # align by IID index; keep only modelling rows later via make_model_table
        block_b_df = block_b_df.copy()
        block_b_df.index = sample_index(block_b_df.index, source="Interaction block B")
        block_b_df = block_b_df.reindex(yser.index)

    # ----------------------------
    # Normalize / validate condition_on_a
    # ----------------------------
    condition_on_a_list = _normalize_condition_on(condition_on_a)
    condition_on_a_set = set(condition_on_a_list)


    # baseline covars
    if b_kind == "COV":
        if covdf is None:
            raise ValueError("b_kind='COV' requires covar.")
        if cov_block_cols is None:
            raise ValueError("b_kind='COV' requires cov_block_cols.")
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
        # Omnibus interaction is defined as: (anchor_col) x (AA_block_cols)
        # Anchor can come from genotypes, covariates, or an external DF block.
        if not (a_kind == "AA" or b_kind == "AA"):
            raise ValueError("one_vs_block_omnibus requires AA on A or B.")


    if verbose:
        print("----------------", flush=True)
        print(f"STARTING INTERACTION ANALYSES on:\tmode={config.mode}, model={config.model_type}", flush=True)
        print(f"BLOCKS: A={a_kind}, B={b_kind}", flush=True)
        if condition_on_a_list:
            print(f"conditioning on A-side variants: {condition_on_a_list}", flush=True)
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

    missing_condition_ids = [vid for vid in condition_on_a_list if vid not in a_id_set]
    if missing_condition_ids:
        raise ValueError(
            f"condition_on_a contains variant ids not found in A block ({a_kind}): {missing_condition_ids}"
        )

    a_blocks = {vid: adapter_a.build_geno(hladat, vid, sample_index=yser.index) for vid in a_ids}

    if b_kind == "COV":
        b_blocks = {"COV_BLOCK": (covdf[cov_block_cols].copy(), {"VARIANT": "COV_BLOCK"})}
    elif b_kind == "DF":

        b_blocks = {"DF_BLOCK": (block_b_df.copy(), {"VARIANT": "DF_BLOCK"})}
    else:
        b_ids = list(adapter_b.iter_variants(hladat))
        b_blocks = {vid: adapter_b.build_geno(hladat, vid, sample_index=yser.index) for vid in b_ids}

    cond_df, cond_cols = _build_condition_covariates(
        adapter_a=adapter_a,
        hladat=hladat,
        variant_ids=condition_on_a_list,
        sample_index=yser.index,
    )

    covdf_model = (
        covdf.drop(columns=cov_block_cols)
        if b_kind == "COV"
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
        "a_kind": a_kind,
        "b_kind": b_kind,
        "config": config,
    }

    # ==============
    # Pairwise mode
    # ==============
    if config.mode == "pairwise":
        tasks: list[tuple[str, str, str, str]] = []
        for a_id, (a_df, a_meta) in a_blocks.items():
            if a_id in condition_on_a_set:
                continue
            for b_id, (b_df, b_meta) in b_blocks.items():
                for ac in a_df.columns:
                    for bc in b_df.columns:
                        ctx = {
                            "analysis": "interaction",
                            "a_kind": a_kind,
                            "b_kind": b_kind,
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
    if a_kind == "AA" and b_kind != "AA":
        aa_block_side = "A"
    elif b_kind == "AA" and a_kind != "AA":
        aa_block_side = "B"
    else:
        aa_block_side = "B"

    # This will store all omnibus interaction tasks to run later.
    # Each task represents:
    # - one anchor column (single predictor)
    # - tested against one AA block (multiple columns together)
    tasks_omni = []

    if aa_block_side == "A":
        # A is the AA block, B provides the anchor column.
        #
        # For each AA variant in A:
        #   - get all columns in that AA block
        # For each variant in B:
        #   - test each column from B as the anchor
        for a_id, (a_df, a_meta) in a_blocks.items():
            if a_id in condition_on_a_set:
                continue

            block_cols = list(a_df.columns)

            # Skip empty AA blocks
            if not block_cols:
                continue

            for b_id, (b_df, b_meta) in b_blocks.items():
                for anchor_col in b_df.columns:
                    # Build a context dict for optional filtering.
                    # This lets pair_filter decide whether this anchor/block test
                    # should be kept or skipped.
                    ctx = {
                        "analysis": "interaction",
                        "a_kind": a_kind,
                        "b_kind": b_kind,
                        "a_id": a_id,
                        "b_id": b_id,
                        "a_meta": a_meta,
                        "b_meta": b_meta,
                        "anchor_col": anchor_col,
                        "block_cols": block_cols,
                    }

                    # Skip task if user-supplied filter says so
                    if pair_filter is not None and not pair_filter(ctx):
                        continue

                    # Store task in a compact tuple.
                    #
                    # Format for A_BLOCK:
                    # ("A_BLOCK", a_id, block_cols, b_id, anchor_col)
                    #
                    # Meaning:
                    # - AA block is on side A
                    # - use a_id block_cols as the omnibus block
                    # - use b_id / anchor_col as the single anchor variable
                    tasks_omni.append(("A_BLOCK", a_id, block_cols, b_id, anchor_col))
    else:
        # B is the AA block, A provides the anchor column.
        #
        # For each AA variant in B:
        #   - get all columns in that AA block
        # For each variant in A:
        #   - test each column from A as the anchor
        for b_id, (b_df, b_meta) in b_blocks.items():
            block_cols = list(b_df.columns)

            # Skip empty AA blocks
            if not block_cols:
                continue

            for a_id, (a_df, a_meta) in a_blocks.items():
                if a_id in condition_on_a_set:
                    continue
                for anchor_col in a_df.columns:
                    # Build filter context for this candidate anchor/block pair
                    ctx = {
                        "analysis": "interaction",
                        "a_kind": a_kind,
                        "b_kind": b_kind,
                        "a_id": a_id,
                        "b_id": b_id,
                        "a_meta": a_meta,
                        "b_meta": b_meta,
                        "anchor_col": anchor_col,
                        "block_cols": block_cols,
                    }

                    # Skip if filtered out
                    if pair_filter is not None and not pair_filter(ctx):
                        continue

                    # Store task in compact tuple.
                    #
                    # Format for B_BLOCK:
                    # ("B_BLOCK", a_id, anchor_col, b_id, block_cols)
                    #
                    # Meaning:
                    # - AA block is on side B
                    # - use b_id block_cols as the omnibus block
                    # - use a_id / anchor_col as the single anchor variable
                    tasks_omni.append(("B_BLOCK", a_id, anchor_col, b_id, block_cols))

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
