"""
Interaction runner with progress reporting.

Pairwise:
- prints total task count (A_cols × B_cols over variant pairs, after optional pair_filter)
- updates progress as results complete
- includes A_AF/B_AF (numeric only) computed on final modelling rows

Omnibus:
- AA-only block omnibus rule enforced
- includes AA_AF_by_col_str on final modelling rows
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from ..parallel import parallel_imap_batched
from ..progress import ProgressPrinter, format_seconds
from ..preprocess import prepare_y, prepare_covar, make_model_table
from ..models.interaction import fit_pairwise_interaction, fit_block_omnibus_interaction


def _af_safe(series: pd.Series) -> float:
    if not pd.api.types.is_numeric_dtype(series):
        return np.nan
    return float(series.mean() / 2.0)


def run_interaction(
    adapter_a,
    adapter_b,
    hladat,
    famfile: pd.DataFrame,
    config,
    a_kind: str,
    b_kind: str,
    covar=None,
    y=None,
    pair_filter=None,
    cov_block_cols: list[str] | None = None,
    baseline_covar_cols: list[str] | None = None,
    block_b_df: pd.DataFrame | None = None,
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
        if block_b_df.index.dtype != object:
            block_b_df.index = block_b_df.index.astype(str)
        block_b_df = block_b_df.reindex(yser.index)

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
    a_blocks = {vid: adapter_a.build_geno(hladat, vid, sample_index=yser.index) for vid in a_ids}

    if b_kind == "COV":
        b_blocks = {"COV_BLOCK": (covdf[cov_block_cols].copy(), {"VARIANT": "COV_BLOCK"})}
    elif b_kind == "DF":
        b_blocks = {"DF_BLOCK": (block_b_df.copy(), {"VARIANT": "DF_BLOCK"})}
    else:
        b_ids = list(adapter_b.iter_variants(hladat))
        b_blocks = {vid: adapter_b.build_geno(hladat, vid, sample_index=yser.index) for vid in b_ids}

    # ============================
    # Pairwise mode (C1 long)
    # ============================
    if config.mode == "pairwise":
        tasks: list[tuple[str, str, str, str]] = []
        for a_id, (a_df, a_meta) in a_blocks.items():
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

        def work(task: tuple[str, str, str, str]) -> dict:
            a_id, ac, b_id, bc = task
            a_df, a_meta = a_blocks[a_id]
            b_df, b_meta = b_blocks[b_id]

            geno = pd.concat([a_df[[ac]], b_df[[bc]]], axis=1)
            abt, qc = make_model_table(geno, yser, covdf)

            row: dict = {}
            row.update(qc)
            row.update(
                {
                    "A_kind": a_kind,
                    "B_kind": b_kind,
                    "A_variant": a_id,
                    "A_col": ac,
                    "B_variant": b_id,
                    "B_col": bc,
                    "A_AF": np.nan,
                    "B_AF": np.nan,
                    **{f"A_{k}": v for k, v in a_meta.items() if k != "AAcount"},
                    **{f"B_{k}": v for k, v in b_meta.items() if k != "AAcount"},
                }
            )

            if abt.shape[0] == 0:
                row.update(
                    {
                        "I_p": np.nan,
                        "I_coef": np.nan,
                        "I_CI_0.025": np.nan,
                        "I_CI_0.975": np.nan,
                        "LR_p": np.nan,
                        "Anova_p": np.nan,
                    }
                )
                return row

            row["A_AF"] = _af_safe(abt[ac])
            row["B_AF"] = _af_safe(abt[bc])

            row.update(fit_pairwise_interaction(abt, ac, bc, baseline_covars, config.model_type))
            return row

        prog = ProgressPrinter(
            total=len(tasks),
            desc="interaction[pairwise]",
            use_tqdm=use_progress_bar,
            print_every=print_every,
        ) if verbose else None

        rows: list[dict] = []
        for r in parallel_imap_batched(
            work,
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

        if verbose:
            print(f"interaction[pairwise] done in {format_seconds(time.perf_counter() - t0)}", flush=True)

        return out

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


    def work_omni(task) -> dict:
        """
        Run one omnibus interaction task.

        A task always tests:
        - one single anchor column
        - against one multi-column AA block

        Returns one output row as a dict.
        """
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
            abt, qc = make_model_table(geno, yser, covdf)

            # Start output row with QC metrics
            row = {}
            row.update(qc)

            # Add metadata describing this interaction test
            row.update(
                {
                    "A_kind": a_kind,
                    "B_kind": b_kind,
                    "AA_block_side": "A",      # AA block came from side A
                    "AA_variant": a_id,        # identifier for AA block variant
                    "Anchor_variant": b_id,    # identifier for anchor variant
                    "Anchor_col": anchor_col,  # actual anchor column tested
                    "AA_AF_by_col_str": np.nan,
                    # Prefix metadata from the AA block side
                    **{f"AA_{k}": v for k, v in a_meta.items() if k != "AAcount"},
                    # Prefix metadata from the anchor side
                    **{f"Anchor_{k}": v for k, v in b_meta.items() if k != "AAcount"},
                }
            )

            # If no rows remain after merging / NA filtering, return empty stats
            if abt.shape[0] == 0:
                row.update({"LR_p": np.nan, "Anova_p": np.nan, "I_terms": np.nan, "I_block_coefs": np.nan})
                return row

            # Compute allele frequency summary for the AA block columns
            # using only the final analysis table rows
            # af_by = {c: float(abt[c].mean() / 2.0) for c in block_cols if pd.api.types.is_numeric_dtype(abt[c])}
            # row["AA_AF_by_col_str"] = ";".join([f"{k}={v:.6g}" for k, v in af_by.items()]) if af_by else np.nan

            # Fit omnibus interaction model:
            # anchor_col x all block_cols jointly
            row.update(fit_block_omnibus_interaction(abt, anchor_col, block_cols, baseline_covars, config.model_type))
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
        abt, qc = make_model_table(geno, yser, covdf)

        # Start output row with QC metrics
        row = {}
        row.update(qc)

        # Add metadata describing this interaction test
        row.update(
            {
                "A_kind": a_kind,
                "B_kind": b_kind,
                "AA_block_side": "B",      # AA block came from side B
                "AA_variant": b_id,        # identifier for AA block variant
                "Anchor_variant": a_id,    # identifier for anchor variant
                "Anchor_col": anchor_col,  # actual anchor column tested
                "AA_AF_by_col_str": np.nan,
                # Prefix metadata from anchor side
                **{f"Anchor_{k}": v for k, v in a_meta.items() if k != "AAcount"},
                # Prefix metadata from AA block side
                **{f"AA_{k}": v for k, v in b_meta.items() if k != "AAcount"},
            }
        )

        # If analysis table is empty, return row with NA stats
        if abt.shape[0] == 0:
            row.update({"LR_p": np.nan, "Anova_p": np.nan, "I_terms": np.nan, "I_block_coefs": np.nan})
            return row

        # Compute AF summary for the AA block columns on final modelling rows
        # af_by = {c: float(abt[c].mean() / 2.0) for c in block_cols if pd.api.types.is_numeric_dtype(abt[c])}
        # row["AA_AF_by_col_str"] = ";".join([f"{k}={v:.6g}" for k, v in af_by.items()]) if af_by else np.nan

        # Fit omnibus interaction model:
        # anchor_col x all block_cols jointly
        row.update(fit_block_omnibus_interaction(abt, anchor_col, block_cols, baseline_covars, config.model_type))
        return row


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
        work_omni,
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
    for c in ("AA_AF_by_col_str", "I_block_coefs"):
        if c in out.columns:
            out[c] = out[c].replace("", np.nan)
    out = out.dropna(axis=1, how="all")
    return out