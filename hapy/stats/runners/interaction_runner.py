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
    *,
    verbose: bool = True,
    use_progress_bar: bool = False,
    print_every: int = 500,
) -> pd.DataFrame:
    t0 = time.perf_counter()

    yser = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, yser.index)

    # baseline covars
    if b_kind == "COV":
        if covdf is None:
            raise ValueError("b_kind='COV' requires covar.")
        if cov_block_cols is None:
            raise ValueError("b_kind='COV' requires cov_block_cols.")
        missing = [c for c in cov_block_cols if c not in covdf.columns]
        if missing:
            raise ValueError(f"cov_block_cols not found: {missing}")
        baseline_covars = list(baseline_covar_cols) if baseline_covar_cols is not None else [c for c in covdf.columns if c not in set(cov_block_cols)]
    else:
        baseline_covars = []
        if covdf is not None:
            baseline_covars = list(baseline_covar_cols) if baseline_covar_cols is not None else list(covdf.columns)

    # AA-only omnibus rule
    if config.mode == "one_vs_block_omnibus":
        if b_kind == "COV":
            raise ValueError("one_vs_block_omnibus not supported with b_kind='COV'.")
        if not (a_kind == "AA" or b_kind == "AA"):
            raise ValueError("one_vs_block_omnibus requires AA on A or B.")

    if verbose:
        print("----------------", flush=True)
        print(f"STARTING INTERACTION ANALYSES on:\tmode={config.mode}, model={config.model_type}", flush=True)
        print(f"BLOCKS: A={a_kind}, B={b_kind}", flush=True)
        print(f"parallel: n_jobs={config.n_jobs}, backend={config.backend}", flush=True)
        print("----------------", flush=True)
        print()

    # Build blocks (simple but memory-heavy)
    a_ids = adapter_a.iter_variants(hladat)
    a_blocks = {vid: adapter_a.build_geno(hladat, vid, sample_index=yser.index) for vid in a_ids}

    if b_kind == "COV":
        b_blocks = {"COV_BLOCK": (covdf[cov_block_cols].copy(), {"VARIANT": "COV_BLOCK"})}
    else:
        b_ids = adapter_b.iter_variants(hladat)
        b_blocks = {vid: adapter_b.build_geno(hladat, vid, sample_index=yser.index) for vid in b_ids}

    # ---- pairwise ----
    if config.mode == "pairwise":
        tasks = []
        for a_id, (a_df, a_meta) in a_blocks.items():
            for b_id, (b_df, b_meta) in b_blocks.items():
                for ac in a_df.columns:
                    for bc in b_df.columns:
                        ctx = {
                            "analysis": "interaction",
                            "a_kind": a_kind, "b_kind": b_kind,
                            "a_id": a_id, "b_id": b_id,
                            "a_meta": a_meta, "b_meta": b_meta,
                            "a_col": ac, "b_col": bc,
                        }
                        if pair_filter is not None and not pair_filter(ctx):
                            continue
                        tasks.append((a_id, ac, b_id, bc))

        if verbose:
            print(f"pairs available to analyse (scheduled tasks): {len(tasks)}", flush=True)

        def work(task) -> dict:
            a_id, ac, b_id, bc = task
            a_df, a_meta = a_blocks[a_id]
            b_df, b_meta = b_blocks[b_id]

            geno = pd.concat([a_df[[ac]], b_df[[bc]]], axis=1)
            abt, qc = make_model_table(geno, yser, covdf)

            row = {}
            row.update(qc)
            row.update({
                "A_kind": a_kind, "B_kind": b_kind,
                "A_variant": a_id, "A_col": ac,
                "B_variant": b_id, "B_col": bc,
                "A_AF": np.nan,
                "B_AF": np.nan,
                **{f"A_{k}": v for k, v in a_meta.items() if k != "AAcount"},
                **{f"B_{k}": v for k, v in b_meta.items() if k != "AAcount"},
            })

            if abt.shape[0] == 0:
                row.update({"I_p": np.nan, "I_coef": np.nan, "I_CI_0.025": np.nan, "I_CI_0.975": np.nan, "LR_p": np.nan, "Anova_p": np.nan})
                return row

            row["A_AF"] = _af_safe(abt[ac])
            row["B_AF"] = _af_safe(abt[bc])

            row.update(fit_pairwise_interaction(abt, ac, bc, baseline_covars, config.model_type))
            return row

        prog = ProgressPrinter(total=len(tasks), desc="interaction[pairwise]", use_tqdm=use_progress_bar, print_every=print_every) if verbose else None

        rows = []
        for r in parallel_imap(work, tasks, config.n_jobs, config.backend, config.chunksize):
            rows.append(r)
            if prog is not None:
                prog.update(1)

        if prog is not None:
            prog.close()

        out = pd.DataFrame(rows)

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"interaction[pairwise] done in {format_seconds(elapsed)}", flush=True)

        return out

    # ---- omnibus ----
    if a_kind == "AA" and b_kind != "AA":
        aa_block_side = "A"
    elif b_kind == "AA" and a_kind != "AA":
        aa_block_side = "B"
    else:
        aa_block_side = "B"

    tasks_omni = []
    if aa_block_side == "A":
        for a_id, (a_df, a_meta) in a_blocks.items():
            block_cols = list(a_df.columns)
            if not block_cols:
                continue
            for b_id, (b_df, b_meta) in b_blocks.items():
                for anchor_col in b_df.columns:
                    ctx = {
                        "analysis": "interaction",
                        "a_kind": a_kind, "b_kind": b_kind,
                        "a_id": a_id, "b_id": b_id,
                        "a_meta": a_meta, "b_meta": b_meta,
                        "anchor_col": anchor_col,
                        "block_cols": block_cols,
                    }
                    if pair_filter is not None and not pair_filter(ctx):
                        continue
                    tasks_omni.append(("A_BLOCK", a_id, block_cols, b_id, anchor_col))
    else:
        for b_id, (b_df, b_meta) in b_blocks.items():
            block_cols = list(b_df.columns)
            if not block_cols:
                continue
            for a_id, (a_df, a_meta) in a_blocks.items():
                for anchor_col in a_df.columns:
                    ctx = {
                        "analysis": "interaction",
                        "a_kind": a_kind, "b_kind": b_kind,
                        "a_id": a_id, "b_id": b_id,
                        "a_meta": a_meta, "b_meta": b_meta,
                        "anchor_col": anchor_col,
                        "block_cols": block_cols,
                    }
                    if pair_filter is not None and not pair_filter(ctx):
                        continue
                    tasks_omni.append(("B_BLOCK", a_id, anchor_col, b_id, block_cols))

    if verbose:
        print(f"omnibus tasks available to analyse (scheduled): {len(tasks_omni)}", flush=True)

    def work_omni(task) -> dict:
        tag = task[0]

        if tag == "A_BLOCK":
            _, a_id, block_cols, b_id, anchor_col = task
            a_df, a_meta = a_blocks[a_id]
            b_df, b_meta = b_blocks[b_id]

            geno = pd.concat([b_df[[anchor_col]], a_df[block_cols]], axis=1)
            abt, qc = make_model_table(geno, yser, covdf)

            row = {}
            row.update(qc)
            row.update({
                "A_kind": a_kind, "B_kind": b_kind,
                "AA_block_side": "A",
                "AA_variant": a_id,
                "Anchor_variant": b_id,
                "Anchor_col": anchor_col,
                "AA_AF_by_col_str": "",
                **{f"AA_{k}": v for k, v in a_meta.items() if k != "AAcount"},
                **{f"Anchor_{k}": v for k, v in b_meta.items() if k != "AAcount"},
            })

            if abt.shape[0] == 0:
                row.update({"LR_p": np.nan, "Anova_p": np.nan, "I_terms": np.nan, "I_block_coefs": ""})
                return row

            af_by = {c: float(abt[c].mean() / 2.0) for c in block_cols if pd.api.types.is_numeric_dtype(abt[c])}
            row["AA_AF_by_col_str"] = ";".join([f"{k}={v:.6g}" for k, v in af_by.items()]) if af_by else ""

            row.update(fit_block_omnibus_interaction(abt, anchor_col, block_cols, baseline_covars, config.model_type))
            return row

        # B_BLOCK
        _, a_id, anchor_col, b_id, block_cols = task
        a_df, a_meta = a_blocks[a_id]
        b_df, b_meta = b_blocks[b_id]

        geno = pd.concat([a_df[[anchor_col]], b_df[block_cols]], axis=1)
        abt, qc = make_model_table(geno, yser, covdf)

        row = {}
        row.update(qc)
        row.update({
            "A_kind": a_kind, "B_kind": b_kind,
            "AA_block_side": "B",
            "AA_variant": b_id,
            "Anchor_variant": a_id,
            "Anchor_col": anchor_col,
            "AA_AF_by_col_str": "",
            **{f"Anchor_{k}": v for k, v in a_meta.items() if k != "AAcount"},
            **{f"AA_{k}": v for k, v in b_meta.items() if k != "AAcount"},
        })

        if abt.shape[0] == 0:
            row.update({"LR_p": np.nan, "Anova_p": np.nan, "I_terms": np.nan, "I_block_coefs": ""})
            return row

        af_by = {c: float(abt[c].mean() / 2.0) for c in block_cols if pd.api.types.is_numeric_dtype(abt[c])}
        row["AA_AF_by_col_str"] = ";".join([f"{k}={v:.6g}" for k, v in af_by.items()]) if af_by else ""

        row.update(fit_block_omnibus_interaction(abt, anchor_col, block_cols, baseline_covars, config.model_type))
        return row

    prog = ProgressPrinter(total=len(tasks_omni), desc="interaction[omnibus]", use_tqdm=use_progress_bar, print_every=print_every) if verbose else None

    rows = []
    for r in parallel_imap(work_omni, tasks_omni, config.n_jobs, config.backend, config.chunksize):
        rows.append(r)
        if prog is not None:
            prog.update(1)

    if prog is not None:
        prog.close()

    out = pd.DataFrame(rows)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"interaction[omnibus] done in {format_seconds(elapsed)}", flush=True)
    out = out.dropna(axis=1, how="all")
    return out