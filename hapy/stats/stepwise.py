"""Forward conditional selection of amino-acid positions."""
from __future__ import annotations

from functools import partial
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd

from .adapters.aa import AAAdapter
from .preprocess import prepare_y, prepare_covar, make_model_table
from .parallel import parallel_imap_batched
from .progress import ProgressPrinter
from .runners.standard_runner import (
    run_standard, _normalize_condition_on, _validate_conditioning_variants, _prepare_shared_null,
    _clean_output,
)
from .types import StandardConfig
from .motif import expand_motif
from ._stepwise_ld import _prepare_ld, _ld_scan_compare, _selected_variants
from ._motif import (_choose_motif_builder, _build_motifs,
                     _motif_genotype, _initial_motifs)


def _position_filter(ctx, *, eligible, user_filter):
    return ctx["variant_id"] in eligible and (user_filter is None or user_filter(ctx))


def _conditioning_label(conditions):
    """Group full AA position IDs into readable, gene-specific labels."""
    groups = {}
    for variant in conditions:
        fields = variant.split("_")
        if len(fields) < 4 or fields[0] != "AA":
            groups.setdefault(None, []).append(variant)
            continue
        groups.setdefault(fields[1], []).append(fields[2])
    labels = []
    for gene, positions in groups.items():
        if gene is None:
            labels.extend(positions)
        else:
            noun = "position" if len(positions) == 1 else "positions"
            labels.append(f"{gene} {noun} {', '.join(positions)}")
    return "; ".join(labels)


def stepwise(hladat, config: StandardConfig, *, kind="AA", selection="position",
             famfile=None, covar=None, y=None, variant_filter=None, condition_on=None,
             reference=None, max_genes=1, ld_check=False, ld_method="cca", ld_threshold=0.9,
             ld_criterion="bic", ld_conditional_threshold=0.01,
             significance_threshold=5e-6, max_steps=20, residue=False, check_rank: bool = False,
             verbose=True, use_progress_bar=False, print_every=100) -> pd.DataFrame:
    """Perform stepwise conditional analysis of AA positions or within-gene motifs.

    The most significant position is selected at each scan until no further
    positions meet the significance threshold. Position selection conditions
    on selected positions, while motif selection compares existing and
    expanded motif models.

    Parameters
    ----------
    Parameters not listed below follow :func:`hapy.stats.analyse`.

    selection : {"position", "motif", "haplotype"}, default "position"
        Selection method. "haplotype" is an alias for "motif".
    reference : HLAReference, optional
        Reference for building motifs from hladat.HLA.data when no retained
        allele mapping is available.
    significance_threshold : float, default 5e-6
        P-value threshold for selecting additional positions.
    max_steps : int, default 20
        Maximum number of newly selected positions. With LD checking, a group
        selected together counts as one step, even when it adds several positions.
    ld_check : bool, default False
        Compare significant winners against eligible positions in high LD.
    ld_method : {"cca", "cramer"}, default "cca"
        CCA permits cross-gene position comparisons. Cramer's V is within-gene.
        Motif comparisons remain within the active gene for either method.
    ld_threshold : float, default 0.9
        Minimum LD for comparing a partner against the current selection.
    ld_criterion : {"bic", "conditional"}, default "bic"
        Rule used to retain, replace or extend the LD winner.
    ld_conditional_threshold : float, default 0.01
        Significance threshold for reciprocal conditional tests, independent
        of significance_threshold for the final association model.
    max_genes : int, default 1
        Maximum number of genes in motif selection, including initial conditions.
        Completed gene motifs remain fixed while searching other genes.
    check_rank : bool, default False
        Whether to perform additional model-rank checks.

    Returns
    -------
    pandas.DataFrame
        Most significant position from each scan, including STEP, SELECTED,
        and the final nonsignificant result. Selected positions, stopping
        reason, runtime and other metadata are stored in ``attrs``.
        Motif results also report tested positions. LD decisions are stored in
        attrs["ld_comparisons"]. SELECTED_VARIANTS lists positions selected together.

    Notes
    -----
    Scans run sequentially, with parallel testing within each scan.
    Motifs are built within genes; max_genes allows successive gene searches.
    To continue, pass ``attrs["condition_on"]`` to a new call.
    """

    started = time.perf_counter()
    if kind != "AA" or selection not in {"position", "motif", "haplotype"}:
        raise ValueError("Stepwise supports kind='AA' and selection='position', 'motif' or 'haplotype'.")
    if selection == "haplotype":
        selection = "motif"
    if residue:
        raise ValueError("Stepwise analysis requires residue=False.")
    if not isinstance(config, StandardConfig):
        raise TypeError("config must be a StandardConfig.")
    if config.model_type not in {"linear", "logit"}:
        raise ValueError("model_type must be 'linear' or 'logit'.")
    if config.backend not in {"process", "thread"}:
        raise ValueError("backend must be 'process' or 'thread'.")
    for name, value in (("n_jobs", config.n_jobs), ("batch_size", config.batch_size),
                        ("max_steps", max_steps), ("max_genes", max_genes), ("print_every", print_every)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config.chunksize is not None and (isinstance(config.chunksize, bool)
            or not isinstance(config.chunksize, Integral) or config.chunksize < 1):
        raise ValueError("chunksize must be None or a positive integer.")
    if (isinstance(significance_threshold, bool) or not isinstance(significance_threshold, Real)
            or not np.isfinite(significance_threshold) or not 0 < significance_threshold <= 1):
        raise ValueError("significance_threshold must be finite and in (0, 1].")
    if not hasattr(hladat, "AA"):
        raise ValueError("AA not loaded. Please load AA data first.")
    conditions = _normalize_condition_on(condition_on)
    initial = list(conditions)
    _validate_conditioning_variants(AAAdapter, hladat, conditions)
    if any(v.startswith("MOTIF_") for v in conditions):
        raise ValueError("Stepwise condition_on requires AA position IDs, not motif group IDs.")
    # Normalise once, preserving analyse's sample alignment and phenotype recoding.
    phenotype = prepare_y(famfile, y=y)
    covariates = prepare_covar(covar, phenotype.index)
    eligible = {v for v in AAAdapter.iter_variants(hladat) if not str(v).startswith("MOTIF_")}
    ld_state = None
    if ld_check:
        from .models.standard import _validate_comparison_options
        _validate_comparison_options(ld_criterion, ld_conditional_threshold)
        if ld_method not in ("cca", "cramer"):
            raise ValueError("ld_method must be 'cca' or 'cramer'.")
        if (isinstance(ld_threshold, bool) or not isinstance(ld_threshold, Real)
                or not np.isfinite(ld_threshold) or not 0 <= ld_threshold <= 1):
            raise ValueError("ld_threshold must be finite and between 0 and 1.")
        ld_state = _prepare_ld(hladat, eligible, ld_method, ld_threshold, reference)
        ld_state.update(criterion=ld_criterion, conditional_threshold=ld_conditional_threshold)
    if verbose:
        print(f"STARTING STEPWISE CONDITIONAL ANALYSIS - {selection} based", flush=True)
    arguments = (hladat, config, phenotype, covariates, eligible, conditions,
                 significance_threshold, max_steps, check_rank, variant_filter,
                 verbose, use_progress_bar, print_every, started)
    if selection == "position":
        out = _position_based(*arguments, ld_state=ld_state)
    elif max_genes == 1:
        out = _motif_based(*arguments, reference, ld_state=ld_state)
    else:
        out = _multi_gene_motif(*arguments, reference, max_genes, ld_state=ld_state)
    reason = out.attrs["stop_reason"]
    scans = out.attrs["scans"]
    elapsed = out.attrs["elapsed_seconds"]
    conditions = out.attrs["condition_on"]
    if verbose:
        if reason == "no_remaining_positions":
            print("No remaining positions. Stopping.", flush=True)
        elif reason == "max_steps":
            print("Maximum selected positions reached. Stopping.", flush=True)
        selected = len(conditions) - len(initial)
        scan_noun = "scan" if scans == 1 else "scans"
        position_noun = "position" if selected == 1 else "positions"
        print("Analysis complete", flush=True)
        print(f"{scans} {scan_noun} | {selected} selected {position_noun} | Total: {elapsed:.1f}s", flush=True)
    if ld_state is not None:
        out.attrs["ld_comparisons"] = ld_state["history"]
    return _clean_stepwise_output(out)


def _position_based(hladat, config, phenotype, covariates, eligible, conditions,
                    significance_threshold, max_steps, check_rank, variant_filter,
                    verbose, use_progress_bar, print_every, started, *, step_offset=0,
                    fixed_conditions=(), ld_state=None, ld_mode="position"):
    """Run the existing position-selection loop and assemble its winners table."""
    initial = list(conditions)
    rows, failures = [], []
    scans = 0
    reason = "max_steps"
    for step in range(1 + step_offset, max_steps + 1 + step_offset):
        remaining = eligible.difference(conditions)
        if not remaining:
            reason = "no_remaining_positions"
            break
        scan_started = time.perf_counter()
        if verbose:
            analysis = "Conditional analysis" if conditions or fixed_conditions else "Initial analysis"
            print(f"Step {step} — {analysis}", flush=True)
            if conditions or fixed_conditions:
                print(f"Conditioning on: {_conditioning_label(list(fixed_conditions) + conditions)}", flush=True)
        table = run_standard(AAAdapter, hladat, config, y=phenotype, covar=covariates,
            condition_on=conditions, residue=False,
            variant_filter=partial(_position_filter, eligible=remaining, user_filter=variant_filter),
            _stepwise_scan=step, _stepwise_positions=remaining,
            _stepwise_check_rank=check_rank, verbose=verbose, use_progress_bar=use_progress_bar,
            print_every=print_every)
        scans += 1
        p = pd.to_numeric(table.get("P-value", pd.Series(np.nan, index=table.index)), errors="coerce")
        info = table.get("INFO", pd.Series("", index=table.index)).fillna("")
        tests = table.get("TEST", pd.Series("", index=table.index))
        valid = np.isfinite(p) & p.between(0, 1) & info.eq("") & tests.isin(["SINGLE", "MULTI"])
        for record in table.loc[~valid].to_dict("records"):
            failures.append(dict(record, STEP=step))
        if not valid.any():
            reason = "no_valid_tests"
            if verbose:
                print("No valid tests. Stopping.", flush=True)
                print(f"Elapsed: {time.perf_counter() - scan_started:.1f}s\n", flush=True)
            break
        candidates = table.loc[valid].copy()
        candidates["P-value"] = p.loc[valid]
        winner = candidates.sort_values(["P-value", "VARIANT"], kind="stable").iloc[0].to_dict()
        if ld_state is not None:
            winner, _ = _ld_scan_compare(hladat, config, winner, state=ld_state,
                eligible=set(candidates["VARIANT"]), conditions=conditions, y=phenotype,
                covar=covariates, significance_threshold=significance_threshold,
                step=step, mode=ld_mode, check_rank=check_rank,
                variant_filter=variant_filter, fixed_conditions=fixed_conditions, verbose=verbose)
        accepted = bool(winner["P-value"] < significance_threshold)
        winner.update(STEP=step, SELECTED=accepted)
        rows.append(winner)
        if verbose:
            if accepted:
                print(f"Most significant: {winner['VARIANT']}", flush=True)
                print(f"P-value: {winner['P-value']:.2e}", flush=True)
            else:
                print(f"Best remaining P-value: {winner['P-value']:.2e}", flush=True)
                print("Threshold not met. Stopping.", flush=True)
            print(f"Elapsed: {time.perf_counter() - scan_started:.1f}s\n", flush=True)
        if not accepted:
            reason = "threshold_not_met"
            break
        conditions.extend(_selected_variants(winner))
    remaining = eligible.difference(conditions)
    if reason == "max_steps" and not remaining:
        reason = "no_remaining_positions"
    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "VARIANT", "P-value", "TEST", "COND_VARIANTS", "STEP", "SELECTED"])
    out["SELECTED"] = out["SELECTED"].astype(bool)
    out["STEP"] = out["STEP"].astype(int)
    elapsed = time.perf_counter() - started
    out.attrs.update(stop_reason=reason, condition_on=list(conditions), initial_condition_on=initial,
                     scans=scans, elapsed_seconds=elapsed,
                     significance_threshold=significance_threshold, failed_tests=failures)
    return out


def _motif_candidate(variant, *, builder, gene, positions, annotations, old_motifs,
                     y, covar, conditions, model_type, null_cache, check_rank, variant_filter):
    """Build and test one expansion; return its row and matrix for winner reuse."""
    prepared = dict(builder=builder, gene=gene, positions=positions,
                    position=annotations[variant]["position"],
                    meta=annotations[variant]["meta"], old_motifs=old_motifs,
                    null_cache=null_cache, variant_filter=variant_filter)
    row, motifs = expand_motif(
        None, StandardConfig(model_type), conditions, variant,
        y=y, covar=covar, check_rank=check_rank, _prepared=prepared,
    )
    return {**annotations[variant]["meta"], **row}, motifs


def _valid_motif_result(row):
    p = row.get("P-value", np.nan)
    info = row.get("INFO")
    return np.isfinite(p) and 0 <= p <= 1 and (pd.isna(info) or info == "")


def _motif_based(hladat, config, phenotype, covariates, eligible, conditions,
                 significance_threshold, max_steps, check_rank, variant_filter,
                 verbose, use_progress_bar, print_every, started, reference, *,
                 return_motifs=False, step_offset=0, fixed_conditions=(), builder=None, ld_state=None):
    """Select a position, then test progressively expanded within-gene motifs."""
    if builder is None:
        builder = _choose_motif_builder(hladat, reference, phenotype.index)
    initial = list(conditions)
    annotations = {}
    for variant, group in hladat.AA.info.groupby("AA_ID", sort=False):
        if variant in eligible:
            meta = group.iloc[0].to_dict()
            annotations[variant] = dict(gene=str(meta["GENE"]), position=str(meta["AA_POS"]), meta=meta)
    rows, failures, scans = [], [], 0
    if not conditions:
        first = _position_based(hladat, config, phenotype, covariates, eligible, [],
            significance_threshold, 1, check_rank, variant_filter, verbose,
            use_progress_bar, print_every, started, ld_state=ld_state, ld_mode="initial")
        rows = first.to_dict("records")
        failures = first.attrs["failed_tests"]
        scans = first.attrs["scans"]
        if first.empty or not first.SELECTED.iloc[0]:
            return (first, None) if return_motifs else first
        conditions.extend(first.attrs["condition_on"])
    genes = {annotations[v]["gene"] for v in conditions}
    if len(genes) != 1:
        raise ValueError("Motif condition_on positions must belong to one HLA gene.")
    gene = next(iter(genes))
    positions = [annotations[v]["position"] for v in conditions]
    if len(set(positions)) != len(positions):
        raise ValueError("Motif condition_on contains duplicate AA positions.")
    old_motifs = _initial_motifs(builder, gene, positions, phenotype.index)

    if rows:
        rows[0].update(MOTIF_POSITIONS=";".join(positions), N_REDUCED_MOTIFS=0,
                       N_EXPANDED_MOTIFS=len(old_motifs.columns))
    initial_motif_groups = len(old_motifs.columns)
    new_selections = len(rows) if ld_state is not None else len(conditions) - len(initial)
    reason = "max_steps"
    remaining = {v for v in eligible.difference(conditions) if annotations[v]["gene"] == gene
                 and annotations[v]["position"] not in positions}

    first_step = step_offset + scans + 1
    for step in range(first_step, first_step + max_steps - new_selections):
        if not remaining:
            reason = "no_remaining_positions"
            break
        scan_started = time.perf_counter()
        if verbose:
            print(f"Step {step} — Conditional analysis", flush=True)
            print(f"Conditioning on: {_conditioning_label(list(fixed_conditions) + conditions)}", flush=True)

        reduced, _ = _motif_genotype(old_motifs, "MOTIF_R")
        baseline, _ = make_model_table(reduced, phenotype, covariates)
        reduced_cols = list(covariates) if covariates is not None else []
        reduced_cols += list(reduced)
        reduced_cache = _prepare_shared_null(baseline.index, phenotype,
            baseline[reduced_cols], reduced_cols, config.model_type)

        worker = partial(_motif_candidate, builder=builder, gene=gene, positions=positions,
            annotations=annotations, old_motifs=old_motifs, y=phenotype, covar=covariates,
            conditions=conditions, model_type=config.model_type, null_cache=reduced_cache,
            check_rank=check_rank, variant_filter=variant_filter)

        progress = ProgressPrinter(total=len(remaining), desc=f"Step {step}",
            use_tqdm=use_progress_bar, print_every=print_every) if verbose else None
        best = None
        valid_candidates = set()
        try:
            for row, new_motifs in parallel_imap_batched(worker, sorted(remaining), config.n_jobs,
                    config.backend, batch_size=config.batch_size, max_in_flight=config.chunksize):
                if progress is not None:
                    progress.update()
                if not _valid_motif_result(row):
                    failures.append(dict(row, STEP=step))
                    continue
                valid_candidates.add(row["VARIANT"])
                if best is None or (row["P-value"], row["VARIANT"]) < (best[0]["P-value"], best[0]["VARIANT"]):
                    best = row, new_motifs
        finally:
            if progress is not None:
                progress.close()
        scans += 1
        if best is None:
            reason = "no_valid_tests"
            if verbose:
                print("No valid tests. Stopping.", flush=True)
                print(f"Elapsed: {time.perf_counter() - scan_started:.1f}s\n", flush=True)
            break
        winner, winning_motifs = best
        if ld_state is not None:
            winner, winning_motifs = _ld_scan_compare(hladat, config, winner,
                state=ld_state, eligible=valid_candidates, conditions=conditions,
                y=phenotype, covar=covariates, significance_threshold=significance_threshold,
                step=step, mode="motif", builder=builder, old_motifs=old_motifs,
                winning_motifs=winning_motifs, check_rank=check_rank,
                variant_filter=variant_filter, fixed_conditions=fixed_conditions, verbose=verbose)

        accepted = bool(winner["P-value"] < significance_threshold)
        winner.update(STEP=step, SELECTED=accepted)
        rows.append(winner)
        if verbose:
            if accepted:
                print(f"Most significant: {winner['VARIANT']}", flush=True)
                print(f"P-value: {winner['P-value']:.2e}", flush=True)
            else:
                print(f"Best remaining P-value: {winner['P-value']:.2e}", flush=True)
                print("Threshold not met. Stopping.", flush=True)
            print(f"Elapsed: {time.perf_counter() - scan_started:.1f}s\n", flush=True)
        if not accepted:
            reason = "threshold_not_met"
            break
        selected = _selected_variants(winner)
        conditions.extend(selected)
        positions += [annotations[v]["position"] for v in selected]

        old_motifs = winning_motifs
        remaining = {v for v in remaining.difference(conditions) if annotations[v]["position"] not in positions}
    if reason == "max_steps" and not remaining:
        reason = "no_remaining_positions"
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["VARIANT", "P-value", "TEST", "COND_VARIANTS", "STEP", "SELECTED"])
    out["SELECTED"] = out["SELECTED"].astype(bool)
    out["STEP"] = out["STEP"].astype(int)
    out.attrs.update(stop_reason=reason, condition_on=list(conditions), initial_condition_on=initial,
        scans=scans, elapsed_seconds=time.perf_counter() - started,
        significance_threshold=significance_threshold, failed_tests=failures,
        gene=gene, selected_positions=positions)
    if return_motifs:
        out.attrs["initial_motif_groups"] = initial_motif_groups
        return out, old_motifs
    return out


def _multi_gene_motif(hladat, config, phenotype, covariates, eligible, conditions,
                      significance_threshold, max_steps, check_rank, variant_filter,
                      verbose, use_progress_bar, print_every, started, reference, max_genes, *, ld_state=None):
    """Finish each gene's motif search before scanning the remaining genes."""
    builder = _choose_motif_builder(hladat, reference, phenotype.index)
    annotations = {variant: group.iloc[0] for variant, group in hladat.AA.info.groupby("AA_ID")
                   if variant in eligible}
    initial = list(conditions)
    by_gene = {}
    for variant in initial:
        by_gene.setdefault(str(annotations[variant]["GENE"]), []).append(variant)
    if len(by_gene) > max_genes:
        raise ValueError("max_genes is smaller than the number of initial genes.")
    adjustment = covariates
    completed, selected = [], []
    rows, failures = [], []
    gene_positions, gene_stops = {}, {}
    scans, selections_left = 0, max_steps

    def record(table, fixed):
        # Local scans already use global step numbers. Only scalar records persist.
        for row in table.to_dict("records"):
            local = row.get("COND_VARIANTS")
            local = [] if pd.isna(local) or not local else local.split(";")
            all_conditions = list(fixed) + local
            row.update(COND_VARIANTS=";".join(all_conditions) if all_conditions else np.nan,
                       COND_KIND="AA" if all_conditions else np.nan,
                       N_COND_VARIANTS=len(all_conditions))
            rows.append(row)
        for failure in table.attrs["failed_tests"]:
            failure = dict(failure)
            local = failure.get("COND_VARIANTS")
            local = [] if pd.isna(local) or not local else local.split(";")
            all_conditions = list(fixed) + local
            failure.update(COND_VARIANTS=";".join(all_conditions) if all_conditions else np.nan,
                           COND_KIND="AA" if all_conditions else np.nan,
                           N_COND_VARIANTS=len(all_conditions))
            failures.append(failure)

    def freeze(gene, motifs):
        nonlocal adjustment
        block, _ = _motif_genotype(motifs, f"FIXED_{gene}")
        # Avoid overwriting a user covariate with an internal motif column.
        while adjustment is not None and set(block).intersection(adjustment.columns):
            block = block.add_prefix("_")
        adjustment = block if adjustment is None else pd.concat([adjustment, block], axis=1)
        completed.append(gene)

    # Earlier supplied genes are fixed; the last supplied gene resumes searching.
    active_gene = next(reversed(by_gene)) if by_gene else None
    for gene, initial_positions in by_gene.items():
        if gene == active_gene:
            break
        table, motifs = _motif_based(hladat, config, phenotype, adjustment, eligible,
            list(initial_positions), significance_threshold, 0, check_rank, variant_filter,
            False, use_progress_bar, print_every, started, reference,
            return_motifs=True, builder=builder, ld_state=ld_state)
        gene_positions[gene] = table.attrs["selected_positions"]
        gene_stops[gene] = "initial_fixed"
        selected.extend(initial_positions)
        freeze(gene, motifs)

    reason = "max_genes"
    for _ in range(max_genes - len(completed)):
        if selections_left <= 0:
            break
        fixed = list(selected)
        new_gene = active_gene is None
        if active_gene is not None:
            initial_positions = list(by_gene[active_gene])
        else:
            if len(completed) >= max_genes:
                reason = "max_genes"
                break
            remaining = {v for v in eligible if str(annotations[v]["GENE"]) not in completed}
            if verbose and completed:
                print("SCANNING POSITIONS OUTSIDE COMPLETED GENE(S).", flush=True)
                print("---------------------------------------------", flush=True)
                print(flush=True)
            initial_result = _position_based(hladat, config, phenotype, adjustment, remaining, [],
                significance_threshold, 1, check_rank, variant_filter, verbose,
                use_progress_bar, print_every, started, step_offset=scans,
                fixed_conditions=fixed, ld_state=ld_state, ld_mode="initial")
            record(initial_result, fixed)
            scans += initial_result.attrs["scans"]
            if initial_result.empty or not initial_result.SELECTED.iloc[0]:
                reason = initial_result.attrs["stop_reason"]
                break
            initial_positions = list(initial_result.attrs["condition_on"])
            active_gene = str(annotations[initial_positions[0]]["GENE"])
            selections_left -= 1

        table, motifs = _motif_based(hladat, config, phenotype, adjustment, eligible,
            initial_positions, significance_threshold, selections_left, check_rank, variant_filter,
            verbose, use_progress_bar, print_every, started, reference,
            return_motifs=True, step_offset=scans, fixed_conditions=fixed, builder=builder, ld_state=ld_state)
        if new_gene:
            # The initial position scan is also the first model of this gene.
            rows[-1].update(MOTIF_POSITIONS=";".join(str(annotations[v]["AA_POS"]) for v in initial_positions),
                            N_REDUCED_MOTIFS=0,
                            N_EXPANDED_MOTIFS=table.attrs["initial_motif_groups"])
        record(table, fixed)
        scans += table.attrs["scans"]
        accepted = int(table.SELECTED.sum())
        selections_left -= accepted
        selected.extend(table.attrs["condition_on"])
        gene_positions[active_gene] = table.attrs["selected_positions"]
        gene_stops[active_gene] = table.attrs["stop_reason"]
        freeze(active_gene, motifs)
        active_gene = None
        if selections_left == 0:
            reason = "max_steps"
            break

    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "VARIANT", "P-value", "TEST", "COND_VARIANTS", "STEP", "SELECTED"])
    out["SELECTED"] = out["SELECTED"].astype(bool)
    out["STEP"] = out["STEP"].astype(int)
    out.attrs.update(stop_reason=reason, condition_on=selected, initial_condition_on=initial,
        scans=scans, elapsed_seconds=time.perf_counter() - started,
        significance_threshold=significance_threshold, failed_tests=failures,
        selected_positions=gene_positions, gene_stop_reasons=gene_stops,
        genes=completed, max_genes=max_genes)
    return out


def _clean_stepwise_output(dataframe):
    """Keep stepwise results concise, with selection fields first and counts last."""
    df = dataframe.drop(columns=[
        "COND_KIND", "N_COND_VARIANTS", "AA_ID", "POS", "alleleA", "alleleB",
        "N_missing_Y", "N_dropped_other",
        "N_REDUCED_MOTIFS", "N_EXPANDED_MOTIFS",
    ], errors="ignore")
    if df.empty:
        return df

    # Zero counts add no information; zero statistics and selection flags still do.
    counts = ["N_case", "N_control"]
    df = df.drop(columns=[c for c in counts if c in df and df[c].eq(0).all()])
    if "COND_VARIANTS" in df and df["COND_VARIANTS"].fillna("").eq("").all():
        df = df.drop(columns="COND_VARIANTS")
    df = _clean_output(df)

    first = [
         "VARIANT", "GENE", "AA_POS", "MOTIF_POSITIONS", "Amino_Acids", "Ref_AA", "AAcount", "AA_AF",
        "P-value", "Statistic", "Beta", "StdErr", "CI-lower", "CI-upper", "DoF","TEST","SELECTED", 
        "COND_VARIANTS"
    ]
    last = ["N_used", "N_case", "N_control", "INFO"]
    first = [c for c in first if c in df]
    last = [c for c in last if c in df]
    middle = [c for c in df if c not in first + last]
    result = df[first + middle + last]
    result.attrs = dataframe.attrs.copy()
    return result.set_index("STEP")
