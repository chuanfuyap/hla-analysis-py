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
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from .runners.standard_runner import (
    run_standard, _normalize_condition_on, _validate_conditioning_variants, _prepare_shared_null,
)
from .types import StandardConfig


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
             reference=None,
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
        Maximum number of newly selected positions.
    check_rank : bool, default False
        Whether to perform additional model-rank checks.

    Returns
    -------
    pandas.DataFrame
        Most significant position from each scan, including STEP, SELECTED,
        and the final nonsignificant result. Selected positions, stopping
        reason, runtime and other metadata are stored in ``attrs``.
        Motif results also report tested positions and motif group counts.

    Notes
    -----
    Scans run sequentially, with parallel testing within each scan.
    Motif selection is restricted to positions within the same HLA gene.
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
                        ("max_steps", max_steps), ("print_every", print_every)):
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
    if verbose:
        print(f"STARTING STEPWISE CONDITIONAL ANALYSIS - {selection} based", flush=True)
    arguments = (hladat, config, phenotype, covariates, eligible, conditions,
                 significance_threshold, max_steps, check_rank, variant_filter,
                 verbose, use_progress_bar, print_every, started)
    if selection == "position":
        out = _position_based(*arguments)
    else:
        out = _motif_based(*arguments, reference)
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
    return out


def _position_based(hladat, config, phenotype, covariates, eligible, conditions,
                    significance_threshold, max_steps, check_rank, variant_filter,
                    verbose, use_progress_bar, print_every, started):
    """Run the existing position-selection loop and assemble its winners table."""
    initial = list(conditions)
    rows, failures = [], []
    scans = 0
    reason = "max_steps"
    for step in range(1, max_steps + 1):
        remaining = eligible.difference(conditions)
        if not remaining:
            reason = "no_remaining_positions"
            break
        scan_started = time.perf_counter()
        if verbose:
            analysis = "Conditional analysis" if conditions else "Initial analysis"
            print(f"Step {step} — {analysis}", flush=True)
            if conditions:
                print(f"Conditioning on: {_conditioning_label(conditions)}", flush=True)
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
        conditions.append(winner["VARIANT"])
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


def _choose_motif_builder(hladat, reference, index):
    """Choose one established builder for the whole search, with per-run caches."""
    from copy import copy
    from ..data.hla_mapping import aa_motif_dosages
    from ..data.hla_dosage import hla_dosage_to_motifs
    from ..data.hla_reference import HLAReference, load_hla_reference

    mapping = getattr(hladat, "hla_mapping", None)
    if mapping is not None:
        mapping = copy(mapping)
        mapping.motif_cache = dict(mapping.motif_cache)
        return partial(aa_motif_dosages, mapping, output="dataframe",
                       simpleQC=False, verbose=False)
    if reference is None:
        raise ValueError("Motif selection requires a stored allele mapping or HLA dosages and reference.")
    reference = reference if isinstance(reference, HLAReference) else load_hla_reference(reference)
    reference = copy(reference)
    reference._resolved = dict(reference._resolved)
    reference._motif_rows = {}
    if not hasattr(hladat, "HLA") or hladat.type != "softcall":
        raise ValueError("Load the original softcall HLA block in hladat; "
                         "AA dosages alone cannot recover phase.")
    if not isinstance(hladat.HLA.data, pd.DataFrame):
        raise TypeError("hladat.HLA.data must be a DataFrame with alleles on rows and samples on columns.")
    dosage = hladat.HLA.data.copy()
    dosage.columns = dosage.columns.astype(str)
    if not dosage.columns.is_unique:
        raise ValueError("HLA dosage sample IDs must be unique.")
    dosage = dosage.reindex(columns=index)
    return partial(hla_dosage_to_motifs, dosage, reference, output="dataframe",
                   simpleQC=False, verbose=False)


def _build_motifs(builder, gene, positions, index):
    """Keep observed groups and their allele assignments, before reference dropping."""
    frame = builder(gene=gene, positions=positions).reindex(index)
    observed = frame.sum(axis=0, min_count=1).gt(0)
    return frame.loc[:, observed].copy()


def _expanded_motif_parents(old_motifs, new_motifs):
    """An expanded group must refine exactly one existing allele group."""
    old_groups = old_motifs.attrs["allele_groups"]
    parent_by_allele = {allele: motif for motif, alleles in old_groups.items() for allele in alleles}
    parents = {}
    for motif in new_motifs:
        alleles = new_motifs.attrs["allele_groups"][motif]
        assigned = {parent_by_allele.get(allele) for allele in alleles}
        if len(assigned) != 1 or None in assigned:
            return None
        parent = next(iter(assigned))
        if parent not in old_motifs:
            return None
        parents[motif] = parent
    return parents


def _motif_genotype(frame, prefix):
    """Drop the chosen reference and use safe formula names, preserving labels."""
    columns = [c for c in frame if c != frame.attrs["reference"]]
    names = {c: f"{prefix}_{i}" for i, c in enumerate(columns)}
    geno = frame[columns].rename(columns=names)
    return geno, {names[c]: c for c in columns}


def _motif_candidate(variant, *, builder, gene, positions, annotations, old_motifs,
                     y, covar, conditions, model_type, null_cache, check_rank, variant_filter):
    """Build and test one expansion; return its row and matrix for winner reuse."""
    from .models.standard import fit_motif_omnibus

    position = annotations[variant]["position"]
    new_motifs = _build_motifs(builder, gene, positions + [position], y.index)
    row = dict(annotations[variant]["meta"], VARIANT=variant, GENE=gene, AA_POS=position,
               COND_KIND="AA", COND_VARIANTS=";".join(conditions),
               N_COND_VARIANTS=len(conditions), TEST="MOTIF", **{"P-value": np.nan},
               MOTIF_POSITIONS=";".join(positions + [position]),
               N_REDUCED_MOTIFS=len(old_motifs.columns), N_EXPANDED_MOTIFS=len(new_motifs.columns))
    parents = _expanded_motif_parents(old_motifs, new_motifs)
    if parents is None:
        return dict(row, INFO="expanded motif groups do not refine the existing groups"), None
    children = [c for c, parent in parents.items() if parent == old_motifs.attrs["reference"]]
    if not children:
        return dict(row, INFO="no observed child of the reduced reference motif"), None
    # Keeping the reference in the same lineage preserves nesting with partial dosages.
    new_motifs.attrs["reference"] = new_motifs[children].sum().idxmax()
    reduced, _ = _motif_genotype(old_motifs, "MOTIF_R")
    expanded, labels = _motif_genotype(new_motifs, "MOTIF_F")
    ctx = dict(analysis="standard", kind="AA", variant_id=variant,
               meta=annotations[variant]["meta"], geno_cols=list(expanded))
    if variant_filter is not None and not variant_filter(ctx):
        return dict(row, INFO="excluded by variant filter"), None
    predictors = pd.concat([reduced, expanded], axis=1)
    if covar is not None and not predictors.columns.intersection(covar.columns).empty:
        raise ValueError("Covariate names collide with internal motif column names.")
    abt, qc = make_model_table(predictors, y, covar)
    row.update(qc)
    if abt.empty:
        return dict(row, INFO="no complete-case samples"), None
    # Verify dosages, not just labels: incomplete mappings may lose a parent contribution.
    for parent in old_motifs:
        children = [c for c, p in parents.items() if p == parent]
        recovered = new_motifs.loc[abt.index, children].sum(axis=1)
        if not np.allclose(recovered, old_motifs.loc[abt.index, parent], rtol=1e-8, atol=1e-8):
            return dict(row, INFO="expanded dosages do not reconstruct existing motif groups"), None
    try:
        row.update(fit_motif_omnibus(abt, list(reduced), list(expanded),
            list(covar) if covar is not None else [], model_type,
            null_cache=null_cache, check_rank=check_rank, term_labels=labels))
    except (PerfectSeparationError, np.linalg.LinAlgError, FloatingPointError,
            OverflowError, ZeroDivisionError) as exc:
        row["INFO"] = f"numerical fit failure: {type(exc).__name__}: {exc}"
    except ValueError as exc:
        if not any(message in str(exc).lower() for message in (
                "first guess on the deviance function returned a nan",
                "nan, inf or invalid value detected in weights")):
            raise
        row["INFO"] = f"numerical fit failure: ValueError: {exc}"
    row["Ref_AA"] = new_motifs.attrs["reference"]
    return row, new_motifs


def _valid_motif_result(row):
    p = row.get("P-value", np.nan)
    info = row.get("INFO")
    return np.isfinite(p) and 0 <= p <= 1 and (pd.isna(info) or info == "")


def _motif_based(hladat, config, phenotype, covariates, eligible, conditions,
                 significance_threshold, max_steps, check_rank, variant_filter,
                 verbose, use_progress_bar, print_every, started, reference):
    """Select a position, then test progressively expanded within-gene motifs."""
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
            use_progress_bar, print_every, started)
        rows = first.to_dict("records")
        failures = first.attrs["failed_tests"]
        scans = first.attrs["scans"]
        if first.empty or not first.SELECTED.iloc[0]:
            return first
        conditions.append(first.VARIANT.iloc[0])
    genes = {annotations[v]["gene"] for v in conditions}
    if len(genes) != 1:
        raise ValueError("Motif condition_on positions must belong to one HLA gene.")
    gene = next(iter(genes))
    positions = [annotations[v]["position"] for v in conditions]
    if len(set(positions)) != len(positions):
        raise ValueError("Motif condition_on contains duplicate AA positions.")
    old_motifs = None
    # Reconstruct the same reference lineage when continuing a previous search.
    # Only prefixes of the supplied conditions are built; no selection scans run.
    for length in range(1, len(positions) + 1):
        motifs = _build_motifs(builder, gene, positions[:length], phenotype.index)
        if len(motifs.columns) < 2:
            raise ValueError("Selected positions have insufficient motif variation.")
        if old_motifs is None:
            reference_motif = motifs.sum().idxmax()
        else:
            parents = _expanded_motif_parents(old_motifs, motifs)
            children = [] if parents is None else [c for c, parent in parents.items()
                if parent == old_motifs.attrs["reference"]]
            if not children:
                raise ValueError("Initial motif conditions do not define a consistent reference lineage.")
            reference_motif = motifs[children].sum().idxmax()
        motifs.attrs["reference"] = reference_motif
        old_motifs = motifs
    
    if rows:
        rows[0].update(MOTIF_POSITIONS=";".join(positions), N_REDUCED_MOTIFS=0,
                       N_EXPANDED_MOTIFS=len(old_motifs.columns))
    new_selections = len(conditions) - len(initial)
    reason = "max_steps"
    remaining = {v for v in eligible.difference(conditions) if annotations[v]["gene"] == gene
                 and annotations[v]["position"] not in positions}

    while new_selections < max_steps:
        if not remaining:
            reason = "no_remaining_positions"
            break
        step = scans + 1
        scan_started = time.perf_counter()
        if verbose:
            print(f"Step {step} — Conditional analysis", flush=True)
            print(f"Conditioning on: {_conditioning_label(conditions)}", flush=True)

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
        try:
            for row, new_motifs in parallel_imap_batched(worker, sorted(remaining), config.n_jobs,
                    config.backend, batch_size=config.batch_size, max_in_flight=config.chunksize):
                if progress is not None:
                    progress.update()
                if not _valid_motif_result(row):
                    failures.append(dict(row, STEP=step))
                    continue
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
        conditions.append(winner["VARIANT"])
        positions = positions + [annotations[winner["VARIANT"]]["position"]]

        old_motifs = winning_motifs
        new_selections += 1
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
    return out
