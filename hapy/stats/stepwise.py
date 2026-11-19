"""Forward conditional selection of amino-acid positions."""
from __future__ import annotations

from functools import partial
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd

from .adapters.aa import AAAdapter
from .preprocess import prepare_y, prepare_covar
from .runners.standard_runner import (
    run_standard, _normalize_condition_on, _validate_conditioning_variants,
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
             significance_threshold=5e-6, max_steps=20, residue=False,
             verbose=True, use_progress_bar=False, print_every=100) -> pd.DataFrame:
    """Perform stepwise conditional association analysis of AA positions.

    The most significant position is added to the conditioning set after
    each scan until no further positions meet the significance threshold.
    Uses the existing single and omnibus tests. Only selection="position"
    and residue=False are supported.

    Parameters follow :func:`hapy.stats.analyse`.
    ``max_steps`` limits the number of newly selected positions.

    Returns
    -------
    pandas.DataFrame
        Most significant position from each scan, including STEP and SELECTED.
        The final nonsignificant result is retained but not selected.
        Conditions, stopping reason, scans, runtime and diagnostics are in attrs.

    Notes
    -----
    Scans run sequentially, with parallel testing within each scan.
    To continue, pass attrs["condition_on"] to a new call.
    """
    started = time.perf_counter()
    if kind != "AA" or selection != "position":
        raise ValueError("Stepwise currently supports only kind='AA', selection='position'.")
    if residue:
        raise ValueError("Position-based stepwise analysis requires residue=False.")
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
        raise ValueError("Position-based stepwise condition_on cannot contain motif groups.")
    # Normalise once, preserving analyse's sample alignment and phenotype recoding.
    phenotype = prepare_y(famfile, y=y)
    covariates = prepare_covar(covar, phenotype.index)
    eligible = {v for v in AAAdapter.iter_variants(hladat) if not str(v).startswith("MOTIF_")}
    rows, failures = [], []
    scans = 0
    reason = "max_steps"
    if verbose:
        print(f"STARTING STEPWISE CONDITIONAL ANALYSIS - {selection} based", flush=True)
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
            _stepwise_scan=step, _stepwise_positions=remaining, verbose=verbose, use_progress_bar=use_progress_bar,
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
