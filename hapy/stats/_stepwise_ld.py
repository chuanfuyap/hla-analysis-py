"""LD checks between a stepwise winner and alternative positions."""
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .ld import LD
from .compare import compare_positions, compare_positions_motif
from .adapters.aa import AAAdapter
from .preprocess import make_model_table
from .runners.standard_runner import _prepare_conditions, run_standard
from .models.standard import fit_omnibus
from .motif import analyse_motif, expand_motif
from ._motif import _choose_motif_builder, _initial_motifs


def _prepare_ld(hladat, eligible, method, threshold, reference):
    """Calculate LD once and map its position labels to AA variant IDs."""
    annotations = {}
    for variant, rows in hladat.AA.info.groupby("AA_ID", sort=False):
        if variant in eligible:
            labels = rows[["GENE", "AA_POS"]].drop_duplicates()
            if len(labels) != 1 or labels.isna().any().any():
                raise ValueError(f"Ambiguous AA position: {variant}")
            annotations[variant] = tuple(labels.iloc[0].astype(str))
    genes = list(dict.fromkeys(g for g, _ in annotations.values()))
    matrix = pd.DataFrame(np.nan, index=list(annotations), columns=list(annotations))
    if genes:
        # CCA can span genes. Cramer's V needs a separate within-gene matrix.
        groups = [genes] if method == "cca" else [[g] for g in genes]
        for group in groups:
            gene = group[0] if len(group) == 1 else group
            values = LD(hladat, gene=gene, method=method, reference=reference)
            def label(variant):
                g, pos = annotations[variant]
                pos = pos if "i" in pos else int(pos)
                return (g, pos) if len(group) > 1 else pos
            variants = [v for v in annotations if annotations[v][0] in group and label(v) in values.index]
            if variants:
                matrix.loc[variants, variants] = values.loc[
                    [label(v) for v in variants], [label(v) for v in variants]].to_numpy()
    return dict(matrix=matrix, annotations=annotations, threshold=threshold,
                reference=reference, history=[])


def _selected_variants(row):
    """One scan may select more than one position after an LD comparison."""
    return row.get("SELECTED_VARIANTS", row["VARIANT"]).split(";")


def _position_model(hladat, config, selected, conditions, y, covar):
    """Test the final position group against the pre-step conditioning model."""
    if len(selected) == 1:
        result = run_standard(AAAdapter, hladat, config, y=y, covar=covar,
            condition_on=conditions, variant_filter=lambda ctx: ctx["variant_id"] == selected[0],
            verbose=False)
        if result.empty:
            raise ValueError("No usable final position model.")
        return result.iloc[0].to_dict()
    adjustment, covar_cols = _prepare_conditions(AAAdapter, hladat, y.index,
                                                 covar.copy() if covar is not None else None,
                                                 conditions)
    blocks, labels = [], {}
    for i, variant in enumerate(selected):
        geno, _ = AAAdapter.build_geno(hladat, variant, y.index)
        geno = geno[list(geno.attrs.get("test_columns", geno.columns))].copy()
        if geno.empty:
            raise ValueError(f"{variant}: no eligible residue predictors")
        names = {c: f"LD_{i}_{j}" for j, c in enumerate(geno)}
        labels.update({names[c]: f"{variant}:{c}" for c in geno})
        blocks.append(geno.rename(columns=names))
    predictors = pd.concat(blocks, axis=1)
    if adjustment is not None and not predictors.columns.intersection(adjustment.columns).empty:
        raise ValueError("Covariates conflict with internal LD predictor columns.")
    abt, qc = make_model_table(predictors, y, adjustment)
    if abt.empty:
        raise ValueError("No complete-case samples.")
    result = fit_omnibus(abt, list(predictors), covar_cols, config.model_type, term_labels=labels)
    return dict(qc, **result)


def _final_model(hladat, config, selected, conditions, y, covar, mode,
                 reference, builder, old_motifs, annotations, check_rank, variant_filter):
    """Refit the proposed selection; do not report the original winner's statistics."""
    if mode == "position":
        return _position_model(hladat, config, selected, conditions, y, covar), None
    gene = annotations[selected[0]][0]
    all_variants = list(conditions) + selected
    if mode == "initial" and len(selected) == 1:
        return _position_model(hladat, config, selected, conditions, y, covar), None
    if mode == "initial":
        result = analyse_motif(hladat, config, all_variants, y=y, covar=covar,
                               reference=reference, check_rank=check_rank).iloc[0].to_dict()
        builder = builder or _choose_motif_builder(hladat, reference, y.index)
        motifs = _initial_motifs(builder, gene, [annotations[v][1] for v in all_variants], y.index)
        return result, motifs
    # Use the same reference choices that continuation will reconstruct.
    final_motifs = _initial_motifs(builder, gene,
        [annotations[v][1] for v in all_variants], y.index)
    prepared = dict(builder=builder, gene=gene, reference_motif=final_motifs.attrs["reference"],
        positions=[annotations[v][1] for v in all_variants[:-1]],
        position=annotations[all_variants[-1]][1], meta={}, old_motifs=old_motifs,
        null_cache=None, variant_filter=variant_filter)
    result, motifs = expand_motif(None, config, all_variants[:-1], all_variants[-1],
        y=y, covar=covar, check_rank=check_rank, _prepared=prepared)
    result.update(COND_VARIANTS=";".join(conditions), N_COND_VARIANTS=len(conditions))
    return result, motifs


def _ld_scan_compare(hladat, config, winner, *, state, eligible, conditions,
                     y, covar, significance_threshold, step,
                     mode="position", builder=None, old_motifs=None,
                     winning_motifs=None, check_rank=False, variant_filter=None,
                     fixed_conditions=(), verbose=False):
    """Compare LD partners, updating the candidate group after each decision."""
    if winner["P-value"] >= significance_threshold:
        return winner, winning_motifs
    matrix, annotations = state["matrix"], state["annotations"]
    selected = [winner["VARIANT"]]
    current_winner = selected[0]
    excluded = set(conditions) | set(fixed_conditions) | set(selected)
    final_row, final_motifs = dict(winner), winning_motifs
    errors = (ValueError, np.linalg.LinAlgError, PerfectSeparationError,
              FloatingPointError, OverflowError, ZeroDivisionError)
    # Use the original winner's partners once; later decisions only update the models.
    partners = []
    for variant in eligible.difference(excluded):
        if mode != "position" and annotations[variant][0] != annotations[current_winner][0]:
            continue
        ld = matrix.loc[variant, current_winner]
        if np.isfinite(ld) and ld >= state["threshold"]:
            partners.append((float(ld), variant))
    partners.sort(key=lambda item: (-item[0], item[1]))

    for ld, partner in partners:
        extras = [v for v in selected if v != current_winner]
        shared = list(conditions) + extras
        history = dict(STEP=step, SELECTION=mode, A=current_winner, B=partner, LD=ld,
                       POSITIONS_BEFORE=";".join(selected), ACTION="retain")
        try:
            # Completed-gene motifs are already in covar. Do not append them twice.
            if mode in ("position", "initial"):
                adjustment, _ = _prepare_conditions(AAAdapter, hladat, y.index,
                    covar.copy() if covar is not None else None, shared)
                comparison = compare_positions(hladat, config, current_winner, partner,
                                                y=y, covar=adjustment)
            else:
                comparison = compare_positions_motif(hladat, config,
                    shared + [current_winner], shared + [partner], y=y, covar=covar,
                    reference=state["reference"])
            history.update(comparison.iloc[0].to_dict())
            recommendation = comparison.iloc[0]["RECOMMENDATION"]
            proposed = list(selected)
            if recommendation == "B":
                proposed = [partner if v == current_winner else v for v in selected]
            elif recommendation == "A+B":
                proposed.append(partner)
            if proposed != selected:
                result, motifs = _final_model(hladat, config, proposed, conditions,
                    y, covar, mode, state["reference"], builder, old_motifs,
                    annotations, check_rank, variant_filter)
                info = result.get("INFO")
                valid = (np.isfinite(result.get("P-value", np.nan)) and
                         result["P-value"] < significance_threshold and
                         (pd.isna(info) or info == "" or info is None))
                if valid:
                    # Replace stale statistics and residue summaries from the first winner.
                    final_row = {k: v for k, v in final_row.items() if k not in (
                        "P-value", "Statistic", "DoF", "Beta", "StdErr", "CI-lower", "CI-upper",
                        "Amino_Acids", "AA_AF", "AAcount", "Ref_AA", "INFO")}
                    final_row.update(result)
                    selected = proposed
                    if recommendation == "B":
                        current_winner = partner
                    final_row.update(VARIANT=current_winner, SELECTED_VARIANTS=";".join(selected),
                        GENE=";".join(dict.fromkeys(annotations[v][0] for v in selected)),
                        AA_POS=";".join(annotations[v][1] for v in selected))
                    if mode != "position":
                        final_row["MOTIF_POSITIONS"] = "-".join(annotations[v][1] for v in conditions + selected)
                    final_motifs = motifs
                    history["ACTION"] = "replace" if recommendation == "B" else "extend"
                    if verbose:
                        old_label = " ".join(annotations[history["A"]])
                        new_label = " ".join(annotations[partner])
                        if recommendation == "B":
                            action = f"{new_label} (BIC: {history['BIC_B']}) replaces {old_label} (BIC: {history['BIC_A']})"
                            delta = -history["Delta_BIC_B_minus_A"]
                        else:
                            action = f"retaining {old_label} and adding {new_label}"
                            delta = history["Delta_BIC_joint_improvement"]
                        print(flush=True)
                        print(f"LD COMPARISON: {action} ", flush=True)
                        print(f"LD: {ld:.3f} | BIC improvement: {delta:.2f} | P-value: {result['P-value']:.2e}", flush=True)
                        print(flush=True)
                else:
                    history.update(ACTION="skip", REASON="final model failed diagnostics or significance threshold",
                                   FINAL_P_VALUE=result.get("P-value"), FINAL_INFO=info)
        except errors as exc:
            # A failed alternative must not discard the valid original selection.
            history.update(ACTION="skip", REASON=str(exc))
        history["POSITIONS_AFTER"] = ";".join(selected)
        state["history"].append(history)
    return final_row, final_motifs
