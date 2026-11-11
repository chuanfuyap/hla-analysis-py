"""Read current cohort outputs and route AA positions across all studies."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _require(study, columns):
    missing = [column for column in columns if column not in study]
    if missing:
        raise ValueError(f"Required input column(s) not found: {missing}")


def _sample_columns(study, n_col, case_col, control_col):
    # Map only the explicitly selected count columns. Do not discover legacy
    # counts or allow an unrelated count column to override the user's choice.
    counts = {target: study[source].copy()
              for source, target in ((n_col, "N_used"), (case_col, "N_case"), (control_col, "N_control"))
              if source is not None and source in study}
    study = study.drop(columns=["N_used", "N", "N_total", "N_case", "N_control"], errors="ignore").copy()
    for target, values in counts.items():
        study[target] = values
    return study


def _ivw_rows(study, beta, se):
    if study.empty:
        return study
    _require(study, [beta, se])
    study = study.copy()
    for column in (beta, se):
        study[column] = pd.to_numeric(study[column], errors="raise")
    study = study.dropna(subset=[beta, se])
    if (not np.isfinite(study[[beta, se]].to_numpy(dtype=float)).all()
            or study[se].le(0).any()):
        raise ValueError("IVW coefficients must be finite and standard errors positive and finite.")
    return study


def read_input_files(file_loc, data_type, interaction=False, *,
                     beta="Beta", se="StdErr", pval="P-value", variant_col="VARIANT",
                     test_col="TEST", n_col="N_used", case_col="N_case", control_col="N_control",
                     partner_col="Partner_col", feature_col="Variant_col", sep=","):
    """Return (IVW cohort tables, Stouffer cohort tables) using configured columns.

    file_loc lists one independent cohort file per nonblank line. Cohort files
    use sep (comma by default; use "\t" for TSV). Column arguments name input
    fields; counts are mapped internally to N_used/N_case/N_control. Missing
    optional count columns are allowed; None disables a count/feature field.

    Standard AA results require TEST (or test_col). OVR and unlabelled skipped
    rows are excluded. If any cohort labels a position MULTI, every SINGLE or
    MULTI row at that position is routed to Stouffer. Remaining SINGLE rows go
    to IVW. Missing statistics do not contribute. Duplicate position-level rows
    within a cohort raise an error. No legacy schema or SE-from-CI inference is
    performed. Caller-selected beta/se/pval/variant column names are preserved.

    AA interaction inputs remain position-level omnibus results, identified by
    variant and partner. HLA/SNP interaction coefficients use IVW; an optional
    encoded feature distinguishes multiple columns of the same variant.
    """
    if data_type not in {"AA", "HLA", "SNP"}:
        raise ValueError(f"Unknown data_type {data_type!r}. Expected AA, HLA or SNP.")
    print(f"Processing {data_type} data")
    studies = []
    seen_paths = set()
    with open(file_loc, "r", encoding="utf-8") as stream:
        paths = [line.strip() for line in stream if line.strip()]
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen_paths:
            raise ValueError(f"Duplicate cohort file in input list: {path}")
        seen_paths.add(resolved)
        study = pd.read_csv(path, sep=sep)
        _require(study, [variant_col])
        study = study.dropna(subset=[variant_col]).copy()
        if data_type == "AA" and not interaction:
            _require(study, [test_col])
            labels = study[test_col].astype("string").str.strip().str.upper()
            unknown = labels.notna() & ~labels.isin(["SINGLE", "MULTI", "OVR"])
            if unknown.any():
                raise ValueError(f"{path}: unknown test labels: {labels[unknown].unique().tolist()}")
            study = study.loc[labels.isin(["SINGLE", "MULTI"])].copy()
            study[test_col] = labels.loc[study.index]
        study = _sample_columns(study, n_col, case_col, control_col)
        identifier = variant_col
        if interaction and not study.empty:
            _require(study, [partner_col])
            study = study.dropna(subset=[partner_col]).copy()
            if data_type == "AA" and beta in study and study[beta].notna().any():
                raise ValueError("AA interaction meta-analysis expects position-level omnibus results, not pairwise coefficients.")
            variant = study[variant_col].astype(str)
            if data_type != "AA" and feature_col is not None and feature_col in study:
                encoded = study[feature_col]
                variant = variant.where(encoded.isna(), variant + ":" + encoded.fillna("").astype(str))
            study["INTERACTION"] = variant + "*" + study[partner_col].astype(str)
            identifier = "INTERACTION"
        if not study.empty and study[identifier].duplicated().any():
            raise ValueError(f"{path}: cohort contains duplicate variant IDs after excluding OVR/skipped rows.")
        studies.append(study)

    # Classify once using every cohort, before selecting rows with usable P-values.
    # A failed MULTI fit still identifies a multiallelic position conservatively.
    multi_positions = set()
    if data_type == "AA" and not interaction:
        for study in studies:
            multi_positions.update(study.loc[study[test_col].eq("MULTI"), variant_col])

    ivw_studies, stouffer_studies = [], []
    for study in studies:
        if study.empty:
            continue
        if data_type == "AA":
            mask = pd.Series(True, index=study.index) if interaction else study[variant_col].isin(multi_positions)
            omnibus, single = study.loc[mask].copy(), study.loc[~mask].copy()
            if not omnibus.empty:
                _require(omnibus, [pval])
                omnibus = omnibus.dropna(subset=[pval])
                if not omnibus.empty:
                    stouffer_studies.append(omnibus)
        else:
            single = study
        if not single.empty:
            single = _ivw_rows(single, beta, se)
            if not single.empty:
                ivw_studies.append(single)
    return ivw_studies, stouffer_studies
