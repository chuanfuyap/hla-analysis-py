"""
IO module for the meta-analysis component of hapy.meta.

Reads a txt file containing all the results files to be meta-analysed, each file is separated by a new line. 
"""
from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
import pandas as pd
from scipy import stats

_CI = 0.95
_Z_CI = stats.norm.ppf(1 - (1 - _CI) / 2)  # ~1.96, computed once

def _compute_se(study: pd.DataFrame) -> pd.DataFrame:
    """Derive SE from 95% CI and filter out non-positive SEs."""
    study = study.copy()
    study["SE"] = (study["CI_0.975"] - study["CI_0.025"]) / (2 * _Z_CI)
    return study[study["SE"] > 0]


def _compute_se_interaction(study: pd.DataFrame) -> pd.DataFrame:
    """Derive SE from 95% CI and filter out non-positive SEs."""
    study = study.copy()
    study["SE"] = (study["I_CI_0.975"] - study["I_CI_0.025"]) / (2 * _Z_CI)
    return study[study["SE"] > 0]

def _process_aa(study: pd.DataFrame, interaction=False) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    """Split AA results into univariate IVW and omnibus Stouffer inputs."""
    study = study.copy()
    if interaction:
        variant_col = "VARIANT" if "VARIANT" in study else "AA_VARIANT"
        partner_col = "Partner_col" if "Partner_col" in study else "Anchor_col"
        study["INTERACTION"] = study[variant_col].astype(str) + "*" + study[partner_col].astype(str)
        if "LR_p" not in study and "P-value" in study:
            if "Beta" in study:
                raise ValueError("AA interaction meta-analysis expects position-level omnibus results, not pairwise coefficients.")
            study["LR_p"] = study["P-value"]
    lr_p = study.get("LR_p", pd.Series(float("nan"), index=study.index))
    omnibus = study[lr_p.notnull()].copy()
    if interaction:
        return None, omnibus
    ivw = study[lr_p.isnull()].copy()
    if not ivw.empty:
        ivw = _compute_se(ivw)
    return ivw, omnibus


def _process_hla_snp(study: pd.DataFrame, interaction=False) -> Tuple[pd.DataFrame, None]:
    """HLA and SNP results: IVW only, derive SE from CI."""
    if interaction:
        study = study.copy()
        partner_col = "Partner_col" if "Partner_col" in study else "Anchor_col"
        if {"VARIANT", partner_col}.issubset(study.columns):
            identifier = study["VARIANT"].astype(str)
            if "Variant_col" in study:
                encoded = study["Variant_col"]
                identifier = identifier.where(encoded.isna(), identifier + ":" + encoded.fillna("").astype(str))
            study["INTERACTION"] = identifier + "*" + study[partner_col].astype(str)
            # Keep the existing meta engine compatible with both export schemas.
            for source, target in (("Beta", "I_coef"), ("StdErr", "I_StdErr"),
                                   ("CI-lower", "I_CI_0.025"), ("CI-upper", "I_CI_0.975")):
                if target not in study and source in study:
                    study[target] = study[source]
        else:
            study["INTERACTION"] = study['A_variant'] + "*" + study['B_col']
        return _compute_se_interaction(study), None
    else:
        return _compute_se(study), None

_PROCESSORS = {
    "AA":  _process_aa,
    "HLA": _process_hla_snp,
    "SNP": _process_hla_snp,
}

def read_input_files(
    file_loc: str,
    data_type: str,
    interaction = False, 
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Read per-cohort results files and prepare them for meta-analysis.

    Parameters
    ----------
    file_loc : str
        Path to a text file where each line is a path to a cohort results CSV.
    data_type : str
        One of "AA", "HLA", "SNP".

    Returns
    -------
    inv_var_studies : list[pd.DataFrame]
        Studies prepared for inverse-variance meta-analysis.
    stouffer_studies : list[pd.DataFrame]
        Studies prepared for Stouffer's meta-analysis (AA omnibus results only).
    """
    if data_type not in _PROCESSORS:
        raise ValueError(f"Unknown data_type {data_type!r}. Expected one of {list(_PROCESSORS)}.")

    process = _PROCESSORS[data_type]
    print(f"Processing {data_type} data")

    inv_var_studies, stouffer_studies = [], []

    seen_paths = set()
    with open(file_loc, "r", encoding="utf-8") as stream:
        for line in stream:
            path = line.strip()
            if not path:
                continue
            resolved = Path(path).resolve()
            if resolved in seen_paths:
                raise ValueError(f"Duplicate cohort file in input list: {path}")
            seen_paths.add(resolved)
            study = pd.read_csv(path)
            ivw, omnibus = process(study, interaction=interaction)
            if ivw is not None and not ivw.empty:
                inv_var_studies.append(ivw)
            if omnibus is not None and not omnibus.empty:
                stouffer_studies.append(omnibus)

    return inv_var_studies, stouffer_studies
