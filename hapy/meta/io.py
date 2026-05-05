"""
IO module for the meta-analysis component of hapy.meta.

Reads a txt file containing all the results files to be meta-analysed, each file is separated by a new line. 
"""
from __future__ import annotations

from typing import List, Tuple
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

def _process_aa(study: pd.DataFrame, interaction=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split AA results into IVW (univariate, LR_p null) and
    Stouffer's (omnibus, LR_p not null) subsets.
    """

    if interaction:
        study["INTERACTION"] = study['AA_VARIANT']+"*" +study['Anchor_col']

    
    if interaction:
        ivw=None
    else:
        ivw = _compute_se(study[study["LR_p"].isnull()])
    stouffer = study[study["LR_p"].notnull()]



    return ivw, stouffer

def _process_hla_snp(study: pd.DataFrame, interaction=False) -> Tuple[pd.DataFrame, None]:
    """HLA and SNP results: IVW only, derive SE from CI."""
    if interaction:
        study["INTERACTION"] = study['A_variant']+"*" +study['B_col']
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
        Path to a text file where each line is a path to a cohort results TSV.
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

    if interaction:
        with open(file_loc, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                study = pd.read_csv(line.strip())
                ivw, stouffer = process(study, interaction=True)

                inv_var_studies.append(ivw)
                if stouffer is not None and not stouffer.empty:
                    stouffer_studies.append(stouffer)
    else:
        with open(file_loc, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                study = pd.read_csv(line.strip())
                ivw, stouffer = process(study)

                inv_var_studies.append(ivw)
                if stouffer is not None and not stouffer.empty:
                    stouffer_studies.append(stouffer)

    return inv_var_studies, stouffer_studies