"""Public API for inverse-variance and weighted Stouffer meta-analysis."""
from __future__ import annotations

import time

from ..stats.progress import format_seconds
from .io import read_input_files
from .combine import inv_var_meta_studies, stouffers_meta_studies


def meta_analyse(fileloc, data_type, beta="Uni_Coef", se="Uni_StdErr",
                 pval="LR_p", variant_col="VARIANT", prefix="META", interaction=False):
    """Return inverse-variance results and weighted Stouffer omnibus results, whichever is loaded.

    The second result combines independent-cohort omnibus p-values using sample
    size weights. A result is None when no applicable cohort tables exist.

    Parameters
    ----------
    file_loc : str
        Path to a text file where each line is a path to a cohort results CSV.
    data_type : str
        One of "AA", "HLA", "SNP".
    beta : str
        the effect size column
    se : str
        the standard error column
    pval : str
         pvalue columns, default = "LR_p"
    variant_col : str
        variant ID column
    prefix : str
        prefix to append when generating results
    interaction : boolean
        defining if interaction results were being meta-analysed

    """
    t0 = time.perf_counter()
    inv_var, omnibus = read_input_files(fileloc, data_type, interaction=interaction)
    fe_meta, stouffer_meta = None, None
    if inv_var:
        print("Performing Inverse-Variance Meta-analysis")
        fe_meta = inv_var_meta_studies(
            inv_var, beta="I_coef" if interaction else beta,
            se="I_StdErr" if interaction else se,
            variant_col="INTERACTION" if interaction else variant_col, prefix=prefix,
        )
    if omnibus:
        print("Performing Weighted Stouffer's Method Meta-analysis")
        stouffer_meta = stouffers_meta_studies(
            omnibus, pval=pval,
            variant_col="INTERACTION" if interaction else variant_col, prefix=prefix,
        )
    print(f"Meta-analysis completed in {format_seconds(time.perf_counter() - t0)}", flush=True)
    return fe_meta, stouffer_meta
