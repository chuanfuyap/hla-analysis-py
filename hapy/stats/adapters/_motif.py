"""Prepare explicit within-gene motif dosage categories for the AA analysis API."""
import numpy as np
import pandas as pd


def build_motif_geno(aadf, info, variant_id, retain_reference=False):
    """Prepare motif genotypes and metadata for analysis.

    Parameters
    ----------
    aadf : pandas.DataFrame
        Motif rows and sample columns, with an AA_ID column.
    info : pandas.DataFrame
        Matching motif annotations, including MOTIF, GENE and AA_POS.
    variant_id : str
        Motif group to analyse.
    retain_reference : bool, default False
        Include the reference dosage alongside the model columns.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Sample-by-genotype matrix and analysis metadata.

    Notes
    -----
    Applies frequency QC. Insertion groups use absence as reference; other
    groups use the most frequent retained motif.
    """
    frame = aadf.drop(columns=["AA_ID"]).T.astype(float).sort_index()
    labels = dict(zip(frame.columns, info["MOTIF"].astype(str)))
    renamed = {marker: f"motif_{i}" for i, marker in enumerate(frame.columns)}
    frame = frame.rename(columns=renamed)
    labels = {renamed[marker]: label for marker, label in labels.items()}
    frequency = frame.mean(axis=0) / 2.0
    passing = frequency.index[(frequency > .005) & (frequency < .995)]
    frame = frame.loc[:, passing]
    frequency = frequency.loc[passing]
    count = len(passing)
    reference = frequency.idxmax() if count else None
    reference_label = labels[reference] if reference is not None else None
    tests = [column for column in frame if column != reference]
    if count < 2 or not any(frame[column].nunique(dropna=True) > 1 for column in tests):
        columns, tests, reference_column = [], [], None
    else:
        columns = tests + ([reference] if retain_reference else [])
        reference_column = reference if retain_reference else None
    # Insertion states use implicit absence as reference, not another insertion.
    if str(variant_id).endswith("ins"):
        count = len(passing) + 1 if len(passing) else 0
        reference_label = "absence" if len(passing) else None
        tests = [column for column in frame if frame[column].nunique(dropna=True) > 1]
        columns, reference_column = tests, None
    geno = frame.loc[:, columns].copy()
    geno.columns = [f"G_{column}" for column in geno]
    geno.attrs["test_columns"] = tuple(f"G_{column}" for column in tests)
    geno.attrs["reference_column"] = f"G_{reference_column}" if reference_column else None
    geno.attrs["residue_labels"] = {f"G_{column}": labels[column] for column in columns}
    meta = {
        "VARIANT": variant_id, "GENE": info["GENE"].iloc[0],
        "AA_POS": info["AA_POS"].iloc[0],
        "Amino_Acids": ", ".join(labels[column] for column in passing),
        "Ref_AA": reference_label, "AAcount": count,
        "AA_AF": ";".join(f"{labels[column]}={frequency[column]:.6g}" for column in passing),
        "INFO": np.nan if tests else "insufficient variation",
    }
    return geno, meta
