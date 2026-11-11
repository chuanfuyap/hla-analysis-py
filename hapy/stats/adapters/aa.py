"""
AAAdapter

Builds per-variant genotype matrices for amino-acid (AA) positions from an HLAdat object.

Contract
--------
- iter_variants(hladat) -> list[str]
- build_geno(hladat, variant_id, sample_index) -> (geno_df, meta)

Output
------
geno_df:
    - index: IID strings
    - columns: numeric genotype features (dosage/copy number)
meta:
    - dict with VARIANT and other informative fields (GENE, AA_POS, Ref_AA, etc.)

Notes
-----
For multi-allelic AA positions, the haplotype builder will drop a reference column (k-1 representation),
which is suitable for omnibus testing.
"""

from __future__ import annotations

from ._samples import genotype_sample_ids
import pandas as pd
import numpy as np 

from ..preprocess import subset_samples_beagle_orientation
from ._aa_haplo import obt_haplo_soft, obt_haplo_hard


def _af_from_col(s: pd.Series) -> float:
    """Compute allele frequency estimate p = mean(dosage)/2."""
    return float(s.mean() / 2.0)

def _af_from_col_hard(s: pd.Series) -> float:
    """Compute allele frequency estimate by counting number of presence."""
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    p_A = (s == "A").mean()
    return 1.0 - p_A

def aa_allele_frequency(aminoacid_df, calltype, refAA, effect_allele=None):
    """
    Computes the allele frequency of the amio acids in the variant.

    Parameters
    ----------
    aminoacid_df:
        The pd.DataFrame that contains the amino acid counts.
    calltype:
        "softcall" or "hardcall".
    refAA:
        the reference amino acid as some file types does not store multiple alleles. 

    Returns
    -------
    af_str:
        AF for each amino acid present in the variant.
    """
    df = aminoacid_df.drop(columns=["AA_ID"]).copy()

    if calltype=="softcall":
        tmp = {c.split("_")[-1].replace(".", "dot").replace("*", "asterisk"): _af_from_col(df.T[c]) for c in df.T.columns}
    else:
        tmp = {c.split("_")[-1].replace(".", "dot").replace("*", "asterisk"): _af_from_col_hard(df.T[c]) for c in df.T.columns}
    
    aa_freq = {}
    for k, v in tmp.items():
        if len(k)==1:
            aa_freq[k]=v
    
    if aa_freq:
        return ";".join([f"{k}={v:.6g}" for k, v in aa_freq.items()])
    else:
        label = effect_allele if effect_allele is not None else refAA
        return f"{label}={list(tmp.values())[0]}"

class AAAdapter:
    """Adapter for HLAdat.AA genotype block."""
    KIND = "AA"

    @staticmethod
    def sample_ids(hladat, requested):
        return genotype_sample_ids(hladat.AA.data, requested, hladat.type)

    @staticmethod
    def iter_variants(hladat) -> list[str]:
        """
        List available AA positions/variants.

        Parameters
        ----------
        hladat:
            HLAdat object with hladat.AA.info containing column 'AA_ID'.

        Returns
        -------
        list[str]
            Unique AA_ID values.
        """
        return list(hladat.AA.info["AA_ID"].unique())

    @staticmethod
    def build_geno(hladat, variant_id: str, sample_index: pd.Index, retain_reference: bool = False) -> tuple[pd.DataFrame, dict]:
        """
        Build genotype feature matrix for one AA_ID.

        Parameters
        ----------
        hladat:
            HLAdat object with fields:
                - hladat.AA.data: variants x samples
                - hladat.AA.info: variant annotations including AA_ID
                - hladat.type: "softcall" or "hardcall"
        variant_id:
            AA_ID key to subset.
        retain_reference : bool, default False
            Retain QC-passing reference dosages for OVR tests. The original
            non-reference design is recorded in DataFrame.attrs["test_columns"].
        sample_index:
            IID index for analysis (strings). Used to subset genotype columns.

        Returns
        -------
        geno_df:
            DataFrame indexed by IID with numeric feature columns (prefixed "G_").
        meta:
            Dict of annotations useful for output tables.
        """
        info = hladat.AA.info

        mask = info["AA_ID"].eq(variant_id).to_numpy()

        aadf = hladat.AA.data.iloc[mask].copy()
        aainfo = info.iloc[mask]

        aadf.index = aadf.index.astype(str)
        aadf = subset_samples_beagle_orientation(
            aadf, sample_index, hladat.type
        )
        if aadf.shape[1] == 0:
            raise ValueError(f"{variant_id}: no overlapping genotype/phenotype sample IDs.")

        aadf["AA_ID"] = aainfo["AA_ID"].to_numpy()

        if hladat.type == "softcall":
            haplodf, AAcount, refAA, aalist, _ = obt_haplo_soft(aadf, aainfo, retain_reference=retain_reference)
        else:
            haplodf, AAcount, refAA, aalist, _ = obt_haplo_hard(aadf, retain_reference=retain_reference)

        haplodf = haplodf.sort_index()
        test_columns = haplodf.attrs.get("test_columns", tuple(haplodf.columns))
        reference_column = haplodf.attrs.get("reference_column")
        haplodf.columns = [f"G_{c}" for c in haplodf.columns]
        haplodf.attrs["test_columns"] = tuple(f"G_{c}" for c in test_columns)
        haplodf.attrs["reference_column"] = f"G_{reference_column}" if reference_column is not None else None
        labels = {c: c[2:] for c in haplodf.columns}
        if "G_solo_amino_acid" in labels:
            suffix = str(aadf.index[0]).split("_")[-1]
            labels["G_solo_amino_acid"] = suffix if len(suffix) == 1 else str(aainfo["alleleA"].iloc[0])
        haplodf.attrs["residue_labels"] = labels

        meta = {
            "VARIANT": variant_id,
            "GENE": aainfo["GENE"].iloc[0] if "GENE" in aainfo else None,
            "AA_POS": aainfo["AA_POS"].iloc[0] if "AA_POS" in aainfo else None,
            "Amino_Acids": ", ".join(sorted({str(x) for x in aalist})),
            "Ref_AA": refAA,
            "AAcount": AAcount,
            "AA_AF": aa_frequency(haplodf, haplodf.index) if retain_reference else aa_allele_frequency(
                aadf, hladat.type, refAA,
                effect_allele=(aainfo["alleleA"].iloc[0]
                               if hladat.type == "softcall" and len(aadf) == 1 else None),
            )
        }

        meta['INFO'] = 'insufficient variation' if len(test_columns) == 0 else np.nan
        return haplodf, meta

def aa_frequency(geno, index, tested=None):
    """Format dosage AFs using final model samples, never a QC-column complement."""
    labels = geno.attrs.get("residue_labels", {})
    columns = [tested] if tested is not None else list(geno.columns)
    values = geno.reindex(index)
    if len(values) == 0:
        return np.nan
    return ";".join(f"{labels.get(c, c.removeprefix('G_'))}={values[c].sum(min_count=len(values)) / (2 * len(values)):.6g}"
                    for c in columns)
