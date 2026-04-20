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
import pandas as pd
import numpy as np 

from ..preprocess import subset_samples_beagle_orientation
from ._aa_haplo import obt_haplo_soft, obt_haplo_hard


def _af_from_col(s: pd.Series) -> float:
    """Compute allele frequency estimate p = mean(dosage)/2."""
    return float(s.mean() / 2.0)

def aa_allele_frequency(aminoacid_df):
    """
    Computes the allele frequency of the amio acids in the variant.

    Parameters
    ----------
    aminoacid_df:
        The pd.DataFrame that contains the amino acid counts.

    Returns
    -------
    af_str:
        AF for each amino acid present in the variant.
    """
    df = aminoacid_df.drop(columns=["AA_ID"]).copy()

    tmp = {c.split("_")[-1].replace(".", "dot").replace("*", "asterisk"): _af_from_col(df.T[c]) for c in df.T.columns}
    
    aa_freq = {}
    for k, v in tmp.items():
        if len(k)==1:
            aa_freq[k]=v
    
    return ";".join([f"{k}={v:.6g}" for k, v in aa_freq.items()]) if aa_freq else np.nan

class AAAdapter:
    """Adapter for HLAdat.AA genotype block."""
    KIND = "AA"

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
    def build_geno(hladat, variant_id: str, sample_index: pd.Index) -> tuple[pd.DataFrame, dict]:
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
        sample_index:
            IID index for analysis (strings). Used to subset genotype columns.

        Returns
        -------
        geno_df:
            DataFrame indexed by IID with numeric feature columns (prefixed "G_").
        meta:
            Dict of annotations useful for output tables.
        """
        df = hladat.AA.data.copy()
        
        info = hladat.AA.info.copy()

        df.index = df.index.astype(str)
        df = subset_samples_beagle_orientation(df, sample_index, hladat.type)

        df["AA_ID"] = info["AA_ID"]
        aadf = df[df["AA_ID"] == variant_id]
        aainfo = info[info["AA_ID"] == variant_id]

        if hladat.type == "softcall":
            haplodf, AAcount, refAA, aalist, _ = obt_haplo_soft(aadf, aainfo)
        else:
            haplodf, AAcount, refAA, aalist, _ = obt_haplo_hard(aadf)

        haplodf = haplodf.sort_index()
        haplodf.columns = [f"G_{c}" for c in haplodf.columns]

        meta = {
            "VARIANT": variant_id,
            "GENE": aainfo["GENE"].iloc[0] if "GENE" in aainfo else None,
            "AA_POS": aainfo["AA_POS"].iloc[0] if "AA_POS" in aainfo else None,
            "Amino_Acids": ", ".join(sorted({str(x) for x in aalist})),
            "Ref_AA": refAA,
            "AAcount": AAcount,
            "AA_AF": aa_allele_frequency(aadf)
        }

        return haplodf, meta