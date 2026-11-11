"""
Amino-acid (AA) haplotype helpers extracted from the legacy stats module.

These functions are used by AAAdapter (and reused by HLA/SNP adapters for hardcall
haplotype construction where applicable).

Design notes
------------
- Input Beagle-derived tables are expected in the legacy orientation:
  rows = variants (AA alleles), columns = samples.
- The calling adapters handle subsetting columns to match sample IDs.

This file intentionally preserves legacy logic (including frequency thresholds and
string encodings like 'T'/'P') to keep results comparable with historical runs.
"""

from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd


def makehaplodf(aa_df: pd.DataFrame, basicQC: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate a haplotype-count matrix for hardcall AA data at one AA position.

    Parameters
    ----------
    aa_df:
        DataFrame containing variants (rows) for a single AA position/gene and samples (columns).
        Must include an 'AA_ID' column.
    basicQC:
        If True, drop haplotypes with frequency <= 0.005 or >= 0.995.

    Returns
    -------
    haplo_df:
        DataFrame indexed by sample ID (without .1 suffix), columns are haplotype strings,
        values are haplotype counts (0/1/2).
    amino_acids:
        List of simplified amino acid labels corresponding to haplotypes.
    """
    df = aa_df.copy()
    aminoacids = df.index

    df = df.drop(columns=["AA_ID"]).T
    df["haplo"] = df.apply(lambda x: "".join(x), axis=1)

    df = df.reset_index()
    df["index"] = df["index"].apply(lambda x: x.split(".")[0])
    df = df.groupby(["index", "haplo"]).size().unstack(fill_value=0)

    if basicQC:
        highfreq = df.sum(0) / 2 / df.shape[0] > 0.005
        highfreq = highfreq[highfreq]
        df = df[highfreq.index]

        highfreq2 = df.sum(0) / 2 / df.shape[0] < 0.995
        highfreq2 = highfreq2[highfreq2]
        df = df[highfreq2.index]

    haplo = df.columns
    aminoacids = get_aminoacids(aminoacids, haplo)

    return df.sort_index(), aminoacids


def makehaploprob(aa_df: pd.DataFrame, basicQC: bool = True) -> pd.DataFrame:
    """
    Generate a haplotype dosage matrix for softcall AA data.

    Parameters
    ----------
    aa_df:
        DataFrame with AA variants (rows) and samples (columns), includes 'AA_ID' column.
    basicQC:
        If True, drop alleles with frequency <= 0.005 or >= 0.995.

    Returns
    -------
    df:
        DataFrame indexed by sample IDs, columns are amino acid letters (non-ambiguous),
        values are dosages.
    """
    df = aa_df.copy()

    df = df.drop(columns=["AA_ID"], axis=1)
    df["AA"] = df.index
    df["AA"] = df["AA"].apply(lambda x: x.split("_")[-1])

    df["aa_length"] = df["AA"].apply(lambda x: len(x))
    df = df[df["aa_length"] == 1]

    df = df.set_index("AA").drop("aa_length", axis=1)
    df = df.astype("float")
    df = df.T.sort_index()

    if basicQC:
        highfreq = df.sum(0) / (df.shape[0] * 2) > 0.005
        highfreq = highfreq[highfreq]
        df = df[highfreq.index]

        highfreq2 = df.sum(0) / (df.shape[0] * 2) < 0.995
        highfreq2 = highfreq2[highfreq2]
        df = df[highfreq2.index]

    return df.sort_index()


def checkAAblock(aablock: list[str]) -> str:
    """
    Reduce an ambiguous AA block (e.g. ['FY','FS','FK']) to a single representative AA if possible.

    The legacy heuristic picks the most frequent AA character across the block if it is uniquely
    most frequent; otherwise returns the original block unchanged (as a string) or empty.

    Parameters
    ----------
    aablock:
        List of composite amino acid labels.

    Returns
    -------
    str:
        Representative AA (single character) if resolvable, else a composite label or empty string.
    """
    aminoacids = [list(i) for i in aablock]
    aminoacids = [x for i in aminoacids for x in i]
    aminoacids = Counter(aminoacids)
    aminoacids = dict(sorted(aminoacids.items(), key=lambda item: item[1], reverse=True))

    if len(aminoacids) > 1:
        aakeys = list(aminoacids.keys())
        if aminoacids[aakeys[0]] != aminoacids[aakeys[1]]:
            aablock = aakeys[0]
        elif "x" in aakeys:
            aablock = ("").join(aakeys)
    elif len(aminoacids) == 1:
        aablock = aablock[0]
    else:
        aablock = ""
    return aablock


def get_aminoacids(idlist: list[str] | pd.Index, haplotypes: list[str] | pd.Index) -> list[str]:
    """
    Map haplotype strings back to simplified amino-acid labels using legacy presence encoding.

    Parameters
    ----------
    idlist:
        Variant IDs containing the AA suffix, e.g. 'AA_A_19_..._F'
    haplotypes:
        Haplotype strings composed of presence/absence characters.

    Returns
    -------
    list[str]:
        Simplified amino acid labels per haplotype.
    """
    aalist = np.array([i.split("_")[-1] for i in idlist])
    aablocks: list[str] = []

    for x in haplotypes:
        haplo = np.array(list(x))
        presence = list(np.where((haplo == "P") | (haplo == "T") | (haplo == "p"))[0])
        block = list(aalist[presence])

        block = checkAAblock(block)
        if isinstance(block, list):
            block = (("").join(block))
        if block:
            aablocks.append(block)

    return aablocks


def getRefAA(haplo: str, aalist: list[str] | pd.Index) -> str:
    """
    Derive the reference AA label for a dropped reference haplotype.

    Parameters
    ----------
    haplo:
        Haplotype string.
    aalist:
        List/Index of AA variant IDs.

    Returns
    -------
    str:
        Reference AA label.
    """
    haplo = np.array(list(haplo))
    presence = list(np.where((haplo == "P") | (haplo == "T") | (haplo == "p"))[0])

    aminoacids = np.array([i.split("_")[-1] for i in aalist])
    aminoacids = aminoacids[presence]
    aminoacids = checkAAblock(list(aminoacids))

    return aminoacids


def obt_haplo_hard(aadf: pd.DataFrame, retain_reference: bool = False) -> tuple[pd.DataFrame, int, str, list[str], int]:
    """
    Build QC-filtered hardcall dosages with the legacy reference selection.

    retain_reference=True appends the original QC-passing reference dosage;
    the default design columns remain recorded in attrs["test_columns"].
    No QC-filtered haplotypes are restored and dosages are not renormalised.

    Returns
    -------
    haplodf:
        Haplotype count matrix (samples x haplotypes), possibly with reference dropped.
    AAcount:
        Number of amino acids/haplotypes considered for decision logic.
    refAA:
        The reference amino acid label.
    aalist:
        Amino acids list.
    haplocount:
        Number of haplotype columns after dropping reference/missing.
    """
    reference = None
    if aadf.shape[0] > 1:
        haplodf, aalist = makehaplodf(aadf)
        AAcount = haplodf.shape[1]
        if AAcount == 0:
            return haplodf, 0, None, [], 0

        missing = "".join(np.repeat("A", aadf.shape[0]))
        missing2 = "".join(np.repeat("a", aadf.shape[0]))
        if missing in haplodf.columns or missing2 in haplodf.columns:
            haplodf = haplodf.drop(columns=[missing, missing2], errors="ignore")
            haplocount = haplodf.shape[1]
            haplodf.columns = aalist
            refAA = "missing"
            aalist.append("missing")
        else:
            haplo = haplodf.columns
            haplodf.columns = aalist
            refix = np.argmax(haplodf.sum())
            refcol = haplodf.columns[refix]
            haplo = haplo[refix]
            reference = haplodf[[refcol]]
            haplodf = haplodf.drop(refcol, axis=1)
            haplocount = haplodf.shape[1]
            refAA = getRefAA(haplo, aadf.index)
    else:
        haplodf, aalist = makehaplodf(aadf)
        aalist = list(haplodf.columns.values)
        haplodf.columns = aalist

        if len(aalist) > 1:
            refix = np.argmax(haplodf.sum())
            refcol = haplodf.columns[refix]
            reference = haplodf[[refcol]]
            haplodf = haplodf.drop(columns = refcol)
            haplocount = haplodf.shape[1]
            refAA = refcol
            AAcount = 2
        else:
            aalist.append("missing")
            haplocount = haplodf.shape[1]
            refAA = "missing"
            AAcount = 0

    haplodf.columns = [cname.replace(".", "dot").replace("*", "asterisk") for cname in haplodf.columns]

    test_columns = tuple(haplodf.columns)
    reference_column = None
    if retain_reference and reference is not None:
        reference = reference.copy()
        reference.columns = [str(c).replace(".", "dot").replace("*", "asterisk") for c in reference.columns]
        reference_column = reference.columns[0]
        haplodf = pd.concat([haplodf, reference], axis=1)
    if retain_reference:
        haplodf.attrs["test_columns"] = test_columns
        haplodf.attrs["reference_column"] = reference_column
    return haplodf, AAcount, refAA, aalist, haplocount


def obt_haplo_soft(aadf: pd.DataFrame, infodf: pd.DataFrame, retain_reference: bool = False) -> tuple[pd.DataFrame, int, str, list[str], int]:
    """
    Legacy helper: build haplotype matrix for softcall AA and drop reference haplotype.

    Parameters
    ----------
    aadf:
        Subset AA dataframe for a single AA_ID.
    infodf:
        Matching info subset for the AA_ID.
    retain_reference : bool, default False
        Keep the original QC-passing reference dosage, without changing reference
        selection or QC thresholds. For a single binary marker with known allele
        labels, its opposite dosage is the complement of that original marker.
        No complement is inferred from a QC-filtered multiresidue matrix.

    Returns
    -------
    haplodf:
        Dosage matrix (samples x alleles) with reference dropped for multi-allelic case.
    AAcount:
        Count used by legacy branching (>=3 => omnibus; ==2 => univariate; else skip).
    refAA:
        Reference AA label.
    aalist:
        List of alleles/labels.
    haplocount:
        Number of columns after reference dropping.
    """
    reference = None
    if aadf.shape[0] > 1:
        haplodf = makehaploprob(aadf)
        aalist = list(haplodf.columns)
        AAcount = len(aalist)
        if haplodf.shape[1] == 0:
            return haplodf, 0, None, [], 0

        refix = np.argmax(haplodf.sum())
        refAA = haplodf.columns[refix]
        reference = haplodf[[refAA]]
        haplodf = haplodf.drop(refAA, axis=1)
        haplocount = haplodf.shape[1]

        haplodf.columns = [cname.replace(".", "dot").replace("*", "asterisk") for cname in haplodf.columns]
    else:
        haplodf = aadf.drop("AA_ID", axis=1).T
        haplodf.columns = ["solo_amino_acid"]

        freq = (haplodf.sum(0) / (haplodf.shape[0] * 2)).values[0]
        if (freq > 0.005) and (freq < 0.995):
            AAcount = 2
            suffix = aadf.index[0].split("_")[-1]
            if len(suffix) == 1:
                refAA = "missing"
                aalist = [suffix, "missing"]
            else:
                refAA = infodf["alleleB"].values[0]
                aalist = list(infodf[["alleleA", "alleleB"]].values[0])
            if refAA != "missing":
                reference = pd.DataFrame({str(refAA): 2.0 - haplodf.iloc[:, 0]}, index=haplodf.index)
            haplocount = haplodf.shape[1]
        else:
            AAcount = 0
            refAA = infodf["alleleB"].values[0]
            aalist = list(infodf[["alleleA", "alleleB"]].values[0])
            haplocount = haplodf.shape[1]

    test_columns = tuple(haplodf.columns)
    reference_column = None
    if retain_reference and reference is not None:
        reference = reference.copy()
        reference.columns = [str(c).replace(".", "dot").replace("*", "asterisk") for c in reference.columns]
        reference_column = reference.columns[0]
        haplodf = pd.concat([haplodf, reference], axis=1)
    if retain_reference:
        haplodf.attrs["test_columns"] = test_columns
        haplodf.attrs["reference_column"] = reference_column
    return haplodf, AAcount, refAA, aalist, haplocount