"""Read PLINK text and binary filesets into HAPY dosage matrices."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pgenlib

from .HLAdat import HLAdata
from .io import GenomicsFileReader, _add_variant_info

__all__ = ["read_plink"]


def _read_pvar(path, rowskip=None):
    # PVAR permits VCF-style ## metadata before its #CHROM header.
    if rowskip is None:
        with open(path, encoding="utf-8") as stream:
            rowskip = 0
            for line in stream:
                if not line.startswith("##"):
                    break
                rowskip += 1
    frame = pd.read_csv(path, sep=r"\s+", skiprows=rowskip, dtype=str,
                        keep_default_na=False)
    required = {"ID", "REF", "ALT"}
    if not required.issubset(frame.columns):
        raise ValueError(f"PVAR must contain columns {sorted(required)}")
    return frame.set_index("ID")


def _validate_ids(values, label):
    ids = pd.Index(values, dtype=str)
    if ids.empty or ids.isin(["", ".", "NA"]).any() or ids.has_duplicates:
        raise ValueError(f"{label} must be nonempty, unique, non-missing identifiers")
    return ids.rename(None)


def _variant_kind(marker):
    if marker.startswith(("HLA_", "HLA*")):
        return "HLA"
    if marker.startswith("AA_"):
        return "AA"
    return "SNP"


def _effect_alleles(variants, dosage_allele):
    if dosage_allele == "ref":
        return variants["REF"].copy()
    effect = variants["ALT"].copy()
    if dosage_allele == "presence":
        # ALT is already correct unless a known HLA/AA presence pair has
        # P/T in REF and A in ALT. Keep original case in the output labels.
        hla_marker = variants.index.str.startswith(("HLA_", "HLA*", "AA_"))
        use_ref = (hla_marker & variants["REF"].str.upper().isin(["P", "T"])
                   & variants["ALT"].str.upper().eq("A"))
        effect.loc[use_ref] = variants.loc[use_ref, "REF"]
    return effect


def _read_raw(path, variants, effect):
    # Inspect just the header first, so excluded types never become strings or
    # numeric arrays in memory. RAW uses a whitespace-delimited, single header.
    with open(path, encoding="utf-8") as stream:
        header = stream.readline().split()
    normalised = [name.lstrip("#") if name in {"#FID", "#IID"} else name
                  for name in header]
    if "IID" not in normalised:
        raise ValueError("PLINK RAW must contain an IID column")
    if len(set(normalised)) != len(normalised):
        raise ValueError("PLINK RAW column names must be unique")
    iid_column = header[normalised.index("IID")]
    metadata = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "PHENO"}
    columns = [original for original, name in zip(header, normalised)
               if name not in metadata]
    parts = [column.rpartition("_") for column in columns]
    all_names = pd.Index([part[0] for part in parts])
    metadata_positions = variants.index.get_indexer(all_names)
    for column, (_, separator, _), position in zip(columns, parts, metadata_positions):
        if not separator or position < 0:
            raise ValueError(f"RAW column {column!r} has no matching PVAR variant")

    keep = all_names.isin(effect.index)
    selected_columns = [column for column, selected in zip(columns, keep) if selected]
    names = all_names[keep]
    counted = np.array([part[2] for part, selected in zip(parts, keep) if selected],
                       dtype=object)
    if len(names):
        _validate_ids(names, "RAW variant IDs")
    info = variants.iloc[metadata_positions[keep]]
    mismatch = (counted != info["REF"].to_numpy()) & (counted != info["ALT"].to_numpy())
    if mismatch.any():
        column = selected_columns[np.flatnonzero(mismatch)[0]]
        raise ValueError(f"RAW column {column!r} counts neither PVAR REF nor ALT")

    # Parse dosages directly as float64 and identifiers as strings. Per-column
    # missing tokens preserve literal sample IDs such as '001' and '-9'.
    dtypes = dict.fromkeys(selected_columns, np.float64)
    dtypes[iid_column] = str
    frame = pd.read_csv(
        path, sep=r"\s+", usecols=[iid_column, *selected_columns], dtype=dtypes,
        na_values=dict.fromkeys(selected_columns, ["", "NA", ".", "-9"]),
        keep_default_na=False,
    )
    samples = _validate_ids(frame[iid_column], "Sample IIDs")
    values = frame[selected_columns].to_numpy(dtype=np.float64, copy=True).T
    invalid = (np.isinf(values) | (values < 0) | (values > 2)).any(axis=1)
    if invalid.any():
        column = selected_columns[np.flatnonzero(invalid)[0]]
        raise ValueError(f"RAW column {column!r} contains dosage outside [0, 2]")
    reverse = counted != effect.reindex(names).to_numpy()
    np.subtract(2.0, values, out=values, where=reverse[:, None])
    return pd.DataFrame(values, index=names, columns=samples)


def _read_binary(path, variants, samples, effect):
    values = np.empty((len(effect), len(samples)), dtype=np.float64)
    reader_options = {}
    # Unselected multiallelic records still affect PGEN's internal allele layout.
    allele_counts = variants["ALT"].str.count(",").to_numpy() + 2
    if np.any(allele_counts > 2):
        reader_options["allele_idx_offsets"] = np.concatenate(
            ([0], np.cumsum(allele_counts))
        ).astype(np.uintp)
    # pgenlib reads both variant-major BED and PGEN; BED requires sample count.
    with pgenlib.PgenReader(str(path).encode(), raw_sample_ct=len(samples),
                           variant_ct=len(variants), **reader_options) as reader:
        source_indices = variants.index.get_indexer(effect.index)
        for output_index, (source_index, marker) in enumerate(zip(source_indices, effect.index)):
            allele_idx = int(effect.at[marker] == variants.at[marker, "ALT"])
            reader.read_dosages(int(source_index), values[output_index], allele_idx=allele_idx)
    values[values == -9] = np.nan
    return pd.DataFrame(values, index=effect.index, columns=samples)


def read_plink(
    fileloc,
    *,
    format: Literal["raw", "pgen", "bfile"] = "raw",
    pvarfileloc=None,
    pvar_rowskip=None,
    pvar_R2_column=None,
    filter_R2=None,
    R2_minimum: float = 0.5,
    simpleQC: bool = True,
    load_types: tuple = ("HLA", "SNP", "AA"),
    dosage_allele: Literal["presence", "ref", "alt"] = "presence",
) -> HLAdata:
    """Read a PLINK RAW, PGEN/PVAR/PSAM, or BED/BIM/FAM fileset.

    Parameters
    ----------
    fileloc : str or pathlib.Path
        RAW file or fileset prefix. PGEN and BED filenames are also accepted.
    format : {"raw", "pgen", "bfile"}
        ``bfile`` means a .bed/.bim/.fam fileset, not a .bfile extension.
    pvarfileloc : path, optional
        Variant metadata for RAW/PGEN. Defaults to <prefix>.pvar. Not used for
        bfile, whose alleles are read from BIM (A1=ALT, A2=REF).
    dosage_allele : {"presence", "ref", "alt"}
        Default counts P or T for HLA/AA A/P or A/T presence/absence pairs.
        Other HLA/AA encodings and all SNPs count labelled ALT. Allele letters
        are retained without inferring biological residue identities.
        RAW counted-allele suffixes are checked: 2 - dosage is applied
        exactly when the requested allele is the other allele. Binary inputs
        read the requested allele directly. REF/ALT options apply to all markers.
    pvar_rowskip : int, optional
        Explicit PVAR header offset; otherwise leading ## lines are skipped.
    pvar_R2_column : str, optional
        R2 field in PVAR INFO, or a named PVAR column. Requires PVAR input.
        Mutually exclusive with filter_R2.
    filter_R2, R2_minimum, simpleQC, load_types
        As in other `hapy` readers. MAF filtering uses observed dosages and keeps
        missing genotypes missing. load_types selects marker types before allele
        validation and dosage decoding. Only selected biallelic, diploid-scaled
        data are supported.

    Notes
    -----
    Binary inputs use the `pgenlib` dependency installed with `hapy`. PGEN fractional dosages are
    retained; BED supplies hardcalls represented as 0/1/2 dosage. Sample IIDs
    must be unique across families. This reader returns unphased dosages.

    By default, `dosage_allele="presence"` counts **P** for HLA/AA A/P pairs or
    **T** for HLA/AA A/T pairs. Other HLA/AA allele pairs (e.g. A/G or G/T) count
    the labelled ALT allele, as do all SNPs including `SNP_`/`SNPS_` markers.
    The reader preserves allele labels; it does not infer biological amino-acid
    identities from nucleotide or other surrogate codes. 
    """
    return PLINKFileReader(
        fileloc, format=format, pvarfileloc=pvarfileloc,
        pvar_rowskip=pvar_rowskip, pvar_R2_column=pvar_R2_column,
        filter_R2=filter_R2, R2_minimum=R2_minimum, simpleQC=simpleQC,
        load_types=load_types, dosage_allele=dosage_allele,
    ).read()


class PLINKFileReader(GenomicsFileReader):
    """PLINK format decoding integrated with the common loading/QC pipeline."""

    def __init__(self, fileloc, *, format="raw", pvarfileloc=None,
                 pvar_rowskip=None, pvar_R2_column=None, filter_R2=None,
                 R2_minimum=0.5, simpleQC=True, load_types=("HLA", "SNP", "AA"),
                 dosage_allele="presence"):
        super().__init__(filter_R2, R2_minimum, simpleQC, load_types)
        self.fileloc = fileloc
        self.format = format
        self.pvarfileloc = pvarfileloc
        self.pvar_rowskip = pvar_rowskip
        self.pvar_R2_column = pvar_R2_column
        self.dosage_allele = dosage_allele
        self._source_filter_R2 = filter_R2

    def read(self):
        self.filter_R2 = self._source_filter_R2
        return self._load_and_process(
            self.fileloc, self._read_dataframe, "softcall", drop_alleles=True,
        )

    def _read_dataframe(self, fileloc, _):
        format = self.format
        pvarfileloc = self.pvarfileloc
        pvar_rowskip = self.pvar_rowskip
        pvar_R2_column = self.pvar_R2_column
        filter_R2 = self.filter_R2
        R2_minimum = self.R2_minimum
        load_types = self.load_types
        dosage_allele = self.dosage_allele
        if format not in {"raw", "pgen", "bfile"}:
            raise ValueError("format must be 'raw', 'pgen', or 'bfile'")
        if dosage_allele not in {"presence", "ref", "alt"}:
            raise ValueError("dosage_allele must be 'presence', 'ref', or 'alt'")
        if isinstance(load_types, str):
            load_types = (load_types,)
        if not load_types or not set(load_types).issubset({"HLA", "SNP", "AA"}):
            raise ValueError("load_types must select HLA, SNP, and/or AA")
        if not 0 <= R2_minimum <= 1:
            raise ValueError("R2_minimum must be between 0 and 1")
        if filter_R2 is not None and pvar_R2_column is not None:
            raise ValueError("Use either filter_R2 or pvar_R2_column, not both")

        path = Path(fileloc)
        suffix = {"raw": ".raw", "pgen": ".pgen", "bfile": ".bed"}[format]
        prefix = str(path)[:-len(suffix)] if str(path).endswith(suffix) else str(path)
        path = Path(prefix + suffix)
        if not path.is_file():
            raise FileNotFoundError(path)

        if format == "bfile":
            if pvarfileloc is not None or pvar_R2_column is not None or pvar_rowskip is not None:
                raise ValueError("bfile uses BIM metadata; PVAR options are not applicable")
            variants = pd.read_csv(prefix + ".bim", sep=r"\s+", header=None,
                                   names=["CHROM", "ID", "CM", "POS", "ALT", "REF"],
                                   dtype=str, keep_default_na=False).set_index("ID")
            sample_frame = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None,
                                       dtype=str, keep_default_na=False)
            if sample_frame.shape[1] != 6:
                raise ValueError("FAM must contain six columns")
            samples = _validate_ids(sample_frame[1], "Sample IIDs")
        else:
            variants = _read_pvar(pvarfileloc or prefix + ".pvar", pvar_rowskip)
            if format == "pgen":
                # PSAM permits ## metadata and either #FID IID or #IID headers.
                sample_frame = _read_psam(prefix + ".psam")
                samples = _validate_ids(sample_frame["IID"], "Sample IIDs")

        _validate_ids(variants.index, "Variant IDs")
        annotations = _add_variant_info(pd.DataFrame(index=variants.index))
        annotations["TYPE"] = variants.index.map(_variant_kind)
        snp_rows = annotations["TYPE"] == "SNP"
        # SNP names are opaque IDs, not AA encodings. The shared legacy parser
        # truncates underscore-separated names and can mistake their text for AA_POS.
        annotations.loc[snp_rows, "AA_ID"] = variants.index[snp_rows]
        annotations.loc[snp_rows, "AA_POS"] = np.nan
        ordinary_snps = snp_rows & ~variants.index.str.startswith(("SNP_", "SNPS_"))
        annotations.loc[ordinary_snps, ["GENE", "POS"]] = np.nan
        selected = variants.loc[annotations["TYPE"].isin(load_types)]
        for column in ("REF", "ALT"):
            if selected[column].isin(["", ".", "0"]).any() or selected[column].str.contains(",", regex=False).any():
                raise ValueError("Only biallelic variants with known REF and ALT are supported")
        if (selected["REF"] == selected["ALT"]).any():
            raise ValueError("REF and ALT must be different")
        effect = _effect_alleles(selected, dosage_allele)
        dosage = (_read_raw(path, variants, effect) if format == "raw"
                  else _read_binary(path, variants, samples, effect))

        if pvar_R2_column is not None:
            if pvar_R2_column in variants:
                scores = variants[pvar_R2_column]
            elif "INFO" in variants:
                scores = variants["INFO"].map(lambda info: dict(
                    field.split("=", 1) for field in info.split(";") if "=" in field
                ).get(pvar_R2_column, np.nan))
            else:
                raise ValueError(f"PVAR has no {pvar_R2_column!r} column or INFO field")
            scores = pd.to_numeric(scores, errors="coerce")
            if scores.isna().all():
                raise ValueError(f"PVAR contains no numeric {pvar_R2_column!r} values")
            self.filter_R2 = scores.to_frame(name=1)
        alleles = pd.DataFrame({"alleleA": effect,
                                "alleleB": selected["REF"].where(effect == selected["ALT"], selected["ALT"])})
        data = annotations.loc[dosage.index].join(alleles).join(dosage)
        data.index.name = "SNP"
        if "POS" in variants:
            data["POS"] = pd.to_numeric(variants["POS"].reindex(data.index), errors="raise")
        return data

    def _annotate_variants(self, dataframe, index_name):
        # PLINK annotation already includes source coordinates, exact SNP IDs,
        # and reader-specific type selection. Do not parse these names again.
        dataframe.index.name = index_name
        return dataframe

    def _prepare_data(self, hladat):
        if hasattr(hladat, "SNP"):
            block = hladat.SNP
            block.extradata.drop(columns=["alleleA", "alleleB"], inplace=True)
            all_snps = pd.concat([block.data, block.extradata]).reindex(block.info.index)
            extra_mask = all_snps.index.str.startswith(("SNP_", "SNPS_"))
            block.data = all_snps.loc[~extra_mask].copy()
            block.extradata = all_snps.loc[extra_mask].copy()
            block.extrainfo = block.info.loc[block.extradata.index].copy()
            block.info = block.info.loc[block.data.index].copy()
        return hladat

    def _apply_simple_qc(self, hladat):
        hladat.maf_filter(preserve_missing=True, include_extradata=True)


def _read_psam(path):
    with open(path, encoding="utf-8") as stream:
        rowskip = 0
        for line in stream:
            if not line.startswith("##"):
                break
            rowskip += 1
    frame = pd.read_csv(path, sep=r"\s+", skiprows=rowskip, dtype=str,
                        keep_default_na=False)
    frame = frame.rename(columns={"#IID": "IID", "#FID": "FID"})
    if "IID" not in frame:
        raise ValueError("PSAM must contain an IID column")
    return frame
