"""Map diploid HLA allele calls through a saved local reference, retaining phase."""
from dataclasses import dataclass, field
from io import StringIO
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from .hla_reference import HLAReference, load_hla_reference, normalise_hla_allele, _normalise_gene, _single_field_names, _sha
from .HLAdat import HLAdata
from ..stats.preprocess import sample_index

_MARKER = re.compile(r"^(AA_([A-Z0-9]+)_(-?\d+(?:i[1-9]\d*)?)_(NA|\d+)_exon(NA|\d+))_([ACDEFGHIKLMNPQRSTVWYX.])$")


def parse_aa_marker(marker):
    """Parse a generated residue marker; NA coordinates remain missing, never zero."""
    match = _MARKER.fullmatch(str(marker))
    if not match or (match[4] != "NA" and int(match[4]) <= 0):
        raise ValueError(f"Invalid mapped AA marker {marker!r}.")
    return {"AA_ID": match[1], "GENE": match[2], "AA_POS": match[3],
            "POS": None if match[4] == "NA" else int(match[4]),
            "EXON": None if match[5] == "NA" else int(match[5]), "RESIDUE": match[6]}


@dataclass(eq=False)
class HLAAlleleMapping:
    """Compact called-allele matrices and two integer allele IDs per sample/gene.

    IDs refer to matrix rows; row zero is unresolved. Matrices include only
    distinct called alleles/consensus calls, not the entire reference database.
    The two allele slots are preserved, but cross-gene phase is not inferred.
    """
    calls: pd.DataFrame
    alleles: dict
    matrices: dict
    allele_ids: dict
    positions: dict
    metadata: dict
    issues: list = field(default_factory=list)
    motif_cache: dict = field(default_factory=dict, repr=False)
    dosage_file_sha256: str | None = field(default=None, repr=False)

    def motif_dosages(self, gene, positions):
        """Return phased motif counts for one locus and ordered AA positions.

        Parameters
        ----------
        gene : str
            Mapped locus, optionally HLA-prefixed.
        positions : iterable of str or int
            Nonempty unique position labels, in the order to join residues.

        Returns
        -------
        pandas.DataFrame
            Sample-by-motif 0/1/2 counts; rows with incomplete selected allele
            sequences are missing. Attributes identify motif allele groups and
            unresolved sample count. Assignments are cached on this mapping.

        Raises
        ------
        ValueError
            For an unmapped gene, unknown positions, or empty/duplicate positions.

        See Also
        --------
        aa_motif_dosages : Public wrapper accepting mapped data or this object.
        """
        gene = _normalise_gene(gene)
        if gene not in self.matrices: raise ValueError(f"Gene {gene!r} was not mapped.")
        positions = tuple(str(p) for p in positions)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("Select a nonempty set of unique AA positions.")
        lookup = {str(pos): i for i, pos in enumerate(self.positions[gene]["AA_POS"])}
        missing = set(positions) - set(lookup)
        if missing: raise ValueError(f"Unknown {gene} positions: {sorted(missing)}")
        key = (gene, self.metadata["reference_version"], positions)
        if key not in self.motif_cache:
            selected = self.matrices[gene][:, [lookup[p] for p in positions]]
            motifs = [None if "?" in row else "|".join(row) for row in selected]
            labels = sorted({m for m in motifs if m is not None})
            codes = {label: i for i, label in enumerate(labels)}
            assignment = np.array([codes.get(m, -1) for m in motifs], dtype=np.int32)
            groups = {label: [name for name, motif in zip(self.alleles[gene], motifs) if motif == label]
                      for label in labels}
            self.motif_cache[key] = (assignment, labels, groups)
        assignment, labels, groups = self.motif_cache[key]
        ids = assignment[self.allele_ids[gene]]
        complete = (ids >= 0).all(axis=1)
        dosage = np.column_stack([(ids == code).sum(axis=1) for code in range(len(labels))]).astype(float) if labels else np.empty((len(ids), 0))
        dosage[~complete, :] = np.nan
        frame = pd.DataFrame(dosage, index=self.calls.index.copy(), columns=labels)
        frame.attrs.update({"gene": gene, "positions": positions, "reference_version": self.metadata["reference_version"],
                            "allele_groups": {k: list(v) for k, v in groups.items()},
                            "N_unresolved": int((~complete).sum())})
        return frame

    def save(self, path):
        """Save calls and phase metadata to a compressed NPZ sidecar.

        Parameters
        ----------
        path : str or pathlib.Path
            Output filename. Existing files are overwritten; parent must exist.

        Returns
        -------
        None

        Notes
        -----
        Stores original calls, called-allele matrices, integer allele pairs, issues,
        positions and metadata without pickle. Motif caches are not saved. The
        optional dosage fingerprint is retained for TSV/sidecar consistency checks.
        """
        payload = {"metadata": np.array(json.dumps({"metadata": self.metadata, "issues": self.issues,
            "calls": json.loads(self.calls.to_json(orient="split")),
            "multi_columns": isinstance(self.calls.columns, pd.MultiIndex),
            "column_names": list(self.calls.columns.names), "genes": list(self.alleles),
            "dosage_file_sha256": self.dosage_file_sha256}))}
        for gene in self.alleles:
            payload[f"{gene}_alleles"] = np.asarray(self.alleles[gene])
            payload[f"{gene}_matrix"] = self.matrices[gene]
            payload[f"{gene}_ids"] = self.allele_ids[gene]
            payload[f"{gene}_positions"] = np.array(self.positions[gene].to_json(orient="table", index=False))
        with Path(path).open("wb") as stream: np.savez_compressed(stream, **payload)

    @classmethod
    def load(cls, path):
        """Load a saved allele mapping without reconstructing phase from dosages.

        Parameters
        ----------
        path : str or pathlib.Path
            NPZ sidecar previously written by HLAAlleleMapping.save.

        Returns
        -------
        HLAAlleleMapping
            Restored calls, positions, issues and read-only sequence/ID arrays.
            Motif cache starts empty.

        Raises
        ------
        ValueError
            If stored array dimensions or allele IDs are inconsistent.
        OSError
            If the file cannot be opened.

        Notes
        -----
        The TSV fingerprint is checked by read_aa_dosage, not by this standalone
        loader. No pickle loading or network access is performed.
        """
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata"]))
            raw = header["calls"]
            columns = pd.MultiIndex.from_tuples([tuple(c) for c in raw["columns"]], names=header["column_names"]) if header["multi_columns"] else raw["columns"]
            calls = pd.DataFrame(raw["data"], index=raw["index"], columns=columns)
            calls.index = sample_index(calls.index, source="HLA allele calls")
            genes = header["genes"]
            alleles = {g: archive[f"{g}_alleles"].tolist() for g in genes}
            matrices = {g: archive[f"{g}_matrix"].copy() for g in genes}
            ids = {g: archive[f"{g}_ids"].copy() for g in genes}
            positions = {g: pd.read_json(StringIO(str(archive[f"{g}_positions"])), orient="table") for g in genes}
        for gene in genes:
            if ids[gene].shape != (len(calls), 2) or matrices[gene].shape != (len(alleles[gene]), len(positions[gene])):
                raise ValueError("Invalid allele-phase sidecar dimensions.")
            if ids[gene].dtype.kind not in "iu" or (ids[gene] < 0).any() or (ids[gene] >= len(alleles[gene])).any():
                raise ValueError("Invalid allele IDs in phase sidecar.")
            matrices[gene].flags.writeable = False; ids[gene].flags.writeable = False
        return cls(calls, alleles, matrices, ids, positions, header["metadata"], header["issues"],
                   dosage_file_sha256=header.get("dosage_file_sha256"))


def _allele_columns(calls, reference, allele_columns):
    if allele_columns is not None:
        columns = dict(allele_columns)
    else:
        columns = {}
        for gene in reference.alleles:
            pair = ((gene, 1), (gene, 2)) if isinstance(calls.columns, pd.MultiIndex) else (gene + "_1", gene + "_2")
            present = [col in calls for col in pair]
            if any(present) and not all(present): raise ValueError(f"{gene}: supply both allele columns.")
            if all(present): columns[gene] = pair
    if not columns: raise ValueError("No allele pairs found; use GENE_1/GENE_2 columns or allele_columns.")
    selected = []
    for gene, pair in columns.items():
        if gene not in reference.alleles or len(pair) != 2 or any(c not in calls for c in pair):
            raise ValueError(f"Invalid allele-column mapping for {gene!r}.")
        selected.extend(pair)
    if len(set(selected)) != len(selected): raise ValueError("Each allele slot must use a distinct input column.")
    return columns


def map_hla_alleles(calls, reference, *, allele_columns=None, output="hladata", save_path=None, strict=False, fallback_to_two_field=False):
    """Map diploid allele calls to amino acid dosages while retaining within-gene phase.

    Parameters
    ----------
    calls : pandas.DataFrame
        Samples in rows, indexed by unique IID. Default columns are GENE_1 and
        GENE_2, such as A_1/A_2. MultiIndex columns (gene, 1)/(gene, 2) are also
        accepted. Values are individual allele names or missing values. Calls
        may omit the gene prefix when the column identifies the locus.
    reference : HLAReference or str or pathlib.Path
        Built reference or path to its saved NPZ file.
    allele_columns : mapping, optional
        Gene-to-two-column-name mapping, for example
        ``{"DRB1": ("DRB1_first", "DRB1_second")}``. Otherwise discover complete
        column pairs for loci in the reference.
    output : {"hladata", "dataframe"}, default "hladata"
        Return an HLAdata object or sample-by-residue pandas DataFrame.
    save_path : str or pathlib.Path, optional
        Write dosages as an IID-indexed TSV and mapping as
        ``<save_path>.phase.npz``. Existing files are overwritten. The sidecar
        preserves calls, allele IDs, sequences, issues and reference metadata.
    strict : bool, default False
        Raise ValueError instead of returning data if any call or sample-position
        pair remains unresolved. Successful complete fallbacks are permitted.
    fallback_to_two_field : bool, default False
        If a three/four-field call has no compatible reference record, try
        two-field consensus while retaining its expression suffix. All matching
        records must agree at each position; conflicting or unknown positions
        stay missing. Never replaces an existing but incomplete sequence.

    Returns
    -------
    HLAdata or pandas.DataFrame
        Residue dosages 0, 1 or 2, with NaN for every residue state at a sample/
        position if either allele is unresolved there. Marker names encode gene,
        AA position, genomic position, exon and residue. Missing coordinates
        use NA. HLAdata exposes ``hla_mapping``; DataFrame output stores it in
        ``attrs["hla_mapping"]``. The mapping retains original calls and integer
        within-gene allele pairs, not cross-gene phase.

    Raises
    ------
    TypeError
        If calls is not a DataFrame.
    ValueError
        For invalid sample IDs, columns, allele notation, output selection or
        unresolved data when strict=True.

    Warns
    -----
    UserWarning
        Summarises unresolved information and use of inferred two-field
        fallback. Individual absent/fallback calls are recorded in mapping.issues;
        fallback entries include the original normalised allele and mapped_as.

    Notes
    -----
    Does not modify calls or apply frequency/variation filtering. Monomorphic
    positions are retained. A known allele is not doubled when its partner is
    missing. Lower-resolution consensus never arbitrarily chooses a representative
    sequence. Fallback is an inference, not confirmation of the unavailable allele.

    Examples
    --------
    >>> hladat = map_hla_alleles(calls, reference, fallback_to_two_field=True)
    >>> dosages = map_hla_alleles(calls, reference, output="dataframe")
    """
    if not isinstance(calls, pd.DataFrame): raise TypeError("calls must be a pandas DataFrame indexed by IID.")
    if not calls.columns.is_unique: raise ValueError("Allele-call columns must be unique.")
    if output not in {"hladata", "dataframe"}: raise ValueError("output must be 'hladata' or 'dataframe'.")
    if not isinstance(reference, HLAReference): reference = load_hla_reference(reference)
    original = calls.copy(deep=True)
    original.index = sample_index(original.index, source="HLA allele calls")
    columns = _allele_columns(original, reference, allele_columns)
    metadata = {"reference_version": reference.version, "imgt_release": reference.metadata["imgt_release"],
                "assembly": reference.metadata["assembly"], "phase": "two allele slots within each gene; cross-gene phase not inferred"}
    mapping = HLAAlleleMapping(original, {}, {}, {}, {}, metadata)
    frames = []
    unresolved = 0
    fallbacks = []
    for gene, pair in columns.items():
        allow_single = _single_field_names(reference.alleles[gene])
        normal = [[normalise_hla_allele(v, gene, allow_single_field=allow_single) for v in original[col]] for col in pair]
        unique = sorted({name for slot in normal for name in slot if name is not None})
        labels, sequences = [""], [np.full(len(reference.positions[gene]), "?", dtype="U1")]
        codes = {None: 0}
        for name in unique:
            row = reference.resolve(name)
            if row is None and fallback_to_two_field:
                match = re.fullmatch(r"(.+\*)(\d+(?::\d+)*)([NLSCAQ]?)", name)
                fields = match[2].split(":")
                if len(fields) > 2:
                    target = match[1] + ":".join(fields[:2]) + match[3]
                    row = reference.resolve(target)
                    if row is not None:
                        issue = {"gene": gene, "allele": name, "mapped_as": target,
                                 "reason": "two-field consensus fallback"}
                        mapping.issues.append(issue)
                        fallbacks.append(issue)
            if row is None:
                codes[name] = 0
                mapping.issues.append({"gene": gene, "allele": name, "reason": "allele not found in reference"})
            else:
                codes[name] = len(labels); labels.append(name); sequences.append(row)
        matrix = np.stack(sequences)
        ids = np.column_stack([[codes[name] for name in slot] for slot in normal]).astype(np.int32)
        position_info = reference.positions[gene].copy(deep=True)
        values, markers = [], []
        for col, base in enumerate(position_info.MARKER_BASE):
            # Expand only one site at a time; do not duplicate each individual's
            # two full protein sequences in memory.
            first, second = matrix[ids[:, 0], col], matrix[ids[:, 1], col]
            complete = (first != "?") & (second != "?")
            unresolved += int((~complete).sum())
            states = sorted(set(matrix[:, col]) - {"?"})
            for state in states:
                dosage = ((first == state).astype(np.float32) + (second == state)).astype(np.float32)
                dosage[~complete] = np.nan
                markers.append(base + "_" + state); values.append(dosage)
        if markers:
            frames.append(pd.DataFrame(np.column_stack(values), index=original.index, columns=markers))
        matrix.flags.writeable = False; ids.flags.writeable = False
        mapping.alleles[gene], mapping.matrices[gene], mapping.allele_ids[gene], mapping.positions[gene] = labels, matrix, ids, position_info
    absent = sum(issue["reason"] == "allele not found in reference" for issue in mapping.issues)
    if absent:# or unresolved:
        message = (f"HLA mapping: {absent} distinct allele calls absent from the reference; ")
                   #f"{unresolved} sample-position pairs unresolved. Affected dosages remain missing.")
        if strict: raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    if fallbacks:
        warnings.warn(
            f"HLA mapping: {len(fallbacks)} distinct unavailable high-resolution calls mapped using "
            "two-field consensus, preserving expression suffixes. This is inferred sequence; "
            "conflicting or unknown positions remain missing. See hla_mapping.issues for details.",
            UserWarning, stacklevel=2,
        )
    result = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=original.index)
    result.attrs["hla_mapping"] = mapping
    result.attrs["reference_metadata"] = dict(metadata)
    if save_path is not None:
        result.to_csv(save_path, sep="\t", index=True, index_label="IID", na_rep="NA")
        mapping.dosage_file_sha256 = _sha(save_path)
        mapping.save(str(save_path) + ".phase.npz")
    return result if output == "dataframe" else read_aa_dosage(result)


def read_aa_dosage(fileloc, *, phase_path=None):
    """Read mapped hard diploid residue dosages into an HLAdata object without QC.

    Parameters
    ----------
    fileloc : pandas.DataFrame or str or pathlib.Path
        Sample-by-residue dosage DataFrame or tab-separated file with IID in
        the first column. Marker columns must follow the mapping format
        AA_GENE_POSITION_COORD_exonEXON_RESIDUE. Coordinates/exons may be NA;
        position labels may include insertions. Values must be 0, 1, 2 or missing.
    phase_path : str or pathlib.Path, optional
        Mapping sidecar for file input. Defaults to ``<fileloc>.phase.npz`` if
        that file exists. An explicitly requested missing sidecar raises.
        For DataFrame input, uses attrs["hla_mapping"] instead; phase_path
        is not consulted.

    Returns
    -------
    HLAdata
        AA data with exon-aware identifiers and nullable coordinate metadata.
        If mapping is available, also restores hla_mapping, hla_alleles and
        reference_metadata. Without it, hla_mapping is None and phase is unavailable.

    Raises
    ------
    ValueError
        For invalid markers/dosages, duplicate columns, sample mismatch, or a
        dosage file whose fingerprint differs from its phase sidecar.
    FileNotFoundError
        If an explicit file/sidecar cannot be found.

    Notes
    -----
    Preserves string IIDs and does not mutate a supplied DataFrame. TSV input
    alone cannot recover allele phase or infer an assembly from coordinates.
    This reader accepts mapped 0/1/2 counts, not arbitrary fractional imputations.
    """
    if isinstance(fileloc, pd.DataFrame):
        mapping = fileloc.attrs.get("hla_mapping")
        # Keep phase on HLAdata itself, not on every downstream genotype frame.
        data = pd.DataFrame(fileloc.to_numpy(copy=True), index=fileloc.index.copy(), columns=fileloc.columns.copy())
    else:
        data = pd.read_csv(fileloc, sep="\t", index_col=0, converters={0: str})
        sidecar = Path(phase_path or (str(fileloc) + ".phase.npz"))
        if phase_path is not None and not sidecar.exists(): raise FileNotFoundError(sidecar)
        mapping = HLAAlleleMapping.load(sidecar) if sidecar.exists() else None
        if mapping is not None and mapping.dosage_file_sha256 != _sha(fileloc):
            raise ValueError("Dosage file differs from its allele-phase sidecar fingerprint.")
    data.index = sample_index(data.index, source="AA dosages")
    if not data.columns.is_unique: raise ValueError("AA residue marker columns must be unique.")
    parsed = [parse_aa_marker(marker) for marker in data.columns]
    data.attrs = {}
    values = data.apply(pd.to_numeric, errors="raise")
    values.attrs = {}
    observed = values.to_numpy(dtype=float)
    if np.isinf(observed).any() or not np.isin(observed[~np.isnan(observed)], [0, 1, 2]).all():
        raise ValueError("Mapped AA dosages must be 0, 1, 2 or missing.")
    if mapping is not None and not mapping.calls.index.equals(data.index):
        raise ValueError("Phase sidecar sample IDs/order do not match the dosage file.")
    # Reuse the existing marker annotation and HLAdata constructor, overriding
    # only the richer exon-aware IDs/metadata introduced by this reader.
    from .io import _add_variant_info
    annotated = _add_variant_info(values.T.copy()) if len(parsed) else pd.DataFrame(index=data.columns, columns=[*data.index, "AA_ID", "TYPE", "GENE", "AA_POS", "POS"])
    for key in ("AA_ID", "GENE", "AA_POS", "POS"):
        annotated[key] = [row[key] for row in parsed]
    annotated["TYPE"] = "AA"; annotated["alleleA"] = "P"; annotated["alleleB"] = "A"
    result = HLAdata(annotated, "softcall", load_types=("AA",))
    result.AA.data.drop(columns=["alleleA", "alleleB"], inplace=True)
    result.AA.info["EXON"] = pd.array([row["EXON"] for row in parsed], dtype="Int64")
    result.AA.info["POS"] = pd.array([row["POS"] for row in parsed], dtype="Int64")
    result.hla_mapping = mapping
    if mapping is not None:
        result.hla_alleles = mapping.calls.copy(deep=True)
        result.reference_metadata = dict(mapping.metadata)
    else:
        result.reference_metadata = {"assembly": None, "phase_available": False}
    return result


def aa_motif_dosages(data, gene, positions):
    """Calculate within-gene motif dosages from preserved allele pairs.

    Parameters
    ----------
    data : HLAdata, pandas.DataFrame or HLAAlleleMapping
        Mapped data with hla_mapping (attribute or DataFrame attrs), or the
        mapping itself. A plain dosage table without phase is insufficient.
    gene : str
        One mapped locus, optionally prefixed with HLA-. Cross-gene motifs
        are not supported.
    positions : iterable of str or int
        Nonempty, unique AA position labels in the desired motif order.
        Use strings for insertion labels such as ``"52i1"``.

    Returns
    -------
    pandas.DataFrame
        Samples by observed motif, with 0/1/2 dosages. Residues are joined by
        ``|`` in requested position order. If either allele is unknown at any
        selected position, all motif dosages for that sample are NaN. Attributes
        include gene, positions, reference_version, allele_groups and N_unresolved.

    Raises
    ------
    ValueError
        If phase is unavailable, the locus/positions are unknown, or positions
        are empty or duplicated.

    Notes
    -----
    Motif assignments are cached per gene, reference version and ordered position
    selection. No statistical test, frequency filtering or cross-gene phase
    inference is performed.

    Examples
    --------
    >>> motifs = aa_motif_dosages(hladat, "DRB1", (11, 13, 71))
    """
    mapping = data if isinstance(data, HLAAlleleMapping) else (data.attrs.get("hla_mapping") if isinstance(data, pd.DataFrame) else getattr(data, "hla_mapping", None))
    if mapping is None: raise ValueError("Original allele-pair mapping is required; residue dosages alone do not establish phase.")
    return mapping.motif_dosages(gene, positions)
