"""Map diploid HLA allele calls through a saved local reference, retaining phase."""
from dataclasses import dataclass, field
from io import StringIO
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from .hla_reference import HLAReference, load_hla_reference, normalise_hla_allele, _normalise_gene, _sha
from .HLAdat import HLAdata
from ._mapping_io import MappingReport
from ._motifs import motif_from_sequences, representative_motif
from ..stats.preprocess import sample_index

_MARKER = re.compile(r"^(AA_([A-Z0-9]+)_(-?\d+(?:i[1-9]\d*)?)_(NA|\d+)(?:_exon(NA|\d+))?)_([ACDEFGHIKLMNPQRSTVWYX.*]+)$")


def parse_aa_marker(marker):
    """Parse residue, gap '.' and alignment '*' markers; NA stays missing."""
    match = _MARKER.fullmatch(str(marker))
    if not match or (match[4] != "NA" and int(match[4]) <= 0):
        raise ValueError(f"Invalid mapped AA marker {marker!r}.")
    return {"AA_ID": f"AA_{match[2]}_{match[3]}_{match[4]}", "GENE": match[2], "AA_POS": match[3],
            "POS": None if match[4] == "NA" else int(match[4]),
            "EXON": None if match[5] in {None, "NA"} else int(match[5]), "RESIDUE": match[6]}


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
    motif_sources: dict = field(default_factory=dict, repr=False)

    def motif_dosages(self, gene, positions):
        """Calculate within-gene motif counts from mapped allele pairs.

        Parameters
        ----------
        gene : str
            HLA gene to map.
        positions : iterable of str or int
            Amino-acid positions in the required motif order.

        Returns
        -------
        pandas.DataFrame
            Sample-by-motif counts with unmapped contributions in attrs.

        Raises
        ------
        ValueError
            For an unmapped gene, unknown positions or empty/duplicate positions.

        Notes
        -----
        Mapped contributions are retained when the partner is unavailable.
        Samples with no usable mapping remain missing.
        """
        gene = _normalise_gene(gene)
        if gene not in self.matrices: raise ValueError(f"Gene {gene!r} was not mapped.")
        positions = tuple(str(p) for p in positions)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("Select a nonempty set of unique AA positions.")
        lookup = {str(pos): i for i, pos in enumerate(self.positions[gene]["AA_POS"])}
        missing = set(positions) - set(lookup)
        if missing: raise ValueError(f"Unknown {gene} positions: {sorted(missing)}")
        key = (gene, self.metadata["reference_version"], positions, "compact-v2")
        if key not in self.motif_cache:
            selected = self.matrices[gene][:, [lookup[p] for p in positions]]
            motifs = [None if "?" in row else "".join(row) for row in selected]
            if gene in self.motif_sources:
                indices = [lookup[p] for p in positions]
                motifs = []
                for source in self.motif_sources[gene]:
                    if source is None:
                        motifs.append(None)
                        continue
                    rows = np.asarray(source)[:, indices]
                    motifs.append(motif_from_sequences(rows))
            elif any(len(state) > 1 for state in selected.flat):
                raise ValueError("Ambiguous mapping lacks candidate motif sequences; remap allele calls.")
            motifs = [representative_motif(label) for label in motifs]
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
        missing = ~(ids >= 0).any(axis=1)
        dosage[missing, :] = np.nan
        frame = pd.DataFrame(dosage, index=self.calls.index.copy(), columns=labels)
        frame.attrs.update({"gene": gene, "positions": positions, "reference_version": self.metadata["reference_version"],
                            "allele_groups": {k: list(v) for k, v in groups.items()},
                            "unmapped_dosage": dict(zip(self.calls.index, (ids < 0).sum(axis=1).astype(float).tolist())),
                            "N_unresolved": int((~complete).sum()),
                            "N_partial": int((~complete & ~missing).sum()),
                            "N_missing": int(missing.sum())})
        return frame

    def save(self, path):
        """Save allele calls and mapping data to an NPZ file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output NPZ file. Existing files are overwritten.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If the output file cannot be written.

        Notes
        -----
        Motif caches are omitted. Reload with HLAAlleleMapping.load.
        """
        payload = {"metadata": np.array(json.dumps({"metadata": self.metadata, "issues": self.issues,
            "calls": json.loads(self.calls.to_json(orient="split")),
            "multi_columns": isinstance(self.calls.columns, pd.MultiIndex),
            "column_names": list(self.calls.columns.names), "genes": list(self.alleles),
            "dosage_file_sha256": self.dosage_file_sha256,
            "motif_sources": self.motif_sources}))}
        for gene in self.alleles:
            payload[f"{gene}_alleles"] = np.asarray(self.alleles[gene])
            payload[f"{gene}_matrix"] = self.matrices[gene]
            payload[f"{gene}_ids"] = self.allele_ids[gene]
            payload[f"{gene}_positions"] = np.array(self.positions[gene].to_json(orient="table", index=False))
        with Path(path).open("wb") as stream: np.savez_compressed(stream, **payload)

    @classmethod
    def load(cls, path):
        """Load allele calls and mapping data from an NPZ file.

        Parameters
        ----------
        path : str or pathlib.Path
            File saved by HLAAlleleMapping.save.

        Returns
        -------
        HLAAlleleMapping
            Allele calls, sequences and within-gene allele pairs.

        Raises
        ------
        ValueError
            For inconsistent saved array dimensions or allele IDs.
        OSError
            If the file cannot be read.

        Notes
        -----
        The dosage-file fingerprint is checked by read_aa_dosage.
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
                   dosage_file_sha256=header.get("dosage_file_sha256"),
                   motif_sources=header.get("motif_sources", {}))


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


def map_hla_alleles(calls, reference, *, allele_columns=None, output="hladata",
                    save_path=None, strict=False, fallback_to_two_field=True,
                    simpleQC=True, allele_filter=0.005, verbose=True):
    """Map paired HLA allele calls to amino-acid dosages.

    Parameters
    ----------
    calls : pandas.DataFrame
        Samples on rows, with paired allele columns such as A_1 and A_2.
    reference : HLAReference or str or pathlib.Path
        HLA reference object or path to a saved reference file.
    allele_columns : mapping or None, default None
        Gene-to-column-pair mapping. None discovers GENE_1/GENE_2 pairs.
    output : {'hladata', 'dataframe'}, default 'hladata'
        Output format.
    save_path : str or pathlib.Path or None, default None
        Optional dosage TSV path. Also saves an allele-mapping sidecar.
    strict : bool, default False
        Raise an error if a call or sample-position remains unresolved.
    fallback_to_two_field : bool, default True
        Try a two-field match when a higher-resolution allele is unavailable.
    simpleQC : bool, default True
        Apply the simple MAF filter to mapped output states.
    allele_filter : float, default 0.005
        Minimum allele frequency; states must also be below 1 - allele_filter.
    verbose : bool, default True
        Print mapping, QC and timing summaries.

    Returns
    -------
    HLAdata or pandas.DataFrame
        Amino-acid dosages with the within-gene allele-pair mapping attached.

    Raises
    ------
    TypeError
        If calls is not a DataFrame.
    ValueError
        For invalid IDs, allele columns/names, output format or MAF threshold.
        Unresolved mappings also raise when strict=True.
    OSError
        If a reference file cannot be read or output cannot be written.

    Notes
    -----
    Retains the two allele slots within each gene. Phase between genes is
    not inferred.
    """
    report = MappingReport("MAPPING HLA ALLELES TO AMINO ACIDS", simpleQC=simpleQC,
                           allele_filter=allele_filter, verbose=verbose, output=output)
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
    n_hla = 0
    for gene, pair in columns.items():
        allow_single = True
        normal = [[normalise_hla_allele(v, gene, allow_single_field=allow_single) for v in original[col]] for col in pair]
        unique = sorted({name for slot in normal for name in slot if name is not None})
        n_hla += len(unique)
        labels, sequences = [""], [np.full(len(reference.positions[gene]), "?", dtype="U1")]
        codes = {None: 0}
        sources = [None]
        for name in unique:
            target = name
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
                match = re.fullmatch(r"(.+\*)(\d+(?::\d+)*)([NLSCAQ]?)", target)
                prefix, fields, suffix = match.groups()
                base = prefix + fields
                candidates = [i for i, allele in enumerate(reference.alleles[gene])
                              if (allele == base or allele.startswith(base + ":") or allele == base + suffix)
                              and (not suffix or allele.endswith(suffix))]
                if not suffix and len(fields.split(":")) == 2:
                    candidates = [i for i in candidates if not reference.alleles[gene][i].endswith("N")]
                sources.append(reference.matrices[gene][candidates].tolist())
        mapping.motif_sources[gene] = sources
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
    if absent or unresolved:
        message = (f"HLA mapping: {absent} distinct allele calls absent from the reference ")
                #   f"{unresolved} sample-position pairs unresolved. Affected dosages remain missing.")
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    if fallbacks:
        warnings.warn(
            f"HLA mapping: {len(fallbacks)} distinct unavailable high-resolution calls mapped using "
            "two-field consensus, preserving expression suffixes. This is inferred sequence; "
            "conflicting residues form combined states; unknown positions remain missing. "
            "See hla_mapping.issues for details.",
            UserWarning, stacklevel=2,
        )
    result = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=original.index)
    result.attrs["hla_mapping"] = mapping
    result.attrs["reference_metadata"] = dict(metadata)
    result = report.process(result, n_hla)
    if save_path is not None:
        result.to_csv(save_path, sep="\t", index=True, index_label="IID", na_rep="NA")
        mapping.dosage_file_sha256 = _sha(save_path)
        mapping.save(str(save_path) + ".phase.npz")
    converted = result if output == "dataframe" else read_aa_dosage(result)
    report.finish(result)
    return converted


def read_aa_dosage(fileloc, *, phase_path=None):
    """Read amino-acid dosages into an HLAdata object.

    Parameters
    ----------
    fileloc : pandas.DataFrame or str or pathlib.Path
        Sample-by-residue dosage DataFrame or TSV file indexed by IID.
    phase_path : str or pathlib.Path or None, default None
        Optional allele-mapping sidecar. File input uses <fileloc>.phase.npz
        when present; DataFrame input uses its hla_mapping attribute.

    Returns
    -------
    HLAdata
        Amino-acid dosages and marker annotations, with allele mapping if present.

    Raises
    ------
    ValueError
        For invalid markers, dosages, sample IDs or mismatched sidecar data.
    OSError
        If the dosage file or requested sidecar cannot be read.

    Notes
    -----
    Dosage values are preserved. Allele phase is available only when an allele
    mapping is attached or loaded.
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
    if np.isinf(observed).any() or (observed < 0).any():
        raise ValueError("Mapped AA dosages must be finite and nonnegative, or missing.")
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
    #result.AA.info["EXON"] = pd.array([row["EXON"] for row in parsed], dtype="Int64")
    result.AA.info["POS"] = pd.array([row["POS"] for row in parsed], dtype="Int64")
    result.hla_mapping = mapping
    if mapping is not None:
        result.hla_alleles = mapping.calls.copy(deep=True)
        result.reference_metadata = dict(mapping.metadata)
    else:
        result.reference_metadata = {"assembly": None, "phase_available": False}
    return result


def aa_motif_dosages(data, gene, positions, *, output="hladata", simpleQC=True,
                     allele_filter=0.005, verbose=True):
    """Calculate within-gene motif dosages from mapped allele pairs.

    Parameters
    ----------
    data : HLAdata, pandas.DataFrame or HLAAlleleMapping
        Mapped data containing the original allele-pair mapping.
    gene : str
        HLA gene to map.
    positions : iterable of str or int
        Amino-acid positions in the required motif order.
    output : {'hladata', 'dataframe'}, default 'hladata'
        Return HLAdata for analysis, or a sample-by-motif dosage DataFrame.
    simpleQC : bool, default True
        Apply the simple MAF filter to mapped output states.
    allele_filter : float, default 0.005
        Minimum allele frequency; states must also be below 1 - allele_filter.
    verbose : bool, default True
        Print mapping, QC and timing summaries.

    Returns
    -------
    HLAdata or pandas.DataFrame
        HLAdata with motif genotypes, annotations and the original allele
        mapping, or a sample-by-motif DataFrame with mapping metadata in attrs.

    Raises
    ------
    ValueError
        For missing allele mapping, invalid gene/positions, output format
        or MAF threshold.

    Notes
    -----
    Uses the stored within-gene allele pairs. Mapped contributions are retained
    when another allele cannot be mapped.
    """
    report = MappingReport("MAPPING HLA ALLELE PAIRS TO MOTIFS", simpleQC=simpleQC,
                           allele_filter=allele_filter, verbose=verbose, output=output)
    mapping = data if isinstance(data, HLAAlleleMapping) else (data.attrs.get("hla_mapping") if isinstance(data, pd.DataFrame) else getattr(data, "hla_mapping", None))
    if mapping is None: raise ValueError("Original allele-pair mapping is required; residue dosages alone do not establish phase.")
    frame = mapping.motif_dosages(gene, positions)
    locus = frame.attrs["gene"]
    frame = report.process(frame, len(mapping.alleles[locus]) - 1)
    if output == "dataframe":
        converted = frame
    else:
        from .hla_dosage import _as_hladata
        converted = _as_hladata(frame, motifs=True, mapping=mapping)
    report.finish(frame, motifs=True)
    return converted
