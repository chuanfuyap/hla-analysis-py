"""Local IPD protein references and once-only assembly/CDS coordinate mapping.

No sequence data are distributed with this module. Inputs are official local
IPD alignments and, for coordinates, local Ensembl lookup/CDS JSON snapshots.
"""
from dataclasses import dataclass, field
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import warnings
import time

import numpy as np
import pandas as pd

GENES = ("A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1")
GENOMICS_URL = "https://www.ebi.ac.uk/ipd/imgt/hla/about/help/genomics/"
SCHEMA = 1
_ALLELE = re.compile(r"^([A-Z0-9]+)\*(\d+(?::\d+)*)([NLSCAQ]?)$")
_ROW = re.compile(r"^\s*([A-Z0-9]+\*\d+(?::\d+)*[NLSCAQ]?)\s+([A-Z.*?\- |]+?)\s*$")


def _normalise_gene(gene):
    gene = str(gene).strip().upper().removeprefix("HLA-").removeprefix("HLA_")
    if not re.fullmatch(r"[A-Z][A-Z0-9]*", gene):
        raise ValueError(f"Invalid locus name {gene!r}.")
    return gene


def _single_field_names(names):
    return any(":" not in name.split("*", 1)[1] for name in names)


def available_hla_genes(alignment_dir):
    """Discover loci in local per-locus IPD protein alignment files.

    Parameters
    ----------
    alignment_dir : str or pathlib.Path
        Directory containing official ``GENE_prot.txt`` files.

    Returns
    -------
    tuple of str
        Sorted locus names with matching allele records. Combined ClassI/DRB
        files and loci represented only by nucleotide files are excluded.

    Notes
    -----
    This performs local discovery only; no files are downloaded. Pass the result
    as ``genes`` to ``build_hla_reference``, or use ``genes="all"`` there.
    """
    genes = []
    for path in sorted(Path(alignment_dir).glob("*_prot.txt")):
        gene = _normalise_gene(path.name.removesuffix("_prot.txt"))
        if gene in {"DRB", "CLASSI"}: continue
        with path.open() as stream:
            for line in stream:
                match = _ROW.fullmatch(line.rstrip("\n"))
                if match and match[1].split("*", 1)[0] == gene:
                    genes.append(gene); break
    return tuple(genes)


def normalise_hla_allele(value, gene=None, *, allow_single_field=False):
    """Normalise an HLA allele name.

    Parameters
    ----------
    value : str or scalar missing value
        HLA allele name. Missing values return None.
    gene : str or None, default None
        Expected gene. Allows gene-free allele names when supplied.
    allow_single_field : bool, default False
        Accept single-field names such as A*01.

    Returns
    -------
    str or None
        Normalised name retaining its resolution and expression suffix.

    Raises
    ------
    ValueError
        For invalid names, conflicting genes or unsupported allele notation.
    """
    if value is None or pd.isna(value) or str(value).strip().upper() in {"", ".", "NA", "NAN", "NONE", "0", "0000"}:
        return None
    text = str(value).strip().upper().removeprefix("HLA-").removeprefix("HLA_")
    if gene:
        gene = _normalise_gene(gene)
        if "*" not in text:
            text = text.removeprefix(gene + "_")
            if text.startswith(gene) and text[len(gene):].isdigit():
                text = text[len(gene):]
            text = gene + "*" + text
    if "*" in text:
        locus, code = text.split("*", 1)
        match = re.fullmatch(r"(\d{4})([NLSCAQ]?)", code)
        if match:
            code = ":".join(match[1][i:i+2] for i in range(0, len(match[1]), 2)) + match[2]
            text = locus + "*" + code
    match = _ALLELE.fullmatch(text)
    if (not match or (gene and match[1] != gene)
            or (not allow_single_field and ":" not in match[2])):
        raise ValueError(f"Invalid or unsupported HLA allele {value!r} for gene {gene!r}.")
    _normalise_gene(match[1])
    return match[1] + "*" + ":".join(part.zfill(2) for part in match[2].split(":")) + match[3]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _alignment(path, gene, protein=True):
    """Read an IPD alignment and resolve reference-relative sequence symbols."""
    blocks, current, header, releases = [], [], "", set()
    prefix = "Prot" if protein else "cDNA"
    for line in Path(path).read_text().splitlines():
        version = re.search(r"IPD-IMGT/HLA\s+(\d+\.\d+\.\d+)", line)
        if version:
            releases.add(version[1])
        if re.match(r"\s*" + prefix + r"\s", line):
            if current:
                blocks.append((header, current))
            current, header = [], line
        match = _ROW.fullmatch(line)
        if match:
            current.append((match[1], re.sub(r"[\s|]", "", match[2]), line, match.start(2)))
    if current:
        blocks.append((header, current))
    if not blocks or len(releases) != 1:
        raise ValueError(f"{path}: expected a versioned IPD {prefix} alignment.")
    reference = blocks[0][1][0][0]
    names = sorted({row[0] for _, rows in blocks for row in rows if row[0].split("*")[0] == gene})
    if not names:
        raise ValueError(f"{path}: no {gene} allele sequences.")
    chunks = {name: [] for name in names}
    ref_chunks, mature = [], None
    offset = 0
    for block_header, rows in blocks:
        records = {name: seq for name, seq, _, _ in rows}
        if len(records) != len(rows):
            raise ValueError(f"{path}: duplicate allele rows within an alignment block.")
        # Official files may have a final extension block containing only an
        # elongated allele. There is no global-reference identity to copy there.
        extension = reference not in records
        ref = records.get(reference, rows[0][1])
        if extension:
            ref = ref.ljust(max(len(row[1]) for row in rows), "?")
        if "-" in ref:
            raise ValueError(f"{path}: reference contains unresolved identity symbols.")
        width = len(ref)
        if protein and mature is None:
            anchor = re.search(r"(?<![-\d])1(?!\d)", block_header)
            if anchor:
                _, _, raw, start = rows[0]
                local = len(re.sub(r"[\s|]", "", raw[start:anchor.start()]))
                if local >= width or ref[local] == ".":
                    raise ValueError(f"{path}: cannot locate mature protein position 1.")
                mature = offset + local
        records = {name: seq for name, seq, _, _ in rows}
        for name in names:
            seq = records.get(name, "").ljust(width, "*")
            if len(seq) != width:
                raise ValueError(f"{path}: inconsistent block width for {name}.")
            decoded = "".join(("?" if extension else ref[i]) if c == "-" else c for i, c in enumerate(seq))
            chunks[name].append(decoded.replace("*", "?") if not protein else
                                "".join(c if c in "ACDEFGHIKLMNPQRSTVWYX.*" else "?" for c in decoded))
        ref_chunks.append(ref)
        offset += width
    matrix = np.array([list("".join(chunks[name])) for name in names], dtype="U1")
    if protein:
        for row in matrix:
            stops = np.flatnonzero(row == "X")
            if len(stops):
                tail = row[stops[0]+1:]
                # Preserve alignment '*' states; other residues after a stop
                # cannot be interpreted as translated amino acids.
                tail[tail != "*"] = "?"
        if mature is None:
            raise ValueError(f"{path}: a position-1 ruler anchor is required; negative ticks alone are insufficient.")
        ref = "".join(ref_chunks)
        before = sum(c != "." for c in ref[:mature])
        positions, number, insertion = [], -before if before else 1, 0
        for c in ref:
            if c == ".":
                insertion += 1
                previous = number - 1 if number != 1 else -1
                positions.append(f"{previous}i{insertion}")
            else:
                insertion = 0
                positions.append(str(number))
                number += 1
                if number == 0:
                    number = 1
    else:
        positions = None
    return names, matrix, positions, reference, releases.pop()


class _Table(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = None; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.row = []
        if tag in {"td", "th"}: self.cell = []
    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            if self.row is not None: self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row); self.row = None


def _genomic_table(path):
    parser = _Table(); parser.feed(Path(path).read_text())
    records = {}
    for row in parser.rows:
        if len(row) < 6:
            continue
        try:
            gene = _normalise_gene(row[0])
        except ValueError:
            continue
        if not any(re.match(r"(?:HLA-)?" + re.escape(gene) + r"\*\d", value) for value in row[4:6]):
            continue
        coordinate = re.search(r"([^: ]+):([\d,]+)", row[3])
        records[gene] = {"declared_reference_allele": row[4], "ipd_reference_allele": row[5],
                         "ipd_reported_atg": int(coordinate[2].replace(",", "")) if coordinate else None,
                         "ipd_location": row[2], "ensembl_gene": row[1],
                         "ipd_reported_strand": -1 if "reverse" in row[2] else (1 if "forward" in row[2] else None)}
    if not records:
        raise ValueError("No HLA-region locus rows found in the official genomic table snapshot.")
    return records


# Standard genetic code, used only to validate the supplied assembly CDS.
_CODONS = dict(zip((a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"),
                  "FFLLSSSSYYXXCCXWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"))


def _translation(cds):
    if len(cds) % 3 or set(cds) - set("ACGT"):
        raise ValueError("Assembly CDS must contain complete, unambiguous codons.")
    return "".join(_CODONS[cds[i:i+3]] for i in range(0, len(cds), 3)).removesuffix("X")


def _coordinates(gene, names, matrix, positions, nucleotide_file, annotation_dir, table, transcript):
    lookup_path = Path(annotation_dir) / f"{gene}.json"
    cds_path = Path(annotation_dir) / f"{gene}.cds.json"
    lookup = json.loads(lookup_path.read_text())
    sequence = json.loads(cds_path.read_text())
    if str(lookup.get("seq_region_name")) != "6":
        raise ValueError(f"{gene}: HLA-region coordinates must use the GRCh38 primary chromosome 6 sequence.")
    if lookup.get("assembly_name") != "GRCh38" or lookup.get("display_name") not in {gene, "HLA-" + gene}:
        raise ValueError(f"{gene}: annotation must identify this gene on GRCh38.")
    candidates = [t for t in lookup.get("Transcript", []) if t.get("Translation")]
    if transcript:
        candidates = [t for t in candidates if t["id"].split(".")[0] == transcript.split(".")[0]
                      and ("." not in transcript or str(t.get("version")) == transcript.split(".", 1)[1])]
    else:
        candidates = [t for t in candidates if t.get("is_canonical")]
    if len(candidates) != 1:
        raise ValueError(f"{gene}: select exactly one protein-coding annotated transcript.")
    transcript = candidates[0]
    if sequence.get("id", "").split(".")[0] != transcript["id"].split(".")[0]:
        raise ValueError(f"{gene}: CDS JSON does not match the selected transcript.")
    if sequence.get("version") is not None and transcript.get("version") is not None and sequence["version"] != transcript["version"]:
        raise ValueError(f"{gene}: annotation and CDS transcript versions differ.")
    cds = sequence["seq"].upper()
    protein = _translation(cds)
    n_names, n_matrix, _, n_reference, n_release = _alignment(nucleotide_file, gene, protein=False)
    n_matches = {name for name, row in zip(n_names, n_matrix) if "".join(row).replace(".", "").rstrip("?") == cds}
    protein_matches = [i for i, row in enumerate(matrix) if "".join(row).replace(".", "").rstrip("?").removesuffix("X") == protein]
    exact = [i for i in protein_matches if names[i] in n_matches]
    match_rows = exact or protein_matches
    if not match_rows:
        raise ValueError(f"{gene}: assembly CDS translation does not match any complete IPD protein; coordinates cannot be validated.")
    strand = int(transcript["strand"])
    if strand not in {-1, 1}:
        raise ValueError(f"{gene}: invalid annotation strand.")
    translation = transcript["Translation"]
    low, high = sorted([int(translation["start"]), int(translation["end"])])
    exons = sorted(transcript["Exon"], key=lambda e: e["start"], reverse=strand == -1)
    genomic, exon_numbers = [], []
    for number, exon in enumerate(exons, 1):
        start, end = max(low, int(exon["start"])), min(high, int(exon["end"]))
        if end < start:
            continue
        bases = list(range(start, end+1)) if strand == 1 else list(range(end, start-1, -1))
        genomic.extend(bases); exon_numbers.extend([number] * len(bases))
    if len(genomic) not in {len(cds), len(cds)-3} or len(genomic) < len(protein)*3:
        raise ValueError(f"{gene}: annotated coding length does not match the assembly CDS.")
    assignments = []
    for i in match_rows:
        row, codon, indices = matrix[i], 0, []
        for state in row:
            indices.append(None if state == "." else codon)
            if state != ".": codon += 1
        assignments.append(indices)
    output = []
    for col in range(len(positions)):
        agreed = {row[col] for row in assignments}
        codon = next(iter(agreed)) if len(agreed) == 1 else None
        if codon is None or codon >= len(protein):
            # An assembly-gap insertion can still have a known exon when its
            # two flanking codons agree. Never assign a fake genomic base.
            exon = None
            if agreed == {None}:
                left = next((j for j in range(col-1, -1, -1) if all(a[j] is not None and a[j] < len(protein) for a in assignments)), None)
                right = next((j for j in range(col+1, len(positions)) if all(a[j] is not None and a[j] < len(protein) for a in assignments)), None)
                if left is not None and right is not None:
                    flanks = {exon_numbers[3*a[j]+1] for a in assignments for j in (left, right)}
                    if len(flanks) == 1: exon = flanks.pop()
            output.append((None, exon, "absent_or_ambiguous_in_assembly"))
        else:
            middle = 3 * codon + 1
            output.append((genomic[middle], exon_numbers[middle], "validated"))
    matched_names = [names[i] for i in match_rows]
    declared = table[gene]["declared_reference_allele"]
    discrepancy = (table[gene]["ipd_reported_atg"] != genomic[0] or declared not in matched_names
                   or table[gene]["ipd_reported_strand"] != strand)
    if discrepancy:
        warnings.warn(f"{gene}: IPD genomic table differs from validated assembly CDS/annotations; "
                      "using the assembly-validated coordinates and recording both sources.", UserWarning, stacklevel=3)
    metadata = {**table[gene], "transcript": transcript["id"], "transcript_version": transcript.get("version"), "chromosome": str(lookup["seq_region_name"]),
                "strand": strand, "validated_atg": genomic[0], "coordinate_alleles": matched_names,
                "coordinate_match": "CDS_identity" if exact else "protein_identity",
                "ipd_table_discrepancy": discrepancy, "nucleotide_release": n_release, "nucleotide_alignment_reference": n_reference,
                "exon_numbering": "selected transcript exon order in transcription direction"}
    return output, metadata, [lookup_path, cds_path, Path(nucleotide_file)]


def _reference_identity(alleles, matrices, positions, metadata):
    info = {k: v for k, v in metadata.items() if k != "reference_version"}
    info["source_sha256"] = sorted(info["source_sha256"].values())
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode())
    for gene in metadata["genes"]:
        digest.update(json.dumps(alleles[gene]).encode())
        digest.update(matrices[gene].tobytes())
        digest.update(positions[gene].to_json(orient="records").encode())
    return digest.hexdigest()


@dataclass(eq=False)
class HLAReference:
    """Allele-by-alignment-position matrices plus precomputed marker metadata."""
    alleles: dict
    matrices: dict
    positions: dict
    metadata: dict
    _resolved: dict = field(default_factory=dict, repr=False)

    @property
    def version(self):
        return self.metadata["reference_version"]

    def resolve(self, allele):
        """Resolve an HLA allele to its amino-acid states.

        Parameters
        ----------
        allele : str or scalar missing value
            Gene-qualified HLA allele name.

        Returns
        -------
        numpy.ndarray or None
            Amino-acid states in reference position order, or None if no allele
            matches.

        Raises
        ------
        ValueError
            If the allele name is invalid.

        Notes
        -----
        Low-resolution matches may produce combined states such as LVY.
        '*' and '.' are mapped states; '?' denotes unavailable sequence.
        """
        allele = normalise_hla_allele(allele, allow_single_field=True)
        if allele is None: return None
        if allele in self._resolved: return self._resolved[allele]
        gene = allele.split("*")[0]
        if gene not in self.alleles: return None
        names = self.alleles[gene]
        match = _ALLELE.fullmatch(allele)
        prefix = gene + "*" + match[2]
        ids = [i for i, name in enumerate(names)
               if (name == prefix or name.startswith(prefix + ":") or name == prefix + match[3])
               and (not match[3] or name.endswith(match[3]))]
        if not ids: return None
        # Exclude null alleles when resolving two-field HLA calls.
        if not match[3] and len(match[2].split(":")) == 2:
            ids = [i for i in ids if not names[i].endswith("N")]
            if not ids:
                return None
        sequences = self.matrices[gene][ids]

        # Variable-width strings preserve ambiguous states such as LVY.
        result = np.asarray([
            "".join(sorted(set(column[column != "?"]))) if (column != "?").any() else "?"
            for column in sequences.T
        ])
        result.flags.writeable = False
        self._resolved[allele] = result
        return result

    def resolve_motif(self, allele, indices):
        """Resolve an HLA allele to candidate motifs.

        Parameters
        ----------
        allele : str or scalar missing value
            Gene-qualified HLA allele name.
        indices : iterable of int
            Alignment column indices in the required motif order.

        Returns
        -------
        str or None
            A compact motif or candidate set such as {LAR;VGR}. Returns None
            when no matching record has states at all selected positions.

        Raises
        ------
        ValueError
            If the allele name is invalid.
        IndexError
            If an alignment index is out of range.

        Notes
        -----
        Uses matching reference records with all selected states.
        '*' and '.' are valid motif states.
        """
        allele = normalise_hla_allele(allele, allow_single_field=True)
        if allele is None:
            return None
        match = _ALLELE.fullmatch(allele)
        gene, fields, suffix = match.groups()
        prefix = gene + "*" + fields
        ids = [i for i, name in enumerate(self.alleles.get(gene, []))
               if (name == prefix or name.startswith(prefix + ":") or name == prefix + suffix)
               and (not suffix or name.endswith(suffix))]
        if not suffix and len(fields.split(":")) == 2:
            ids = [i for i in ids if not self.alleles[gene][i].endswith("N")]
        if not ids:
            return None
        selected = self.matrices[gene][ids][:, list(indices)]
        from ._motifs import motif_from_sequences
        return motif_from_sequences(selected)

    def save(self, path):
        """Save the HLA reference to an NPZ file.

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
        Includes sequence matrices and reference metadata. Reload with load_hla_reference.
        """
        payload = {"metadata": np.array(json.dumps(self.metadata))}
        for gene in self.alleles:
            payload[f"{gene}_alleles"] = np.asarray(self.alleles[gene])
            payload[f"{gene}_matrix"] = self.matrices[gene]
            payload[f"{gene}_positions"] = np.array(self.positions[gene].to_json(orient="table", index=False))
        with Path(path).open("wb") as stream: np.savez_compressed(stream, **payload)


def load_hla_reference(path):
    """Load a saved HLA reference and verify its fingerprint.

    Parameters
    ----------
    path : str or pathlib.Path
        File saved by HLAReference.save or build_hla_reference.

    Returns
    -------
    HLAReference
        Allele sequences, position annotations and reference metadata.

    Raises
    ------
    ValueError
        For an unsupported schema, inconsistent dimensions or fingerprint mismatch.
    OSError
        If the file cannot be read.

    Notes
    -----
    Loading does not require the original alignment files.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema") != SCHEMA: raise ValueError("Unsupported HLA reference schema.")
        genes = metadata["genes"]
        alleles = {g: archive[f"{g}_alleles"].tolist() for g in genes}
        matrices = {g: archive[f"{g}_matrix"].copy() for g in genes}
        from io import StringIO
        positions = {}

        for g in genes:
            table = json.loads(str(archive[f"{g}_positions"]))
            df = pd.DataFrame(table["data"])

            # Restore nullable integer columns.
            for col in ("POS", "EXON"):
                df[col] = pd.to_numeric(df[col]).astype("Int64")

            positions[g] = df
    for gene in genes:
        if matrices[gene].shape != (len(alleles[gene]), len(positions[gene])):
            raise ValueError(f"{gene}: inconsistent saved reference dimensions.")
        matrices[gene].flags.writeable = False
    if _reference_identity(alleles, matrices, positions, metadata) != metadata["reference_version"]:
        raise ValueError("HLA reference contents do not match their version fingerprint.")
    return HLAReference(alleles, matrices, positions, metadata)


def build_hla_reference(alignment_dir, *, genes=GENES, annotation_dir=None,
                        genomic_table=None, transcripts=None, imgt_release=None,
                        assembly="GRCh38", save_path=None, coordinate_checks=None):
    """Build an HLA amino-acid reference from local IPD alignments.

    Parameters
    ----------
    alignment_dir : str or pathlib.Path
        Folder containing per-gene protein alignment files.
    genes : str or iterable of str, default GENES
        Genes to include. GENES contains A, B, C, DRB1, DQA1, DQB1, DPA1 and
        DPB1. Use 'all' to discover available per-gene protein files.
    annotation_dir : str or pathlib.Path or None, default None
        Folder containing Ensembl lookup/CDS JSON files for genomic coordinates.
        Provide together with genomic_table and matching nucleotide alignments.
    genomic_table : str or pathlib.Path or None, default None
        Local IPD genomic reference table in HTML format.
    transcripts : mapping or None, default None
        Gene-to-transcript overrides. None uses canonical coding transcripts.
    imgt_release : str or None, default None
        Required IPD release. None uses the first alignment's release.
    assembly : str, default 'GRCh38'
        Coordinate assembly. Only GRCh38 is supported.
    save_path : str or pathlib.Path or None, default None
        Optional output NPZ file.
    coordinate_checks : pandas.DataFrame or str or pathlib.Path or None, default None
        Coordinate checks with GENE, AA_POS, POS and EXON columns, or a TSV path.

    Returns
    -------
    HLAReference
        Allele sequences, position annotations and source metadata.

    Raises
    ------
    ValueError
        For invalid genes, inconsistent releases, unsupported assembly or
        invalid coordinate/transcript inputs.
    OSError
        If a required input cannot be read or output cannot be written.

    Notes
    -----
    Genomic positions use the middle nucleotide of each validated codon.
    Protein-only builds leave genomic coordinates missing.
    """
    started = time.perf_counter()
    print("Building HLA reference data for allele mapping", flush=True)
    print("----------------------------------------------", flush=True)

    if assembly != "GRCh38": raise ValueError("Only validated GRCh38 coordinate building is currently supported.")
    if isinstance(genes, str):
        genes = available_hla_genes(alignment_dir) if genes.lower() == "all" else (genes,)
    genes = tuple(_normalise_gene(g) for g in genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("genes must select a nonempty set of unique loci with protein alignments.")
    if (annotation_dir is None) != (genomic_table is None):
        raise ValueError("Provide annotation_dir and genomic_table together for coordinates.")
    source_files, metadata, alleles, matrices, info = [], {}, {}, {}, {}
    genomic = _genomic_table(genomic_table) if genomic_table else {}
    if genomic_table: source_files.append(Path(genomic_table))
    for gene in genes:
        stem = "DRB" if re.fullmatch(r"DRB[1-9]", gene) and not (Path(alignment_dir)/f"{gene}_prot.txt").exists() else gene
        path = Path(alignment_dir)/f"{stem}_prot.txt"
        names, matrix, positions, ref, release = _alignment(path, gene)
        if imgt_release is None: imgt_release = release
        if release != imgt_release: raise ValueError("All alignments must use the requested IMGT release.")
        source_files.append(path)
        coordinates = [(None, None, "not_provided")] * len(positions)
        gene_metadata = {"protein_alignment_reference": ref}
        if annotation_dir is not None:
            if gene not in genomic: raise ValueError(f"{gene}: missing official genomic table entry.")
            coordinates, details, paths = _coordinates(gene, names, matrix, positions,
                Path(alignment_dir)/f"{stem}_nuc.txt", annotation_dir, genomic, (transcripts or {}).get(gene))
            if details["nucleotide_release"] != release: raise ValueError("Protein and nucleotide releases differ.")
            gene_metadata.update(details); source_files.extend(paths)
        frame = pd.DataFrame({"GENE": gene, "AA_POS": positions,
            "POS": pd.array([c[0] for c in coordinates], dtype="Int64"),
            "EXON": pd.array([c[1] for c in coordinates], dtype="Int64"),
            "COORD_STATUS": [c[2] for c in coordinates]})
        frame["MARKER_BASE"] = [f"AA_{gene}_{pos}_{coord if coord is not None else 'NA'}_exon{exon if exon is not None else 'NA'}"
                                 for pos, (coord, exon, _) in zip(positions, coordinates)]
        alleles[gene], matrices[gene], info[gene] = names, matrix, frame
        matrices[gene].flags.writeable = False
        metadata[gene] = gene_metadata
    sources = {str(path.resolve()): _sha(path) for path in source_files}
    meta = {"schema": SCHEMA, "imgt_release": imgt_release, "assembly": assembly,
            "genes": list(genes), "gene_metadata": metadata, "source_sha256": sources,
            "genomic_table_source": GENOMICS_URL, "coordinate_convention": "1-based middle nucleotide of codon"}
    # Paths do not affect content identity, but distinct releases/annotations do.
    meta["reference_version"] = _reference_identity(alleles, matrices, info, meta)
    result = HLAReference(alleles, matrices, info, meta)
    if coordinate_checks is not None:
        checks = coordinate_checks if isinstance(coordinate_checks, pd.DataFrame) else pd.read_csv(coordinate_checks, sep="\t", dtype={"AA_POS": str})
        for _, row in checks.iterrows():
            match = info[row.GENE].loc[info[row.GENE].AA_POS.eq(str(row.AA_POS))]
            if len(match) != 1 or pd.isna(match.iloc[0].POS) or int(match.iloc[0].POS) != int(row.POS) or int(match.iloc[0].EXON) != int(row.EXON):
                raise ValueError(f"Coordinate validation failed for {row.GENE} position {row.AA_POS}.")
    if save_path is not None: result.save(save_path)

    elapsed = time.perf_counter() - started
    
    print(f"Reference building completed in {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes).", flush=True)
    print("----------------------------------------------------------------------------------------------------------", flush=True)

    return result
