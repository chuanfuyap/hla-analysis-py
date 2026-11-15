"""Download official inputs locally; do not commit or redistribute derived data.

Usage: hapy-download-reference local_reference_inputs
Use --imgt-ref with a commit SHA/release branch to pin an IPD snapshot.
Use --all-alignments for the entire upstream alignments folder and Ensembl
annotations for every discovered per-locus protein alignment.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .hla_reference import GENES


def download_hla_inputs(output, *, genes=GENES, imgt_ref="Latest",
                        ensembl_server="https://rest.ensembl.org", all_alignments=False):
    """Download official IPD alignments and Ensembl coordinate inputs locally.

    Parameters
    ----------
    output : str or pathlib.Path
        New or empty destination directory. Created if it does not exist.
    genes : str or iterable of str, optional
        Loci to download, defaulting to the eight classical genes. Superseded
        by all_alignments=True, which discovers every per-locus protein file.
    imgt_ref : str, default "Latest"
        IPD Git commit or release branch. Pin a commit for reproducibility.
    ensembl_server : str, default "https://rest.ensembl.org"
        Ensembl REST endpoint providing canonical GRCh38 lookup/CDS data.
    all_alignments : bool, default False
        Download the entire upstream alignments folder plus annotations for
        every discovered protein locus. Combined group files and nucleotide-
        only loci are downloaded but excluded from annotation discovery.

    Returns
    -------
    pathlib.Path
        Output folder containing alignments, genomic-table HTML, Ensembl
        lookup/CDS JSON and download_manifest.json with URLs and SHA256 hashes.

    Raises
    ------
    ValueError
        For invalid loci, nonempty output, incomplete listings or invalid
        canonical GRCh38 annotations. Annotation failures are not skipped.
    HTTPError or URLError
        If a network request fails after retries.

    Notes
    -----
    Requires network access; prints progress and total elapsed time. In full-
    folder mode, resolves the IPD ref to one commit before fetching files.
    No reference is built here. Failures may leave partial files; retry using
    a new empty directory. Review the IPD licence before redistribution.

    Examples
    --------
    >>> inputs = download_hla_inputs("reference_inputs", all_alignments=True)
    """
    started = time.perf_counter()
    print("Starting download of reference HLA data", flush=True)
    print("", flush=True)
    output = Path(output)
    if isinstance(genes, str): genes = (genes,)
    genes = [gene.strip().upper().removeprefix("HLA-") for gene in genes]
    if len(set(genes)) != len(genes) or any(not re.fullmatch(r"[A-Z][A-Z0-9]*", g) for g in genes):
        raise ValueError("Select unique valid IPD locus names.")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new empty directory to avoid mixing reference snapshots.")
    manifest = {"imgt_ref": imgt_ref, "ensembl_server": ensembl_server, "files": {}}

    def fetch(url, relative):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                request = Request(url, headers={"User-Agent": "hapy-reference-builder", "Accept": "application/json" if path.suffix == ".json" else "*/*"})
                with urlopen(request, timeout=90) as response:
                    data = response.read()
                path.write_bytes(data)
                manifest["files"][relative] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
                return path
            except (HTTPError, URLError):
                if attempt == 3: raise
                time.sleep(2 ** attempt)

    if all_alignments:
        # Resolve a mutable release branch once, so every file uses one snapshot.
        commit_path = fetch(
            f"https://api.github.com/repos/ANHIG/IMGTHLA/commits/{quote(imgt_ref, safe='')}",
            "source/commit.json",
        )
        commit = json.loads(commit_path.read_text())["sha"]
        manifest["resolved_imgt_commit"] = commit
        tree_path = fetch(
            f"https://api.github.com/repos/ANHIG/IMGTHLA/git/trees/{commit}?recursive=1",
            "source/tree.json",
        )
        tree = json.loads(tree_path.read_text())
        if tree.get("truncated"):
            raise ValueError("GitHub returned an incomplete file listing; refusing a partial alignments download.")
        entries = [entry for entry in tree["tree"] if entry["type"] == "blob"
                   and entry["path"].startswith("alignments/")]
        if not entries:
            raise ValueError("No alignment files found in the selected IPD snapshot.")
        for entry in sorted(entries, key=lambda entry: entry["path"]):
            relative = entry["path"]
            if ".." in Path(relative).parts or entry.get("mode") == "120000":
                raise ValueError(f"Unsupported alignment path: {relative!r}.")
            fetch(f"https://raw.githubusercontent.com/ANHIG/IMGTHLA/{commit}/{quote(relative, safe='/')}", relative)
        print(f"Downloaded the complete alignments folder ({len(entries)} files, commit {commit}).", flush=True)
        genes = []
        for path in sorted((output / "alignments").glob("*_prot.txt")):
            gene = path.name.removesuffix("_prot.txt").upper()
            if gene in {"DRB", "CLASSI"} or not re.fullmatch(r"[A-Z][A-Z0-9]*", gene):
                continue
            with path.open() as stream:
                if any(re.match(rf"^\s*{re.escape(gene)}\*\d", line) for line in stream):
                    genes.append(gene)
        if not genes:
            raise ValueError("No per-locus protein alignments found for Ensembl annotation downloads.")
        print(f"Downloading Ensembl annotations for {len(genes)} loci: {', '.join(genes)}.", flush=True)
    manifest["annotation_genes"] = genes


    fetch("https://www.ebi.ac.uk/ipd/imgt/hla/about/help/genomics/", "genomics.html")
    for gene in genes:
        stem = "DRB" if gene == "DRB1" else gene
        for kind in (() if all_alignments else ("prot", "nuc")):
            fetch(f"https://raw.githubusercontent.com/ANHIG/IMGTHLA/{quote(imgt_ref, safe='')}/alignments/{stem}_{kind}.txt",
                  f"alignments/{stem}_{kind}.txt")
        server = ensembl_server.rstrip("/")
        symbol = gene if gene in {"HFE", "MICA", "MICB", "TAP1", "TAP2"} else "HLA-" + gene
        lookup = fetch(f"{server}/lookup/symbol/homo_sapiens/{symbol}?expand=1;content-type=application/json",
                       f"annotations/{gene}.json")
        record = json.loads(lookup.read_text())
        transcripts = [t for t in record.get("Transcript", []) if t.get("is_canonical") and t.get("Translation")]
        if record.get("assembly_name") != "GRCh38" or len(transcripts) != 1:
            raise ValueError(f"{gene}: expected one canonical protein-coding GRCh38 transcript.")
        transcript = transcripts[0]["id"]
        fetch(f"{server}/sequence/id/{transcript}?type=cds;content-type=application/json", f"annotations/{gene}.cds.json")
        print(f"Downloaded {gene} inputs ({transcript}).", flush=True)
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    elapsed = time.perf_counter() - started
    print("----------------------------------------------------------------------------------------------------------", flush=True)
    print(f"Download completed in {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes).", flush=True)
    print("Inputs saved locally. Build with hp.build_hla_reference(); review the IPD licence before redistribution.")
    print("----------------------------------------------------------------------------------------------------------", flush=True)

    return output


def main():
    """Run download_hla_inputs using command-line options; print total duration.

    Accepts output, --genes, --imgt-ref, --ensembl-server and --all-alignments.
    Returns None so successful console execution exits with status zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--imgt-ref", default="Latest", help="IPD Git commit SHA or release branch")
    parser.add_argument("--genes", nargs="+", default=list(GENES), help="IPD loci; defaults to the eight classical genes; superseded by --all-alignments")
    parser.add_argument("--all-alignments", action="store_true", help="Download all alignments and Ensembl annotations for all discovered protein loci (supersedes --genes)")
    parser.add_argument("--ensembl-server", default="https://rest.ensembl.org")
    args = parser.parse_args()
    download_hla_inputs(**vars(args))


if __name__ == "__main__":
    main()
