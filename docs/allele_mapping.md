# HLA allele calls to amino acid genotypes

The human leukocyte antigen (HLA) region is one of the most genetically diverse 
regions of the human genome. Historically, HLA variation was characterised using 
serological typing, but advances in molecular techniques have enabled the identification 
of HLA alleles at increasingly high resolution. Today, HLA alleles can be identified
using targeted typing assays or inferred from sequencing and genotyping data using computational methods.

As HLA alleles encode specific amino acid sequences, they provide an alternative 
route to investigating genetic variation at the protein level. When imputed amino 
acid data are unavailable, HLA alleles can be mapped to their corresponding amino 
acid sequences using reference databases such as IPD-IMGT/HLA. This allows amino 
acid variation to be reconstructed from existing HLA allele data, enabling downstream 
association testing and fine-mapping to identify specific amino acid positions and 
residues associated with disease susceptibility.

To facilitate this process, `hapy` provides functions for mapping HLA alleles to 
amino acid sequences and generating amino acid data suitable for downstream analyses.

To use this feature, a reference is needed, which can be built once, then be used to
map sample allele pairs without reparsing
alignments or recalculating genomic coordinates. The mapper returns either an
`HLAdata` object for existing analysis functions or a sample-by-residue dosage
DataFrame. Both retain the original allele calls and within-gene allele pairs.

The default genes are **A, B, C, DRB1, DQA1, DQB1, DPA1 and DPB1**.
Use `genes` to select additional loci represented in your official protein alignments.
GRCh38 is the default and currently the only supported coordinate assembly.
There is no frequency or polymorphic-position filtering during reference
building or sample conversion. Monomorphic positions remain available.

## Official inputs and downloads

Obtain Immuno Polymorphism Database (IPD) files from the [official download page](https://www.ebi.ac.uk/ipd/imgt/hla/download/)
or its linked [ANHIG/IMGTHLA repository](https://github.com/ANHIG/IMGTHLA).
Pin an IPD commit or release branch when reproducing an analysis. Use all
alignment files from the same release.

| Local input | Purpose |
|---|---|
| `alignments/GENE_prot.txt` | Official protein alignment, including unknown/gap/identity symbols and the mature-protein numbering ruler. |
| `alignments/GENE_nuc.txt` | Official coding-sequence alignment used to validate the assembly reference. It does not itself provide genomic coordinates. |
| `genomics.html` | Local snapshot of the [IPD genomic reference table](https://www.ebi.ac.uk/ipd/imgt/hla/about/help/genomics/), recording the declared reference alleles, strand and start coordinates. |
| `annotations/GENE.json` | Ensembl expanded GRCh38 gene lookup, containing transcripts, coding bounds and exon structure. |
| `annotations/GENE.cds.json` | CDS sequence JSON for the selected annotated transcript. |

For DRB loci, shared `DRB_prot.txt` and `DRB_nuc.txt` files are supported as
fallbacks. Per-gene files such as `DRB1_prot.txt`/`DRB1_nuc.txt` take priority. The builder does not need
`*_gen.txt`, an entire genome FASTA or the full `hla.dat` database.

Download inputs using the command installed with hapy:

```bash
# "local_reference_inputs" is the folder name to store the reference data
hapy-download-reference local_reference_inputs 
```

The same downloader is available in Python:

```python
import hapy as hp

# "local_reference_inputs" is the folder name to store the reference data
inputs = hp.download_hla_inputs("local_reference_inputs", all_alignments=True)
```

It returns the destination as a `pathlib.Path`. Python options are `genes`,
`imgt_ref`, `ensembl_server` and `all_alignments`, matching the CLI options.
Use a separate empty destination for each download, as it would not overwrite an 
already existing folder.

To download the **entire upstream `alignments/` folder**, like the supplied local
IPD folder, add `--all-alignments`:

```bash
hapy-download-reference local_reference_inputs --all-alignments
```

With `--all-alignments`, Ensembl lookup and CDS annotations are downloaded for
**every discovered per-locus protein alignment**, superseding `--genes`.

By default without `--all-alignments`, the script downloads the eight genes and 
saves a URL/hash manifest. To pin IPD, add `--imgt-ref YOUR_COMMIT_SHA`. To
download a subset, add, for example, `--genes A DRB1`. Additional loci can also 
be requested explicitly, for example `--genes A DRB1 DRB3 E MICA`.

Ensembl provides [expanded gene/exon lookup](https://rest.ensembl.org/documentation/info/lookup)
and [CDS sequence retrieval](https://rest.ensembl.org/documentation/info/sequence_id).
Manual downloads must follow the filenames above and use the same transcript
in the lookup and CDS JSON. The builder checks the CDS transcript ID, assembly,
gene, coding length and translated protein before assigning coordinates.

## Build and save a reference

```python
import hapy as hp

reference = hp.build_hla_reference(
    "local_reference_inputs/alignments",
    annotation_dir="local_reference_inputs/annotations",
    genomic_table="local_reference_inputs/genomics.html",
    assembly="GRCh38",
    save_path="hla_reference.npz",
)

print(reference.metadata["imgt_release"])
print(reference.version)
print(reference.positions["DRB1"].head())
```

`reference.matrices[gene]` is the allele-by-aligned-position matrix;
`reference.alleles[gene]` identifies its rows. `reference.positions[gene]` stores
`GENE`, `AA_POS`, `POS`, `EXON`, `COORD_STATUS` and the residue-free `MARKER_BASE`.
All alignment columns are retained, including monomorphic and insertion columns.
The compressed NPZ stores arrays and JSON rather than pickled objects.

Omitting `genes` always selects the eight default genes, even if additional
alignment files are present. An explicit list selects exactly those loci:

```python
reference = hp.build_hla_reference(
    "local_reference_inputs/alignments",
    genes=("A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1",
           "DRA", "DRB3", "DRB4", "DRB5", "E", "F", "G", "MICA", "MICB"),
    save_path="extended_reference.npz",
)
```

To discover and build every available per-locus protein alignment:

```python
alignment_dir = "local_reference_inputs/alignments"
print(hp.available_hla_genes(alignment_dir))
reference = hp.build_hla_reference(
    alignment_dir, genes="all", save_path="all_loci_reference.npz",
)
```

For a subset of genes, pass `genes=("A", "DRB1")`. `imgt_release="3.66.0"`, for
example, requires matching release headers rather than silently accepting other
versions. 

A protein-only build is possible by omitting both `annotation_dir` and
`genomic_table`. It preserves sequences and position labels but explicitly
records missing genomic coordinates and exon annotations. It is not a source
of inferred genomic positions.

### Load the saved file for later analyses:

```python
reference = hp.load_hla_reference("hla_reference.npz")
```

## Amino acid coordinates and numbering

The reference builder maps HLA amino acid positions to **GRCh38 genomic coordinates**, 
using the middle nucleotide of each codon. Coordinates are validated against the reference 
genome to ensure accurate mapping.

Amino acid numbering follows the **IPD-IMGT/HLA convention**, where the mature protein 
starts at position 1 and leader sequences have negative positions.

The builder accounts for exon boundaries, strand orientation and differences between IPD 
sequences and GRCh38. **Ambiguous or unmappable positions are assigned missing coordinates rather than estimated values.**

### Marker naming

Amino acid markers follow this format:

`AA_DRB1_80_32584153_exon2_D`

This identifies the gene (DRB1), amino acid position (80), GRCh38 coordinate (32584153), 
exon (2) and residue (D).

Insertions use labels such as `52i1` and `52i2`. Missing coordinates are represented by `NA`.

### Optional coordinate validation

Users can provide independently verified coordinates to check the reference during construction:

```python
import pandas as pd

checks = pd.DataFrame({
    "GENE": ["DRB1"],
    "AA_POS": ["80"],
    "POS": [32584153],
    "EXON": [2],
})

reference = hp.build_hla_reference(
    "local_reference_inputs/alignments",
    annotation_dir="local_reference_inputs/annotations",
    genomic_table="local_reference_inputs/genomics.html",
    coordinate_checks=checks,
    save_path="hla_reference.npz",
)
```

Coordinate checks can also be supplied as a TSV file. The builder reports inconsistencies 
and raises an error if the reference sequence cannot be validated against GRCh38.

## Map HLA alleles to amino acids

Use one row per sample, indexed by IID, and two allele columns per gene:

```python
# example data
calls = pd.DataFrame({
    "A_1": ["A*01:01:01:01", "A*02:01:01:01"],
    "A_2": ["A*02:01:01:01", "A*02:01:01:01"],
    "DRB1_1": ["DRB1*01:01:01:01", "DRB1*15:01:01:01"],
    "DRB1_2": ["DRB1*04:01:01:01", "DRB1*15:01:01:01"],
}, index=["001", "002"])

hladat = hp.map_hla_alleles(calls, reference)
print(hladat.AA.data)  # residue markers × samples
print(hladat.AA.info)  # exon-aware position IDs and marker metadata
```

Map only genes represented by complete column pairs in the call table. A missing
second column is an input error. Extra non-allele columns are retained with the
original calls but are not converted. Custom column names can be selected with:

```python
hladat = hp.map_hla_alleles(
    calls, reference,
    allele_columns={"A": ("A_1", "A_2")},
)
``` 

This would generate a `HLAdata` object compatible with the other `hapy` functions.

### DataFrame output

Sample by amino acid pandas DataFrame can also be created for convenience.

```python
dosages = hp.map_hla_alleles(
    calls, reference, output="dataframe", save_path="aa_dosages.tsv",
)
# dosages: samples × residue markers

reloaded = hp.read_aa_dosage("aa_dosages.tsv")
```

It uses the existing `HLAdata` construction and marker annotation. By default,
simple MAF filtering is applied to mapped output states; original allele calls and
the full allele-pair mapping are preserved.

Export writes both:

- `aa_dosages.tsv`: IID-indexed diploid residue dosages, with missing values as `NA`.
- `aa_dosages.tsv.phase.npz`: original calls, integer allele IDs, called-allele
  matrices, complete position metadata and reference identity.

Keep both files together to preserve the information required for downstream motif analysis. 
The dosage file can be loaded independently, but without the accompanying phase file, 
the original allele calls and phase cannot be recovered, preventing motif construction.

A DataFrame returned directly retains this information in dosages.attrs["hla_mapping"].

## QC before and after mapping

For dosage HLA-to-AA mapping, avoid applying `simpleQC` when loading the data. 
Load the allele data with `simpleQC=False`: rare HLA alleles can
carry common amino acids, so removing those alleles removes their contributions
to the mapped AA dosages. Filtering input alleles on imputation quality can also
leave an incomplete panel.

Apply simple QC **after mapping**, to the AA states used for analysis. Both
`map_hla_alleles()` and `hla_dosage_to_aa()` do this by default:

```python
aa = hp.hla_dosage_to_aa(allele_dosages, reference, simpleQC=True)
```

## HLA allele resolution

By default, the mapper **retains full HLA allele resolution** rather than reducing
 alleles to two-field names.

Although two-field resolution generally identifies the same protein sequence, some 
alleles differ in amino acid sequence, insertions or missing sequence information. 
Reducing resolution could therefore introduce errors in amino acid mapping and motif analysis.

The mapper preserves the original allele names and sequences, including expression suffixes. 
Lower-resolution allele calls are supported through consensus mapping where possible.

This ensures that amino acid dosages and motifs are derived from the most accurate sequence 
information available.

For further details, see the official [HLA nomenclature](https://www.hla.alleles.org/pages/nomenclature/naming_alleles/) and [expression suffix rules](https://www.hla.alleles.org/pages/nomenclature/assigning_suffixes/).

## Allele resolution and missing sequences

The mapper supports full-resolution and two-field HLA allele names, with or without 
the `HLA-` prefix. Older formats such as `A0101` are also supported.

If a higher-resolution allele is absent from the reference,
`fallback_to_two_field=True` uses compatible records sharing its two-field name,
preserving expression suffixes. This is the default for `map_hla_alleles()` and
`hla_dosage_to_aa()`.

```python
hladat = hp.map_hla_alleles(
    calls, reference,
    fallback_to_two_field=True,
)
```

Missing allele calls or unresolved sequences result in missing amino acid dosages at the affected 
positions. The mapper reports these issues through warnings or raises an error when `strict=True`.

Mapping issues can be inspected using `hladat.hla_mapping.issues`.

## Ambiguous amino acids and alignment gaps

At two-field HLA allele resolution, amino acid sequences are generally consistent across matching reference alleles. Differences mainly arise from incomplete sequences (*) and alignment gaps (.).

- * (missing sequence): If a known residue is available from another matching allele, that residue is used. Otherwise, the position remains unresolved (?).
- . (alignment gap): Represents the absence of an amino acid at that alignment position. Unlike *, this is a known state.
- Combined states (e.g. .P): When matching alleles contain either a gap or a residue, both possibilities are retained as a single category.

Unresolved positions (?) are not assigned an amino acid dosage. For fractional dosage mapping, contributions from known alleles are retained, while unresolved contributions are reported in aa.conversion_metadata["resolution"].

## Phased amino acid motifs

The mapper preserves HLA allele phasing within each gene, allowing amino acid motifs (haplotypes) 
to be constructed from selected positions.

```python
motifs = hp.aa_motif_dosages(hladat, "DRB1", (11, 13, 71))

print(motifs.AA.data)  # motif markers × samples; compact labels such as LFR
print(motifs.conversion_metadata["allele_groups"])
```

The function identifies unique amino acid combinations across the selected positions and 
calculates their **dosages (0, 1 or 2)** for each sample. Alleles sharing the same motif are grouped together.

Confirmed motif contributions are retained when the other allele cannot be
mapped. Samples with no usable motif assignment remain missing.

**Note:** Phasing is preserved within each HLA gene, but not between genes.

## Parameters

### `build_hla_reference()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alignment_dir` | `str` or `pathlib.Path` | **Required** | Local official alignment files. |
| `genes` | `str` or iterable of `str` | The eight classical genes | Loci to build; a single locus string or `"all"` is also accepted. |
| `annotation_dir` | `str`, `pathlib.Path` or `None` | `None` | Per-gene GRCh38 lookup/CDS JSON files. Required with `genomic_table` for coordinates. |
| `genomic_table` | `str`, `pathlib.Path` or `None` | `None` | Local official IPD genomic-table HTML snapshot. |
| `transcripts` | mapping or `None` | `None` | Explicit gene-to-transcript IDs; otherwise select the annotated canonical transcript. Supply matching CDS JSON files. |
| `imgt_release` | `str` or `None` | `None` | Required release, or infer a single consistent release from alignment headers. |
| `assembly` | `str` | `"GRCh38"` | Only GRCh38 coordinate building is implemented. |
| `save_path` | `str`, `pathlib.Path` or `None` | `None` | Optional compressed reference output. |
| `coordinate_checks` | `pandas.DataFrame`, TSV path or `None` | `None` | Same-build `GENE`, `AA_POS`, `POS`, `EXON` validation cases. |

### `map_hla_alleles()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calls` | `pandas.DataFrame` | **Required** | IID-indexed table containing two allele slots per selected gene. |
| `reference` | `HLAReference`, `str` or `pathlib.Path` | **Required** | Built reference object or saved NPZ. |
| `allele_columns` | mapping or `None` | `None` | Gene-to-pair column selection; otherwise detect standard names. |
| `output` | `str` | `"hladata"` | `"hladata"` or `"dataframe"`. |
| `save_path` | `str`, `pathlib.Path` or `None` | `None` | Write sample-by-marker TSV and phase sidecar. |
| `fallback_to_two_field` | `bool` | `True` | For unavailable three/four-field calls, use suffix-preserving two-field consensus and report the inference. |
| `strict` | `bool` | `False` | Raise instead of warning when calls/positions cannot be resolved. |
| `simpleQC` | `bool` | `True` | Apply simple MAF filtering to mapped output states. |
| `allele_filter` | `float` | `0.005` | Minimum output-state frequency; also require frequency below `1 - allele_filter`. |
| `verbose` | `bool` | `True` | Print mapping, filtering and timing summaries. |

`load_hla_reference(path)` loads a saved reference. `read_aa_dosage(fileloc,
phase_path=None)` reads the mapped TSV or dosage DataFrame; a supplied sidecar
path must exist. `aa_motif_dosages(data, gene, positions)` requires preserved
allele-pair mapping and a nonempty unique sequence of position identifiers.

## Updating and reproducibility

When updating to a new IPD-IMGT/HLA release, rebuild the reference using the latest 
sequence alignments and genome annotations. **Keep previous reference versions** to ensure 
reproducibility.

Each reference records its IPD release, genome assembly, annotation details and source
information. Avoid mixing references from different genome assemblies without coordinate conversion.

## Licence and citation

IPD-IMGT/HLA data are distributed under a [Creative Commons Attribution-NoDerivatives licence](https://www.ebi.ac.uk/ipd/imgt/hla/licence/). Users should acknowledge the database and obtain permission before redistributing modified reference data.

For this reason, **hapy does not distribute IPD sequence alignments or derived reference files**. 
Users must download the data and build their references locally.

When publishing, cite the [IPD-IMGT/HLA database](https://www.ebi.ac.uk/ipd/imgt/hla/) and the 
relevant Ensembl annotation resource. Record the IPD release and `hapy` version to support reproducibility.
