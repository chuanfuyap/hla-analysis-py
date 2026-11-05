# HLA amino acid position visualisation

`hp.vis.plot_aa_association()` and `hp.vis.plot_aa_interaction()` return `(fig, ax)`.

Matplotlib is the only plotting dependency. Plots are not displayed unless `show=True`.

## Choosing the X-axis

The same `x_axis` options are available for `plot_aa_association()` and both
`plot_aa_interaction()` modes (`scatter3d` and `heatmap`):

| `x_axis` | Spacing and meaning | Gene selection |
|---|---|---|
| `"amino_acid"` **(default)** | Numeric amino acid sequence positions; gaps reflect differences in sequence position. | Select one gene, e.g. `genes="DRB1"`. |
| `"genomic"` | Actual genomic base-pair coordinates. Positions within genes cluster because inter-gene gaps are much larger. | One or multiple genes. |
| `"equally_spaced"` | Distinct AA positions are ordered by genomic coordinate and spaced evenly, like categorical string plotting. Spacing represents neither genomic nor sequence distance. | One or multiple genes. |

For example, add one of these configurations to either plotting function:

```python
# Numeric sequence positions within HLA-DRB1 (the default axis).
x_axis="amino_acid", genes="DRB1"

# Physical genomic spacing across HLA genes.
x_axis="genomic"

# Even spacing across HLA positions, with gene/position tick labels.
x_axis="equally_spaced"
```

`equally_spaced` prints an INFO note on each call explaining that distances are
not represented. In interaction plots, the partner Y-axis remains genomic in all
three modes. `position_unit="Mb"` formats genomic axes only; it does not scale
amino acid sequence positions or equally spaced indices. Changing the axis never
combines P-values or merges distinct genomic records.

## Input and significance

An ID such as `AA_DRB1_11_32552129` is parsed into gene DRB1, **amino acid sequence position 11**, 
and **genomic coordinate 32552129**. Plots default to numeric amino acid sequence positions within one gene. Select
`genes="DRB1"`, for example. Use `x_axis="genomic"` for physical genomic spacing
across multiple genes. IDs are case-normalised for gene matching. No genome
assembly or coordinate conversion is inferred; `reference_build="GRCh38"` adds a
label only. Coordinates must all use the same reference build supplied by you.

Only position-level tests are accepted. Residue/exon suffixes are rejected because
a residue test cannot be assumed to represent an omnibus position test. Duplicate
rows for a position raise an error. Obtain position-level tests from the amino acid association
analysis; this module never combines residue P-values. Different genomic coordinates
are preserved even if their gene and sequence position are the same.

Select `pval_col="LR_p"` (the default when neither significance column is supplied),
or `log10p_col="META_log10P"`. The latter means **positive −log₁₀(P)**, not log₁₀(P).
For standard AA results with both single-column and omnibus tests, explicitly use
`pval_col="LRp_Unip"` to plot the runner's coalesced test column. No test columns are
silently mixed by the plotting functions. Interaction results use `I_p`.

Raw P-values must satisfy `0 < P <= 1`. Strings or Decimal values such as
`"6.09e-412"` are accepted without underflow. A p-value reported as zero may represent an extremely small, 
non-zero probability that falls below the smallest representable floating-point value: supply a finite −log₁₀(P) instead. 
Labels derive the scientific notation  and exponent directly from the log value; they do not exponentiate
the entire tiny probability. `p_digits=3` controls displayed significant figures.
Accuracy is limited by the precision of the input. P=1 is valid and visible.

## Association plot

This is for plotting amino acid position associations results.

```python
import pandas as pd
import hapy as hp

association = pd.read_csv("examples/visualisation/association.csv")
fig, ax = hp.vis.plot_aa_association(
    association, x_axis="genomic", variant_col="VARIANT", log10p_col="META_log10P",
    highlight_top=True, annotate_p=True, position_unit="Mb",
    title="HLA amino acid position association",
    significance_threshold=5e-8, save_path="association.pdf",
)
```

`highlight_top=True` selects the strongest hit; `highlight_top=3` selects three.
`highlights=[("B", 63), ("DRB1", 11)]` selects gene and sequence-position pairs;
exact position IDs are also accepted. Gene is always part of tuple matching.
`genes="DRB1"` or `genes=["B", "DPB1"]` filters before selecting top hits.
Multiple coordinates for a selected gene/sequence position are all highlighted.
Missing requested highlights produce a warning.

`significance_threshold` accepts actual P-values (including scientific-notation
strings); `threshold_log10p` instead accepts −log₁₀(P). Either accepts a scalar or
list. `threshold_kwargs` customises the Matplotlib horizontal lines.

## Interaction scatter and sparse heatmap

This is for plotting GxG (amino acid positions and SNP) interaction results.

```python
from matplotlib.colors import Normalize

interactions = pd.read_csv("examples/visualisation/interaction.csv")
hits = [("B", 63, 38175678), ("DPB1", 9, 37289819), ("DPB1", 35, 37289819)]
shared_scale = Normalize(vmin=0, vmax=6)
for kind in ["scatter3d", "heatmap"]:
    fig, ax = hp.vis.plot_aa_interaction(
        interactions, x_axis="genomic", variant_col="AA_VARIANT", partner_pos_col="TCR_POS",
        partner_variant_col="PARTNER", log10p_col="META_log10P",
        kind=kind, highlights=hits, annotate_p=True, position_unit="Mb",
        norm=shared_scale, figsize=(12, 7), annotation_fontsize=8,
        annotation_offsets={
            ("B", 63, 38175678): (8, 12),
            ("DPB1", 9, 37289819): (-230, 45),
            ("DPB1", 35, 37289819): (-230, 95),
        },
        save_path=f"interaction_{kind}.pdf",
    )
```

X is the HLA position's genomic coordinate; Y is the partner genomic coordinate.
The 3D Z-axis and all point colours are −log₁₀(P). Annotation labels include the
HLA gene, sequence position, partner information, and actual P-value.
`partner_variant_col` can be parsed without an explicit position column for IDs
such as `chr7:38175678`, `chr7_38175678_A_G`, or `SNP_A_7_38175678_exon1`.
rsIDs need `partner_pos_col`. When both sources provide coordinates, disagreements
are rejected. `partner_chrom="7"` supplies a chromosome label when only numeric
positions are available; chromosomes are otherwise derived only from partner IDs.
Coordinates spanning multiple partner chromosomes are rejected: filter to one
chromosome per plot to avoid overlaying unrelated chromosome axes.

Heatmaps are **sparse point heatmaps**: squares centred on exact observed coordinate
pairs. They do not form a raster, interpolate missing combinations, or assume equal
spacing. 
- Background (`missing_color`) indicates unobserved combinations; observed non-significant values remain coloured. Point size is in display units, not genomic bin width. 
- Reduce `marker_size` or zoom in if nearby observations overlap. 
- There is no binning. 
- `annotate_cells=True` labels every observed point; otherwise only highlighted points are labelled.

Repeated coordinate pairs are preserved by default (`duplicate="keep"`). Distinct
partner variants can share a genomic coordinate. In 3D they retain their individual
significance heights. In the heatmap all points remain, with a warning that one
point can cover another; coordinates are never shifted. Filter specific variants
for an unambiguous 2D view. Explicit `duplicate="error"` rejects repeats, while
`duplicate="max"` warns and selects the largest −log₁₀(P) per pair (ties keep the
first input row). This optional visual aggregation does not combine P-values.

## Annotations and styling

`annotate_p=False` highlights without default labels. `annotations` can be a mapping
from highlight tuples (or position IDs) to custom text, or a callable receiving a
copy of the parsed row (`id`, `gene`, `aa_pos`, `x`, `score`, and interaction `y` /
`partner_label`). A callable can return None to omit a label. Custom text replaces
the entire default label. Labels use heuristic screen-space placement, with explicit offsets available for
fine adjustments. `annotation_offsets` maps highlight keys to `(dx, dy)` offsets in
points; use it for dense figures. 3D labels are reprojected on every draw/view change.

Shared styling keywords (passed directly to either plotting function):

| Options | Meaning / defaults |
|---|---|
| `figsize`, `dpi` | Figure size `(9, 5)` inches; resolution `300` |
| `title`, `title_size`, `title_weight` | Title and font styling (`14`, `normal`) |
| `xlabel`, `ylabel`, `zlabel`, `label_size`, `label_weight` | Axis labels and fonts (`11`, `normal`) |
| `tick_size`, `x_ticks`, `y_ticks`, `z_ticks` | Tick font size `9`; approximate maximum tick intervals `6`, `6`, `5` |
| `cmap`, `vmin`, `vmax`, `norm` | `viridis`; default range zero to largest score (at least 1). Supply limits or a shared Matplotlib Normalize, not both |
| `marker`, `marker_size`, `alpha`, `edgecolor`, `linewidth` | Circle / square; area 25 / 100 points²; alpha .9; no edges; width .5 |
| `highlight_color`, `highlight_size` | Gold fill/red outline; default area max(80, 2.5 times ordinary markers) |
| `colorbar`, `colorbar_label` | Enabled; label −log₁₀(P) |
| `annotation_fontsize`, `annotation_offset`, `annotation_arrows` | `9`, `(8, 12)` points, connectors enabled |
| `annotation_kwargs` | Additional Matplotlib annotation text properties (e.g. color, bbox, ha) |
| `xlim`, `ylim`, `zlim` | Limits in original base-pair / significance units, even when ticks display Mb |
| `position_unit`, `xformatter`, `yformatter` | `bp` or `Mb`; optional Matplotlib Formatter or `(value, tick_position)` callable |
| `reference_build` | User-supplied build label; no coordinate transformation |
| `elev`, `azim` | 3D viewing angles `25`, `-60` degrees |
| `missing_color` | Heatmap background `#eeeeee` |
| `rasterized` | False; rasterise scatter artists when requested |
| `save_path`, `save_kwargs`, `show` | Optional PNG/PDF/SVG path; extra savefig options; display False |
| `p_digits` | Significant figures for actual P-value labels: `3` |

`ax=` accepts an existing 2D axes for association/heatmap, or a 3D axes for scatter3d.
With existing axes, their figure size/layout is retained; `figsize` applies only to
new figures. All functions return the same figure and axes for further editing.
No global Matplotlib style/backend is changed. For vector publication output leave
`rasterized=False` and save PDF/SVG; PNG resolution uses `dpi`. Save directories must
already exist. Figures remain open for the caller to modify or close.

## Reproduce examples and checks

From the repository root:

```bash
MPLBACKEND=Agg python examples/visualisation/plot_examples.py --output /tmp/hapy_figures
python -m pytest tests/test_vis.py
```

The example CSVs are synthetic and demonstrate plotting, not scientific findings.
The script writes all three plot modes in PNG, PDF and SVG. Tests cover parsing,
position/residue distinctions, extreme significance, gene-aware highlights, sparse
coordinates, duplicate policies, non-finite/missing values, shared scales, existing
axes, input immutability, headless rendering and exports.

