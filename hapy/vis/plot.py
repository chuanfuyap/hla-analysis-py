"""Matplotlib visualisations of HLA amino acid *position-level* test results.

Both functions return (fig, ax), never alter input frames, and display only when
show=True. See docs/visualisation.md for shared styling options and examples.
"""
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator
from matplotlib.text import Annotation, Text
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d

from ._utils import (prepare_data, add_partners, unique_rows, select_hits,
                     format_pvalue, _log_probability)

__all__ = ['plot_aa_association', 'plot_aa_interaction']


@dataclass
class _Style:
    x_axis: str = 'amino_acid'
    figsize: tuple = (9, 5)
    dpi: int = 300
    title: str | None = None
    title_size: float = 14
    title_weight: str = 'normal'
    xlabel: str | None = None
    ylabel: str | None = None
    zlabel: str | None = None
    label_size: float = 11
    label_weight: str = 'normal'
    tick_size: float = 9
    x_ticks: int = 6
    y_ticks: int = 6
    z_ticks: int = 5
    cmap: object = 'viridis'
    norm: object = None
    vmin: float | None = None
    vmax: float | None = None
    marker: str | None = None
    marker_size: float | None = None
    alpha: float = .9
    edgecolor: object = 'none'
    linewidth: float = .5
    highlight_color: object = 'gold'
    highlight_facecolor: object = None
    highlight_edgecolor: object = 'red'
    highlight_linewidth: float = 1.2
    highlight_alpha: float = 1.0
    highlight_size: float | None = None
    colorbar: bool = True
    colorbar_label: str = '−log₁₀(P)'
    annotation_style: str = "full"
    annotation_partner: bool = True
    annotation_avoid_overlap: bool = True
    annotation_fontsize: float = 9
    annotation_offset: tuple = (8, 12)
    annotation_arrows: bool = True
    annotation_kwargs: dict = field(default_factory=dict)
    xlim: tuple | None = None
    ylim: tuple | None = None
    zlim: tuple | None = None
    position_unit: str = 'bp'
    xformatter: object = None
    yformatter: object = None
    reference_build: str | None = None
    rasterized: bool = False
    elev: float = 25
    azim: float = -60
    missing_color: object = '#eeeeee'
    save_path: object = None
    save_kwargs: dict = field(default_factory=dict)
    show: bool = False
    p_digits: int = 3


def _axes(ax, style, three=False):
    if ax is None:
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi, constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d') if three else fig.add_subplot(111)
    else:
        if (getattr(ax, 'name', '') == '3d') != three:
            raise ValueError('scatter3d requires a 3D axes; other modes require a 2D axes.')
        fig = ax.figure
    return fig, ax


def _normalization(frame, s):
    if s.norm is not None:
        if s.vmin is not None or s.vmax is not None:
            raise ValueError('Supply norm or vmin/vmax, not both.')
        return s.norm
    lo = 0 if s.vmin is None else s.vmin
    hi = max(1., float(frame.score.max())) if s.vmax is None else s.vmax
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError('Colour limits must be finite with vmax > vmin.')
    return Normalize(lo, hi)


def _coord_formatter(s, custom):
    if custom is not None:
        return FuncFormatter(custom) if callable(custom) else custom
    if s.position_unit == 'Mb':
        return FuncFormatter(lambda value, _: f'{value / 1e6:g}')
    if s.position_unit != 'bp':
        raise ValueError("position_unit must be 'bp' or 'Mb'.")
    return FuncFormatter(lambda value, _: f'{value:,.0f}')


def _coordinate_label(text, s):
    return f'{text} ({s.position_unit}' + (f'; {s.reference_build}' if s.reference_build else '') + ')'


def _display_positions(frame, s):
    if s.x_axis not in {'genomic', 'amino_acid', 'equally_spaced'}:
        raise ValueError("x_axis must be 'genomic', 'amino_acid', or 'equally_spaced'.")
    frame = frame.copy()
    if s.x_axis == 'amino_acid':
        if frame.gene.nunique() != 1:
            raise ValueError("x_axis='amino_acid' requires a single HLA gene; select it with genes='DRB1', for example.")
        s._aa_gene = frame.gene.iloc[0]
        frame['plot_x'] = frame.aa_pos
    elif s.x_axis == 'equally_spaced':
        # Identity includes the gene and genomic coordinate, so neither distinct
        # genes nor different coordinates are collapsed into one sequence position.
        positions = frame[['gene', 'aa_pos', 'x']].drop_duplicates().sort_values(
            ['x', 'gene', 'aa_pos'], kind='stable')
        keys = list(positions.itertuples(index=False, name=None))
        order = {key: number for number, key in enumerate(keys)}
        frame['plot_x'] = [order[key] for key in frame[['gene', 'aa_pos', 'x']].itertuples(index=False, name=None)]
        s._position_labels = [f'HLA-{gene}:{int(pos)}' for gene, pos, _ in keys]
        print('INFO: AA positions are equally spaced in genomic-coordinate order. '
              'X-axis spacing does not represent genomic distance or amino acid sequence distance.', flush=True)
    else:
        frame['plot_x'] = frame.x
    return frame


def _decorate(fig, ax, artist, s, interaction=False, three=False, partner_chrom=None):
    ax.set_title(s.title or '', fontsize=s.title_size, fontweight=s.title_weight)
    labels = dict(fontsize=s.label_size, fontweight=s.label_weight)
    if s.x_axis == 'amino_acid':
        default_xlabel = f'HLA-{s._aa_gene} amino acid sequence position'
    elif s.x_axis == 'equally_spaced':
        default_xlabel = 'HLA amino acid positions (equally spaced)'
    else:
        default_xlabel = _coordinate_label('HLA amino acid genomic position', s)
    ax.set_xlabel(s.xlabel or default_xlabel, **labels)
    partner = f'Partner genomic position' + (f' on chr{partner_chrom}' if partner_chrom else '')
    ax.set_ylabel(s.ylabel or (_coordinate_label(partner, s) if interaction else '−log₁₀(P)'), **labels)
    ax.tick_params(axis='both', labelsize=s.tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=s.x_ticks, integer=s.x_axis == "amino_acid"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}")
        if s.x_axis == "amino_acid" and s.xformatter is None else _coord_formatter(s, s.xformatter))
    if s.x_axis == 'equally_spaced':
        tick_positions = np.unique(np.linspace(0, len(s._position_labels)-1,
            min(len(s._position_labels), max(1, int(s.x_ticks)))).round().astype(int))
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        if s.xformatter is None:
            def format_position(value, _):
                number = int(round(value))
                return s._position_labels[number] if 0 <= number < len(s._position_labels) and abs(value-number) < 1e-6 else ''
            ax.xaxis.set_major_formatter(FuncFormatter(format_position))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=s.y_ticks))
    if interaction:
        ax.yaxis.set_major_formatter(_coord_formatter(s, s.yformatter))
    for key in ['xlim', 'ylim'] + (['zlim'] if three else []):
        if getattr(s, key) is not None:
            getattr(ax, 'set_' + key)(getattr(s, key))
    if three:
        ax.set_zlabel(s.zlabel or '−log₁₀(P)', **labels)
        ax.zaxis.set_major_locator(MaxNLocator(nbins=s.z_ticks))
        ax.tick_params(axis='z', labelsize=s.tick_size)
        ax.view_init(elev=s.elev, azim=s.azim)
    if s.colorbar:
        cb = fig.colorbar(artist, ax=ax, pad=.12 if three else .02, shrink=.8 if three else 1.)
        cb.set_label(s.colorbar_label, **labels)
        cb.ax.tick_params(labelsize=s.tick_size)


def _point_key(row, interaction):
    key = (row.gene, int(row.aa_pos))
    return key + (int(row.y),) if interaction else key


def _lookup(mapping, row, interaction, default):
    if mapping is None:
        return default
    key = _point_key(row, interaction)
    return mapping.get(key, mapping.get(row.id, default))


class _PositionedAnnotation(Annotation):
    """Place labels in screen space, respecting explicit user offsets."""
    def draw(self, renderer):
        if self._hapy_first:
            self._hapy_boxes.clear()
        self.set_position(self._hapy_offset)
        self.update_positions(renderer)
        if self._hapy_auto:
            initial = Text.get_window_extent(self, renderer)
            points_per_pixel = 72 / self.figure.dpi
            width = initial.width * points_per_pixel
            height = initial.height * points_per_pixel
            dx, dy = self._hapy_offset
            candidates = [(dx, dy)]
            for step in [0, 1, 2, 3, 4]:
                for x in [dx, -width - abs(dx)]:
                    candidates.extend([(x, dy + step * (height + 5)),
                                       (x, -abs(dy) - height - step * (height + 5))])
            anchors = self._hapy_anchor_points
            if anchors.shape[1] == 3:
                px, py, _ = proj3d.proj_transform(*anchors.T, self.axes.get_proj())
                anchors = np.column_stack([px, py])
            anchors = self.axes.transData.transform(anchors)
            radius = self._hapy_anchor_radius * self.figure.dpi / 72 + 3
            anchor_boxes = [Bbox.from_bounds(x-radius, y-radius, 2*radius, 2*radius)
                            for x, y in anchors]
            best, best_cost = self._hapy_offset, float('inf')
            bounds = self.axes.bbox
            for offset in candidates:
                self.set_position(offset)
                self.update_positions(renderer)
                box = Text.get_window_extent(self, renderer).padded(3)
                inside = Bbox.intersection(box, bounds)
                outside = box.width * box.height - (inside.width * inside.height if inside is not None else 0)
                overlap = 0
                for previous in self._hapy_boxes:
                    intersection = Bbox.intersection(box, previous)
                    if intersection is not None:
                        overlap += intersection.width * intersection.height
                hit_overlap = 0
                for anchor_box in anchor_boxes:
                    intersection = Bbox.intersection(box, anchor_box)
                    if intersection is not None:
                        hit_overlap += intersection.width * intersection.height
                cost = hit_overlap * 30 + overlap * 10 + outside * 5 + .01 * ((offset[0]-dx)**2 + (offset[1]-dy)**2)
                if cost < best_cost:
                    best, best_cost = offset, cost
            self.set_position(best)
            self.update_positions(renderer)
        self._hapy_boxes.append(Text.get_window_extent(self, renderer).padded(3))
        super().draw(renderer)


class _Annotation3D(_PositionedAnnotation):
    """Project immediately before drawing, including after changing the view."""
    def __init__(self, text, point, **kwargs):
        super().__init__(text, xy=(0, 0), **kwargs)
        self.point = point

    def draw(self, renderer):
        self.xy = proj3d.proj_transform(*self.point, self.axes.get_proj())[:2]
        super().draw(renderer)


class _ProjectedHighlights(Line2D):
    """Draw selected 3D points above the cloud, without changing data positions."""
    def __init__(self, points, **kwargs):
        super().__init__([], [], **kwargs)
        self.points = points

    def draw(self, renderer):
        x, y, _ = proj3d.proj_transform(*self.points, self.axes.get_proj())
        self.set_data(x, y)
        super().draw(renderer)


def _annotate(ax, frame, selected, s, annotate_p, annotations, offsets, interaction=False, three=False):
    if not annotate_p and annotations is None:
        return
    if s.annotation_style not in {"full", "compact"}:
        raise ValueError("annotation_style must be 'full' or 'compact'.")
    boxes = []
    anchor_columns = ['plot_x', 'y', 'score'] if three else (['plot_x', 'y'] if interaction else ['plot_x', 'score'])
    anchor_points = frame.loc[selected, anchor_columns].to_numpy(dtype=float)
    marker_area = s.marker_size if s.marker_size is not None else (100 if interaction and not three else 25)
    highlight_area = s.highlight_size if s.highlight_size is not None else max(80, marker_area * 2.5)
    created = 0
    for number, (_, row) in enumerate(frame.loc[selected].iterrows()):
        label = (f'{row.gene} pos {int(row.aa_pos)}' if s.annotation_style == 'compact'
                 else f'{row.gene} amino acid position {int(row.aa_pos)}')
        if interaction and s.annotation_partner:
            label += f'\n{row.partner_label}'
        if annotate_p:
            separator = '\n' if s.annotation_style == 'compact' else ', '
            label += separator + f'p = {format_pvalue(row.score, s.p_digits)}'
        if callable(annotations):
            label = annotations(row.copy())
        elif annotations is not None:
            label = _lookup(annotations, row, interaction, label)
        if label is None:
            continue
        # Alternate offsets to reduce common overlaps; users can position every label.
        default = (s.annotation_offset[0], s.annotation_offset[1] + (0 if s.annotation_avoid_overlap else (number % 3) * 16))
        offset = _lookup(offsets, row, interaction, default)
        options = dict(fontsize=s.annotation_fontsize, ha='left', va='bottom',
                       zorder=1000, annotation_clip=False, clip_on=False)
        options.update(s.annotation_kwargs)
        if three:
            annotation = _Annotation3D(str(label), (row.plot_x, row.y, row.score),
                xytext=offset, textcoords='offset points',
                arrowprops=dict(arrowstyle='-', color='0.4', lw=.6) if s.annotation_arrows else None,
                **options)
            ax.add_artist(annotation)
        else:
            y = row.y if interaction else row.score
            annotation = _PositionedAnnotation(str(label), xy=(row.plot_x, y), xytext=offset, textcoords='offset points',
                        arrowprops=dict(arrowstyle='-', color='0.4', lw=.6) if s.annotation_arrows else None,
                        **options)
            ax.add_artist(annotation)
        explicit = _lookup(offsets, row, interaction, None) is not None
        annotation._hapy_anchor_points = anchor_points
        annotation._hapy_anchor_radius = np.sqrt(highlight_area) / 2
        annotation._hapy_boxes = boxes
        annotation._hapy_first = created == 0
        annotation._hapy_offset = offset
        annotation._hapy_auto = s.annotation_avoid_overlap and not explicit and int(selected.sum()) <= 50
        created += 1


def _scatter(ax, frame, s, norm, interaction=False, three=False, highlight=False):
    coordinates = [frame.plot_x, frame.y if interaction else frame.score]
    if three:
        coordinates.append(frame.score)
    size = s.marker_size if s.marker_size is not None else (100 if interaction and not three else 25)
    kwargs = dict(s=(s.highlight_size if s.highlight_size is not None else max(80, size * 2.5)) if highlight else size,
                  marker=s.marker or ('s' if interaction and not three else 'o'),
                  alpha=s.highlight_alpha if highlight else s.alpha, edgecolors=s.highlight_edgecolor if highlight else s.edgecolor,
                  linewidths=s.highlight_linewidth if highlight else s.linewidth, rasterized=s.rasterized)
    if three:
        kwargs['depthshade'] = False
    if highlight:
        fill = s.highlight_color if s.highlight_facecolor is None else s.highlight_facecolor
        kwargs.update(facecolors=fill, edgecolors=s.highlight_edgecolor, zorder=900)
        if three:
            overlay = _ProjectedHighlights(
                (frame.plot_x.to_numpy(), frame.y.to_numpy(), frame.score.to_numpy()),
                linestyle='None', marker=kwargs['marker'], markersize=np.sqrt(kwargs['s']),
                markerfacecolor=fill, markeredgecolor=s.highlight_edgecolor,
                markeredgewidth=s.highlight_linewidth, alpha=s.highlight_alpha,
                zorder=900, rasterized=s.rasterized)
            ax.add_line(overlay)
            return overlay
    else:
        kwargs.update(c=frame.score, cmap=s.cmap, norm=norm)
    return ax.scatter(*coordinates, **kwargs)


def _finish(fig, s):
    if s.save_path is not None:
        path = Path(s.save_path)
        if path.suffix.lower() not in {'.png', '.pdf', '.svg'}:
            raise ValueError('save_path must end with .png, .pdf, or .svg.')
        kwargs = dict(dpi=s.dpi, bbox_inches='tight')
        kwargs.update(s.save_kwargs)
        fig.savefig(path, **kwargs)
    if s.show:
        plt.show()


def plot_aa_association(data, *, variant_col='VARIANT', pval_col=None, log10p_col=None,
                        genes=None, x_axis='amino_acid', highlights=None, highlight_top=False, annotate_p=True,
                        annotations=None, annotation_offsets=None, significance_threshold=None,
                        threshold_log10p=None, threshold_kwargs=None, ax=None, **style):
    """Plot HLA amino acid position associations at genomic coordinates.

    X defaults to amino acid sequence position for one gene; use x_axis="genomic"
    for physical coordinates across multiple genes.
    Y and point colour
    are -log10(P). Highlight labels identify the HLA gene, amino acid sequence
    position, and actual P-value, including probabilities below float precision.

    Parameters
    ----------
    data : pandas.DataFrame
        One position-level test per AA identifier. Input is not modified. Residue
        or exon-suffixed identifiers and repeated tests at the same position are
        rejected; their P-values are never silently combined.
    variant_col : str, default 'VARIANT'
        Column with IDs such as 'AA_DRB1_11_32552129': gene DRB1, amino acid
        sequence position 11, genomic coordinate 32552129 (used on X).
    pval_col : str or None, default None
        Actual P-value column, mutually exclusive with log10p_col. Defaults to
        'LR_p' if neither is supplied. Values must satisfy 0 < P <= 1. Scientific
        notation strings or Decimal values such as '6.09e-412' avoid underflow.
        For mixed standard AA results use pval_col='LRp_Unip' explicitly.
    log10p_col : str or None, default None
        Finite, nonnegative -log10(P) column, e.g. 'META_log10P'. This is positive
        minus-log probability, not log10(P). Use this for underflowed raw P=0.
    genes : str, iterable of str or None, default None
        Optional gene filter, case-insensitive, e.g. 'DRB1' or ['B', 'DPB1'].
        Filtering occurs before top-hit selection.
    x_axis : {'genomic', 'amino_acid', 'equally_spaced'}, default 'amino_acid'
        X coordinate to display. 'amino_acid' uses sequence positions and requires
        a single HLA gene, selected with genes='DRB1', for example. Genomic
        coordinates remain available internally and are never merged by sequence
        position. position_unit='Mb' still formats the interaction partner Y axis
        but does not scale sequence positions. xlim uses sequence units in this mode.
        'equally_spaced' assigns consecutive ordinal X values to distinct AA
        positions, ordered by genomic coordinate (then gene/sequence position
        for ties). Multiple genes are supported; ticks identify gene and sequence
        position. An INFO note states that spacing does not represent distances.
        In that mode xlim and custom formatters use zero-based ordinal indices;
        Mb formatting applies only to partner Y coordinates.
    highlights : iterable of str or (str, int) tuples or None, default None
        Exact position IDs or (gene, amino_acid_sequence_position) selections,
        e.g. [('B', 63), ('DRB1', 11)]. Gene is always part of tuple matching.
        All distinct genomic coordinates matching a tuple are highlighted.
    highlight_top : bool or nonnegative int, default False
        True selects the strongest hit; an integer selects the top N plotted
        rows. False selects none automatically. Combined with manual highlights;
        ties are resolved in input order. No statistical threshold is implied.
    annotate_p : bool, default True
        Annotate highlighted hits with gene, sequence position and actual P-value.
        False disables default labels while preserving highlighting. Custom
        annotations still apply if provided.
    annotations : mapping, callable or None, default None
        Custom complete labels keyed by position ID or (gene, sequence_position).
        Tuple keys take precedence over ID keys. Missing mapping keys use the
        default label. A callable receives a copy of each selected parsed row
        (pandas.Series with id, gene, aa_pos, x, score, value, source_row) and
        returns text, or None to suppress that label. score is -log10(P).
    annotation_offsets : mapping or None, default None
        Per-hit (dx, dy) offsets in display points, using the same keys as
        annotations. Overrides annotation_offset for those hits.
    significance_threshold : float, str, Decimal, iterable or None, default None
        Actual P-value(s) for horizontal threshold lines, e.g. 5e-8 or [0.05, 5e-8].
        Mutually exclusive with threshold_log10p.
    threshold_log10p : float, iterable of float or None, default None
        Horizontal threshold(s) already in finite, nonnegative -log10(P) units.
    threshold_kwargs : dict or None, default None
        Matplotlib Axes.axhline styling; defaults are grey, dashed, width 0.8.
    ax : matplotlib.axes.Axes or None, default None
        Existing 2D axes to use. Otherwise create a new figure and axes.
    **style : keyword arguments
        Shared styling and export options detailed under Other Parameters.

    Other Parameters
    ----------------
    figsize : tuple of (float, float), default (9, 5)
        Width and height in inches for a new figure. Ignored when ax is supplied.
    dpi : int, default 300
        New figure and saved raster resolution in dots per inch.
    title : str or None, default None
        Plot title; None produces no title.
    title_size : float, default 14
        Title font size in points.
    title_weight : str or int, default 'normal'
        Matplotlib font weight, e.g. 'normal', 'bold', or 700.
    xlabel, ylabel, zlabel : str or None, default None
        Override automatic axis labels. zlabel applies only to scatter3d.
    label_size : float, default 11
        Axis-label and colour-bar-label font size in points.
    label_weight : str or int, default 'normal'
        Axis-label and colour-bar-label font weight.
    tick_size : float, default 9
        Axis and colour-bar tick-label font size in points.
    x_ticks, y_ticks, z_ticks : int, defaults 6, 6, 5
        Approximate maximum number of major tick intervals (MaxNLocator).
        z_ticks applies only to scatter3d.
    cmap : str or matplotlib.colors.Colormap, default 'viridis'
        Colour map for -log10(P).
    norm : matplotlib.colors.Normalize or None, default None
        Shared colour normalisation, e.g. Normalize(0, 20), for comparable plots.
        Mutually exclusive with vmin/vmax. An unset norm range is autoscaled by
        Matplotlib; use explicit bounds to guarantee a shared scale.
    vmin, vmax : float or None, default None
        Colour limits in -log10(P) units. Without norm, vmin defaults to zero and
        vmax to max(1, largest plotted score). Must be finite with vmax > vmin.
    marker : str or matplotlib.markers.MarkerStyle or None, default None
        Matplotlib scatter marker. Defaults to 's' for heatmap and 'o' otherwise.
    marker_size : float or None, default None
        Scatter area in points squared; defaults to 100 for heatmap, 25 otherwise.
    alpha : float, default 0.9
        Marker opacity between zero (transparent) and one (opaque).
    edgecolor : Matplotlib colour specification, default 'none'
        Ordinary marker edge colour, e.g. 'black' or an RGB(A) tuple.
    linewidth : float, default 0.5
        Ordinary marker outline width in points; highlights use highlight_linewidth.
    highlight_color : Matplotlib colour specification, default 'gold'
        Default highlight fill colour. Ordinary points retain significance colours.
    highlight_facecolor : Matplotlib colour or None, default None
        Override highlight_color for marker fill. Use 'none' for outline-only hits.
    highlight_edgecolor : Matplotlib colour, default 'red'
        Contrasting outline of highlighted markers.
    highlight_linewidth : float, default 1.2
        Highlight outline width in points.
    highlight_alpha : float, default 1.0
        Highlight opacity, independent of ordinary marker alpha.
    highlight_size : float or None, default None
        Highlight area in points squared; defaults to max(80, 2.5 * ordinary area).
    colorbar : bool, default True
        Add a colour bar to the figure.
    colorbar_label : str, default '−log₁₀(P)'
        Colour-bar label text.
    annotation_style : {'full', 'compact'}, default 'full'
        'compact' uses 'GENE pos N' and the actual P-value on separate lines.
    annotation_partner : bool, default True
        Include partner information in interaction labels. False gives compact
        gene/position/P labels like a publication callout; no data are discarded.
    annotation_avoid_overlap : bool, default True
        Try screen-space label placements on every draw/view change for up to
        50 labels. Explicit annotation_offsets take precedence and stay fixed.
        Placement is heuristic, not guaranteed collision-free in dense figures.
    annotation_fontsize : float, default 9
        Annotation text size in points.
    annotation_offset : tuple of (float, float), default (8, 12)
        Base text displacement (dx, dy) in display points from the hit. Automatic
        placement tries nearby alternatives; with avoidance disabled, labels
        stagger dy by 0, 16, 32 points. annotation_offsets overrides per hit.
    annotation_arrows : bool, default True
        Draw connector lines between labels and points.
    annotation_kwargs : dict, default {}
        Additional Matplotlib annotation text properties, e.g. {'color': 'navy',
        'bbox': {'facecolor': 'white', 'alpha': 0.8}}, or ha/va. Do not supply xy,
        xytext, textcoords, arrowprops or text, which are controlled internally.
    xlim, ylim, zlim : tuple of (float, float) or None, default None
        Axis limits in original data units. Genomic limits are always base pairs (sequence X uses amino acid positions),
        including when position_unit='Mb'. Association Y and scatter3d Z limits
        are -log10(P); heatmap/scatter3d Y limits are partner base pairs.
    position_unit : {'bp', 'Mb'}, default 'bp'
        Display genomic ticks in base pairs or megabases. Does not transform data.
    xformatter, yformatter : matplotlib.ticker.Formatter, callable or None
        Override genomic tick formatting. A callable receives (value, tick_index)
        in original base-pair units and returns text. yformatter applies only to
        interaction genomic Y axes. Defaults are comma-separated bp or decimal Mb.
    reference_build : str or None, default None
        User-supplied assembly label, e.g. 'GRCh38'. Added to automatic genomic
        axis labels only; no assembly inference, validation, or liftover occurs.
    rasterized : bool, default False
        Rasterise scatter artists in vector exports; text and axes remain vector.
    elev, azim : float, defaults 25, -60
        Scatter3d viewing elevation and azimuth in degrees. Ignored by 2D plots.
    missing_color : Matplotlib colour specification, default '#eeeeee'
        Sparse heatmap background for unobserved coordinate combinations.
    save_path : str, pathlib.Path or None, default None
        Optional output filename ending in .png, .pdf, or .svg. Parent directories
        must exist. Figures remain open after saving for further modification.
    save_kwargs : dict, default {}
        Extra matplotlib.figure.Figure.savefig arguments, e.g. transparent=True.
        Defaults include dpi=dpi and bbox_inches='tight'; supplied keys override.
    show : bool, default False
        Call matplotlib.pyplot.show() after optional saving. False supports
        headless use; set MPLBACKEND=Agg before importing HAPY where required.
    p_digits : int, default 3
        Positive number of significant figures in annotated actual P-values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        New figure or the figure owning ax.
    ax : matplotlib.axes.Axes
        The plotted 2D axes; available for further editing.

    Raises
    ------
    TypeError
        data is not a DataFrame, or an unknown styling keyword is supplied.
    ValueError
        Missing columns, malformed/residue-level IDs, duplicate position tests,
        invalid significance, conflicting arguments, wrong axes, or no valid data.

    Warns
    -----
    UserWarning
        Missing identifiers/significance or non-finite log scores are dropped;
        a requested highlight is absent from the plotted data.

    Notes
    -----
    No genome assembly is inferred. Genomic coordinates and sequence positions
    are distinct. Labels format tiny probabilities from the log score without
    exponentiating the entire probability; accuracy is limited by input precision.
    Automatic label placement does not guarantee collision-free placement. Data points
    can overlap at whole-region scale; adjust limits, markers or annotations.

    Examples
    --------
    >>> fig, ax = plot_aa_association(
    ...     df, log10p_col='META_log10P', highlight_top=True, x_axis='genomic',
    ...     highlights=[('B', 63)], position_unit='Mb',
    ...     title='HLA amino acid association', save_path='association.pdf')
    >>> fig, ax = plot_aa_association(
    ...     df, pval_col='LRp_Unip', genes=['B', 'DPB1'], x_axis='genomic',
    ...     highlight_top=3, significance_threshold=5e-8, annotate_p=False)

    See Also
    --------
    plot_aa_interaction : Position-level genetic interaction plots.
    """
    s = _Style(x_axis=x_axis, **style)
    frame = unique_rows(prepare_data(data, variant_col, pval_col, log10p_col, genes))
    frame = _display_positions(frame, s)
    selected = select_hits(frame, highlights, highlight_top)
    norm = _normalization(frame, s)
    fig, ax = _axes(ax, s)
    artist = _scatter(ax, frame, s, norm)
    ax.margins(x=.08, y=.15)
    if selected.any():
        _scatter(ax, frame.loc[selected], s, norm, highlight=True)
    if significance_threshold is not None and threshold_log10p is not None:
        raise ValueError('Supply significance_threshold or threshold_log10p, not both.')
    thresholds = threshold_log10p if threshold_log10p is not None else significance_threshold
    if thresholds is not None:
        thresholds = [thresholds] if np.isscalar(thresholds) else thresholds
        for value in thresholds:
            score = float(value) if threshold_log10p is not None else _log_probability(value)
            if not np.isfinite(score) or score < 0:
                raise ValueError('Threshold -log10(P) must be finite and nonnegative.')
            opts = dict(color='0.5', linestyle='--', linewidth=.8)
            opts.update(threshold_kwargs or {})
            ax.axhline(score, **opts)
    _decorate(fig, ax, artist, s)
    _annotate(ax, frame, selected, s, annotate_p, annotations, annotation_offsets)
    _finish(fig, s)
    return fig, ax


def plot_aa_interaction(data, *, variant_col='AA_VARIANT', partner_pos_col=None,
                        partner_variant_col=None, partner_chrom=None, pval_col=None,
                        log10p_col=None, kind='scatter3d', genes=None, x_axis='amino_acid', highlights=None,
                        highlight_top=False, annotate_p=True, annotations=None,
                        annotation_offsets=None, duplicate='keep', annotate_cells=False,
                        ax=None, **style):
    """Plot HLA amino acid position interactions in 3D or as a sparse heatmap.

    X defaults to HLA amino acid sequence position for one gene; use
    x_axis='genomic' for physical coordinates across genes. Y is the partner's genomic
    coordinate. Point colour and scatter3d Z are -log10(P). Highlight labels
    include gene, amino acid sequence position, partner information and actual P.

    Parameters
    ----------
    data : pandas.DataFrame
        Position-level interaction results, not individual residue tests. The
        original frame is not modified. Distinct observations at the same genomic
        coordinate pair are preserved by default.
    variant_col : str, default 'AA_VARIANT'
        HLA position ID column, e.g. 'AA_DRB1_11_32552129'. For HAPY meta-analysis,
        variant_col='INTERACTION' also accepts 'AA_DRB1_11_32552129*partner':
        the AA part before '*' is parsed automatically.
    partner_pos_col : str or None, default None
        Column of positive integer partner genomic base-pair coordinates, e.g.
        'TCR_POS'. Supply this for rsIDs or other IDs lacking parseable coordinates.
        At least one of partner_pos_col and partner_variant_col must be supplied.
    partner_variant_col : str or None, default None
        Optional partner ID column. Supported coordinate IDs include
        'chr7:38175678', 'chr7_38175678_A_G' and 'SNP_A_7_38175678_exon1'.
        With partner_pos_col, IDs label the partner and any parseable coordinates
        must agree with that column. With 'INTERACTION' IDs the part after '*' is
        used. rsIDs alone cannot be mapped to coordinates.
    partner_chrom : str, int or None, default None
        Explicit partner chromosome, e.g. '7' or 'chr7', for labels/consistency
        checking. Otherwise inferred only from parseable partner IDs. One partner
        chromosome per plot is supported; no assembly is inferred.
    pval_col : str or None, default None
        Actual P-value column, mutually exclusive with log10p_col. Defaults to
        'LR_p' if neither is supplied; use 'I_p' for pairwise Wald results. Values
        must satisfy 0 < P <= 1. Strings or Decimal values preserve tiny raw P.
    log10p_col : str or None, default None
        Column of finite nonnegative -log10(P), e.g. 'META_log10P'. Use this when
        raw probabilities have underflowed to zero. Not a column of log10(P).
    kind : {'scatter3d', 'heatmap'}, default 'scatter3d'
        'scatter3d' maps X/Y/Z to HLA coordinate/partner coordinate/-log10(P).
        'heatmap' draws square points at observed X/Y coordinates, coloured by
        -log10(P). It does not invent rectangular-grid cells or interpolate.
    genes : str, iterable of str or None, default None
        Case-insensitive HLA gene filter; applied before top-hit selection.
    x_axis : {'genomic', 'amino_acid', 'equally_spaced'}, default 'amino_acid'
        X coordinate to display. 'amino_acid' uses sequence positions and requires
        a single HLA gene, selected with genes='DRB1', for example. Genomic
        coordinates remain available internally and are never merged by sequence
        position. position_unit='Mb' still formats the interaction partner Y axis
        but does not scale sequence positions. xlim uses sequence units in this mode.
        'equally_spaced' assigns consecutive ordinal X values to distinct AA
        positions, ordered by genomic coordinate (then gene/sequence position
        for ties). Multiple genes are supported; ticks identify gene and sequence
        position. An INFO note states that spacing does not represent distances.
        In that mode xlim and custom formatters use zero-based ordinal indices;
        Mb formatting applies only to partner Y coordinates.
    highlights : iterable of (str, int, int) tuples or None, default None
        (HLA gene, amino_acid_sequence_position, partner_genomic_position), e.g.
        [('B', 63, 38175678), ('DPB1', 9, 37289819)]. Genomic partner coordinates
        remain in base pairs even when position_unit='Mb'. Matching distinct HLA
        genomic coordinates are all selected; sequence position alone is insufficient.
    highlight_top : bool or nonnegative int, default False
        True selects the strongest interaction; integer N selects the top N plotted
        rows. Manual selections are included too. Ties retain input order.
    annotate_p : bool, default True
        Generate actual-P labels for selected points. False suppresses defaults
        while retaining highlight markers; custom annotations still apply.
    annotations : mapping, callable or None, default None
        Complete custom labels keyed by highlight tuple or parsed AA position ID
        (without '*partner'). Tuple keys take precedence. A callable receives a
        copy of a parsed row (pandas.Series: id, gene, aa_pos, x, y, score,
        partner_label, partner_chrom, value, source_row) and returns text or None.
        score is -log10(P). Custom text replaces the full automatic label.
    annotation_offsets : mapping or None, default None
        Per-hit (dx, dy) offsets in display points, using the same keys as
        annotations. Overrides annotation_offset; useful for nearby hits.
    duplicate : {'keep', 'error', 'max'}, default 'keep'
        Policy for repeated exact (HLA genomic coordinate, partner coordinate)
        pairs, including distinct IDs at the same coordinates. 'keep' preserves
        every observation: scatter3d uses its own Z score, while heatmap warns
        about overlap. No coordinates are shifted. 'error' rejects repeated pairs.
        'max' warns and retains the largest -log10(P) at each pair; ties retain
        the first input row. This is visual aggregation, NOT a combined test.
        Residue-suffixed IDs remain invalid under all policies.
    annotate_cells : bool, default False
        Annotate all observed points instead of only highlighted points, using
        annotate_p/annotations rules. Applies to either interaction mode.
    ax : matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D or None
        Existing axes, or create new axes by default. scatter3d requires 3D axes;
        heatmap requires 2D axes. Existing figure size/layout is retained.
    **style : keyword arguments
        Shared styling and export options detailed under Other Parameters.

    Other Parameters
    ----------------
    figsize : tuple of (float, float), default (9, 5)
        Width and height in inches for a new figure. Ignored when ax is supplied.
    dpi : int, default 300
        New figure and saved raster resolution in dots per inch.
    title : str or None, default None
        Plot title; None produces no title.
    title_size : float, default 14
        Title font size in points.
    title_weight : str or int, default 'normal'
        Matplotlib font weight, e.g. 'normal', 'bold', or 700.
    xlabel, ylabel, zlabel : str or None, default None
        Override automatic axis labels. zlabel applies only to scatter3d.
    label_size : float, default 11
        Axis-label and colour-bar-label font size in points.
    label_weight : str or int, default 'normal'
        Axis-label and colour-bar-label font weight.
    tick_size : float, default 9
        Axis and colour-bar tick-label font size in points.
    x_ticks, y_ticks, z_ticks : int, defaults 6, 6, 5
        Approximate maximum number of major tick intervals (MaxNLocator).
        z_ticks applies only to scatter3d.
    cmap : str or matplotlib.colors.Colormap, default 'viridis'
        Colour map for -log10(P).
    norm : matplotlib.colors.Normalize or None, default None
        Shared colour normalisation, e.g. Normalize(0, 20), for comparable plots.
        Mutually exclusive with vmin/vmax. An unset norm range is autoscaled by
        Matplotlib; use explicit bounds to guarantee a shared scale.
    vmin, vmax : float or None, default None
        Colour limits in -log10(P) units. Without norm, vmin defaults to zero and
        vmax to max(1, largest plotted score). Must be finite with vmax > vmin.
    marker : str or matplotlib.markers.MarkerStyle or None, default None
        Matplotlib scatter marker. Defaults to 's' for heatmap and 'o' otherwise.
    marker_size : float or None, default None
        Scatter area in points squared; defaults to 100 for heatmap, 25 otherwise.
    alpha : float, default 0.9
        Marker opacity between zero (transparent) and one (opaque).
    edgecolor : Matplotlib colour specification, default 'none'
        Ordinary marker edge colour, e.g. 'black' or an RGB(A) tuple.
    linewidth : float, default 0.5
        Ordinary marker outline width in points; highlights use highlight_linewidth.
    highlight_color : Matplotlib colour specification, default 'gold'
        Default highlight fill colour. Ordinary points retain significance colours.
    highlight_facecolor : Matplotlib colour or None, default None
        Override highlight_color for marker fill. Use 'none' for outline-only hits.
    highlight_edgecolor : Matplotlib colour, default 'red'
        Contrasting outline of highlighted markers.
    highlight_linewidth : float, default 1.2
        Highlight outline width in points.
    highlight_alpha : float, default 1.0
        Highlight opacity, independent of ordinary marker alpha.
    highlight_size : float or None, default None
        Highlight area in points squared; defaults to max(80, 2.5 * ordinary area).
    colorbar : bool, default True
        Add a colour bar to the figure.
    colorbar_label : str, default '−log₁₀(P)'
        Colour-bar label text.
    annotation_style : {'full', 'compact'}, default 'full'
        'compact' uses 'GENE pos N' and the actual P-value on separate lines.
    annotation_partner : bool, default True
        Include partner information in interaction labels. False gives compact
        gene/position/P labels like a publication callout; no data are discarded.
    annotation_avoid_overlap : bool, default True
        Try screen-space label placements on every draw/view change for up to
        50 labels. Explicit annotation_offsets take precedence and stay fixed.
        Placement is heuristic, not guaranteed collision-free in dense figures.
    annotation_fontsize : float, default 9
        Annotation text size in points.
    annotation_offset : tuple of (float, float), default (8, 12)
        Base text displacement (dx, dy) in display points from the hit. Automatic
        placement tries nearby alternatives; with avoidance disabled, labels
        stagger dy by 0, 16, 32 points. annotation_offsets overrides per hit.
    annotation_arrows : bool, default True
        Draw connector lines between labels and points.
    annotation_kwargs : dict, default {}
        Additional Matplotlib annotation text properties, e.g. {'color': 'navy',
        'bbox': {'facecolor': 'white', 'alpha': 0.8}}, or ha/va. Do not supply xy,
        xytext, textcoords, arrowprops or text, which are controlled internally.
    xlim, ylim, zlim : tuple of (float, float) or None, default None
        Axis limits in original data units. Genomic limits are always base pairs (sequence X uses amino acid positions),
        including when position_unit='Mb'. Association Y and scatter3d Z limits
        are -log10(P); heatmap/scatter3d Y limits are partner base pairs.
    position_unit : {'bp', 'Mb'}, default 'bp'
        Display genomic ticks in base pairs or megabases. Does not transform data.
    xformatter, yformatter : matplotlib.ticker.Formatter, callable or None
        Override genomic tick formatting. A callable receives (value, tick_index)
        in original base-pair units and returns text. yformatter applies only to
        interaction genomic Y axes. Defaults are comma-separated bp or decimal Mb.
    reference_build : str or None, default None
        User-supplied assembly label, e.g. 'GRCh38'. Added to automatic genomic
        axis labels only; no assembly inference, validation, or liftover occurs.
    rasterized : bool, default False
        Rasterise scatter artists in vector exports; text and axes remain vector.
    elev, azim : float, defaults 25, -60
        Scatter3d viewing elevation and azimuth in degrees. Ignored by 2D plots.
    missing_color : Matplotlib colour specification, default '#eeeeee'
        Sparse heatmap background for unobserved coordinate combinations.
    save_path : str, pathlib.Path or None, default None
        Optional output filename ending in .png, .pdf, or .svg. Parent directories
        must exist. Figures remain open after saving for further modification.
    save_kwargs : dict, default {}
        Extra matplotlib.figure.Figure.savefig arguments, e.g. transparent=True.
        Defaults include dpi=dpi and bbox_inches='tight'; supplied keys override.
    show : bool, default False
        Call matplotlib.pyplot.show() after optional saving. False supports
        headless use; set MPLBACKEND=Agg before importing HAPY where required.
    p_digits : int, default 3
        Positive number of significant figures in annotated actual P-values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        New figure or the figure owning ax.
    ax : matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        Plotted axes; available for further editing and 3D view changes.

    Raises
    ------
    TypeError
        data is not a DataFrame, or an unknown styling keyword is supplied.
    ValueError
        Missing columns, malformed/residue-level IDs, invalid significance or
        coordinates, disagreement between partner coordinates/IDs, multiple partner
        chromosomes, duplicate coordinate pairs with duplicate='error', invalid kind
        or axes, conflicting colour options, or no valid data after filtering.

    Warns
    -----
    UserWarning
        Missing identifiers/significance, non-finite log scores, or missing/non-finite
        partner coordinates are dropped; a requested highlight is absent; explicit
        duplicate='max' aggregation discards repeated coordinate pairs; or
        heatmap points overlap under duplicate='keep'.

    Notes
    -----
    Heatmaps use exact, potentially irregular genomic coordinates. No binning,
    interpolation or automatic P-value combination occurs. Missing combinations
    show missing_color; observed P=1 points remain coloured at score zero. Square
    size is in display points, not genomic bin width, so very close coordinates
    can overlap. Zoom in or reduce marker_size to distinguish them.

    Annotation offsets do not guarantee collision-free placement; supply explicit
    per-hit offsets for dense figures. 3D labels reproject on draw after view changes.
    Each plot covers one partner chromosome. All inputs must share an appropriate
    reference build; reference_build only labels it and does not perform liftover.

    Examples
    --------
    >>> hits = [('B', 63, 38175678), ('DPB1', 9, 37289819)]
    >>> fig, ax = plot_aa_interaction(
    ...     df, partner_pos_col='TCR_POS', log10p_col='META_log10P', x_axis='genomic',
    ...     kind='scatter3d', highlights=hits, elev=30, azim=-45,
    ...     partner_chrom='7', position_unit='Mb', save_path='interactions.pdf')
    >>> from matplotlib.colors import Normalize
    >>> fig, ax = plot_aa_interaction(
    ...     df, partner_variant_col='PARTNER', log10p_col='META_log10P', x_axis='genomic',
    ...     kind='heatmap', norm=Normalize(0, 10), highlights=hits,
    ...     annotation_offsets={('B', 63, 38175678): (12, 20)})

    See Also
    --------
    plot_aa_association : Position-level HLA association scatter plot.
    """
    if kind not in {'scatter3d', 'heatmap'}:
        raise ValueError("kind must be 'scatter3d' or 'heatmap'.")
    s = _Style(x_axis=x_axis, **style)
    frame = prepare_data(data, variant_col, pval_col, log10p_col, genes, interaction=True)
    frame = add_partners(frame, data, partner_pos_col, partner_variant_col, partner_chrom)
    frame = unique_rows(frame, interaction=True, duplicate=duplicate)
    frame = _display_positions(frame, s)
    if kind == 'heatmap' and duplicate == 'keep' and frame.duplicated(['plot_x', 'y']).any():
        warnings.warn(
            "Multiple interactions share heatmap coordinates: all observations are retained, "
            "but overlapping points may hide one another. Filter to specific variants or "
            "explicitly use duplicate='max' to show the strongest per coordinate pair.",
            UserWarning, stacklevel=2)
    selected = select_hits(frame, highlights, highlight_top, interaction=True)
    norm = _normalization(frame, s)
    three = kind == 'scatter3d'
    fig, ax = _axes(ax, s, three)
    if not three:
        ax.set_facecolor(s.missing_color)
    artist = _scatter(ax, frame, s, norm, interaction=True, three=three)
    if not three:
        ax.margins(x=.08, y=.15)
    if selected.any():
        _scatter(ax, frame.loc[selected], s, norm, interaction=True, three=three, highlight=True)
    chromosomes = frame.partner_chrom.dropna().unique()
    _decorate(fig, ax, artist, s, interaction=True, three=three,
              partner_chrom=chromosomes[0] if len(chromosomes) else None)
    labels = np.ones(len(frame), dtype=bool) if annotate_cells else selected
    _annotate(ax, frame, labels, s, annotate_p, annotations, annotation_offsets, interaction=True, three=three)
    _finish(fig, s)
    return fig, ax
