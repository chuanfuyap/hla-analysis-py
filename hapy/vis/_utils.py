"""Parsing, significance and highlighting shared by HLA position plots."""
from decimal import Decimal, InvalidOperation, localcontext
import re
import warnings

import numpy as np
import pandas as pd

_AA = re.compile(r'^AA_([A-Za-z0-9]+)_(-?\d+)_(\d+)$')
_PARTNER = re.compile(r'^(?:chr)?([0-9]+|X|Y|MT|M)[:_](\d+)(?:[:_].*)?$', re.I)
_LEGACY = re.compile(r'^(?:SNP|SNPS)_[^_]+_([0-9]+|X|Y|MT|M)_(\d+)(?:_.*)?$', re.I)


def parse_aa_identifier(value):
    """Parse an exact AA_GENE_SEQUENCEPOSITION_GENOMICPOSITION position ID.

    Suffixes are rejected: a residue-specific row is not a position-level test.
    Coordinates are returned as supplied; no reference assembly is inferred.
    """
    match = _AA.fullmatch(str(value))
    if not match or int(match[3]) <= 0:
        raise ValueError(f"Invalid AA position identifier {value!r}; expected "
                         "AA_GENE_AAPOS_GENOMICPOS with a positive genomic coordinate. "
                         "Residue/exon suffixes are not accepted as position-level tests.")
    return match[1].upper(), int(match[2]), int(match[3])


def parse_partner_identifier(value):
    """Parse chr7:38175678, chr7_38175678_A_G or SNP-style coordinate IDs."""
    match = _PARTNER.fullmatch(str(value)) or _LEGACY.fullmatch(str(value))
    if not match or int(match[2]) <= 0:
        raise ValueError(f"Cannot parse partner coordinate from {value!r}; supply partner_pos_col.")
    return match[1].upper().removeprefix('CHR'), int(match[2])


def _log_probability(value):
    try:
        p = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid P-value {value!r}.") from exc
    if not p.is_finite() or not Decimal(0) < p <= Decimal(1):
        raise ValueError("P-values must satisfy 0 < P <= 1; for underflowed P=0, "
                         "supply a finite -log10(P) column instead.")
    with localcontext() as ctx:
        ctx.prec = 40
        return float(-p.log10())


def format_pvalue(log10p, digits=3):
    """Format a probability from -log10(P), without exponentiating tiny P-values.

    Returns plain Unicode text, e.g. '6.09 × 10⁻⁴¹²'. Precision is limited by the
    supplied log value. Raw scientific-notation strings are parsed using Decimal.
    """
    value = float(log10p)
    if not np.isfinite(value) or value < 0:
        raise ValueError('-log10(P) must be finite and nonnegative.')
    if digits < 1:
        raise ValueError('p_digits must be positive.')
    if value == 0:
        return '1'
    exponent = int(np.floor(-value))
    mantissa = float(f'{10 ** (-value - exponent):.{digits}g}')
    if mantissa >= 10:
        mantissa /= 10
        exponent += 1
    if exponent >= -2:
        return f'{mantissa * 10 ** exponent:.{digits}g}'
    superscript = str(exponent).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{mantissa:.{digits}g} × 10{superscript}'


def prepare_data(data, variant_col, pval_col, log10p_col, genes, interaction=False):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame.')
    if log10p_col is not None and pval_col is not None:
        raise ValueError('Supply pval_col or log10p_col, not both.')
    value_col = log10p_col or pval_col or 'LR_p'
    for col in [variant_col, value_col]:
        if col not in data:
            raise ValueError(f'Missing required column {col!r}.')
    # Use only private columns, preserving the original input and its column names.
    frame = pd.DataFrame({'id': data[variant_col].to_numpy(), 'value': data[value_col].to_numpy(),
                          'source_row': np.arange(len(data))})
    missing = frame[['id', 'value']].isna().any(axis=1)
    if missing.any():
        warnings.warn(f'Dropped {int(missing.sum())} rows with missing identifiers or significance.', UserWarning, stacklevel=3)
    frame = frame.loc[~missing].copy()
    if interaction:
        # Meta-analysis stores AA_position*partner in its INTERACTION key.
        frame["id"] = [str(v).split("*", 1)[0] for v in frame.id]
    parsed = [parse_aa_identifier(v) for v in frame.id]
    for i, key in enumerate(['gene', 'aa_pos', 'x']):
        frame[key] = [p[i] for p in parsed]
    if log10p_col is not None:
        try:
            values = pd.to_numeric(frame.value, errors='raise').to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError('-log10(P) values must be numeric.') from exc
        finite = np.isfinite(values)
        if (~finite).any():
            warnings.warn(f'Dropped {int((~finite).sum())} non-finite -log10(P) values.', UserWarning, stacklevel=3)
        frame = frame.loc[finite].copy()
        values = values[finite]
        if (values < 0).any():
            raise ValueError('-log10(P) must be nonnegative.')
        frame['score'] = values
    elif pd.api.types.is_numeric_dtype(frame.value):
        # Numeric inputs already have floating-point precision. Avoid millions
        # of expensive Decimal.log10 calls; reserve Decimal for string/object P.
        values = frame.value.to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0) & (values <= 1)
        if not valid.all():
            _log_probability(values[np.flatnonzero(~valid)[0]])
        frame['score'] = -np.log10(values)
    else:
        frame['score'] = [_log_probability(v) for v in frame.value]
    if genes is not None:
        selected = [genes] if isinstance(genes, str) else genes
        frame = frame[frame.gene.isin([str(g).upper() for g in selected])].copy()
    if frame.empty:
        raise ValueError('No valid observations remain after filtering.')
    return frame.reset_index(drop=True)


def add_partners(frame, data, partner_pos_col, partner_variant_col, partner_chrom):
    if partner_pos_col is None and partner_variant_col is None:
        raise ValueError('Supply partner_pos_col or partner_variant_col.')
    for col in [partner_pos_col, partner_variant_col]:
        if col is not None and col not in data:
            raise ValueError(f'Missing partner column {col!r}.')
    frame = frame.copy()
    partner_columns = list(dict.fromkeys(c for c in [partner_pos_col, partner_variant_col] if c is not None))
    rows = data[partner_columns].iloc[frame.source_row.to_numpy()]
    ids = rows[partner_variant_col].to_numpy() if partner_variant_col else [None] * len(rows)
    positions, chromosomes, labels, keep = [], [], [], []
    for i, identifier in enumerate(ids):
        if isinstance(identifier, str) and "*" in identifier:
            identifier = identifier.split("*", 1)[1]
        raw_pos = rows.iloc[i][partner_pos_col] if partner_pos_col else None
        if (partner_pos_col and pd.isna(raw_pos)) or (not partner_pos_col and pd.isna(identifier)):
            keep.append(False); positions.append(np.nan); chromosomes.append(None); labels.append('')
            continue
        parsed = None
        if identifier is not None and not pd.isna(identifier):
            try:
                parsed = parse_partner_identifier(identifier)
            except ValueError:
                if partner_pos_col is None:
                    raise
        if partner_pos_col:
            try:
                pos = float(raw_pos)
            except (ValueError, TypeError) as exc:
                raise ValueError('Partner genomic positions must be positive integers.') from exc
            if not np.isfinite(pos):
                keep.append(False); positions.append(np.nan); chromosomes.append(None); labels.append('')
                continue
            if pos <= 0 or pos != int(pos):
                raise ValueError('Partner genomic positions must be positive integers.')
            pos = int(pos)
            if parsed and parsed[1] != pos:
                raise ValueError('Partner identifier and partner_pos_col coordinates disagree.')
        else:
            pos = parsed[1]
        chrom = str(partner_chrom).upper().removeprefix('CHR') if partner_chrom is not None else (parsed[0] if parsed else None)
        if parsed and partner_chrom is not None and parsed[0] != chrom:
            raise ValueError('partner_chrom disagrees with parsed partner identifiers.')
        coordinate = f'chr{chrom}:{pos}' if chrom else f'genomic position {pos}'
        if identifier is None or pd.isna(identifier):
            label = coordinate
        elif parsed is not None:
            # Coordinate-only aliases are displayed canonically. IDs containing
            # allele/context information already include their coordinate.
            coordinate_only = re.fullmatch(r'(?:chr)?(?:[0-9]+|X|Y|MT|M)[:_]\d+', str(identifier), re.I)
            label = coordinate if coordinate_only else str(identifier)
        else:
            label = f'{identifier} ({coordinate})'
        positions.append(pos); chromosomes.append(chrom); labels.append(label); keep.append(True)
    frame['y'] = positions
    frame['partner_chrom'] = chromosomes
    frame['partner_label'] = labels
    if not all(keep):
        warnings.warn(f'Dropped {keep.count(False)} rows with missing/non-finite partner coordinates.', UserWarning, stacklevel=3)
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError('No valid interactions remain.')
    if frame.partner_chrom.dropna().nunique() > 1:
        raise ValueError('Partner coordinates span multiple chromosomes; filter to one chromosome per plot.')
    return frame


def select_hits(frame, highlights, highlight_top, interaction=False):
    mask = np.zeros(len(frame), dtype=bool)
    if isinstance(highlight_top, bool):
        n = int(highlight_top)
    elif isinstance(highlight_top, (int, np.integer)) and highlight_top >= 0:
        n = int(highlight_top)
    else:
        raise ValueError('highlight_top must be True, False, or a nonnegative integer.')
    mask[frame.score.nlargest(min(n, len(frame))).index] = True
    for hit in ([] if highlights is None else highlights):
        if isinstance(hit, str):
            if interaction:
                raise ValueError('Interaction highlights must be (gene, amino_acid_position, partner_position) tuples.')
            found = frame.id.eq(hit).to_numpy()
        else:
            size = 3 if interaction else 2
            if len(hit) != size:
                raise ValueError(f'Highlight must have {size} elements.')
            found = (frame.gene.eq(str(hit[0]).upper()) & frame.aa_pos.eq(int(hit[1]))).to_numpy()
            if interaction:
                found &= frame.y.eq(int(hit[2])).to_numpy()
        if not found.any():
            warnings.warn(f'Requested highlight {hit!r} was not found.', UserWarning, stacklevel=3)
        mask |= found
    return mask


def unique_rows(frame, interaction=False, duplicate='error'):
    if duplicate not in {'keep', 'error', 'max'}:
        raise ValueError("duplicate must be 'keep', 'error', or 'max'.")
    if not interaction:
        if frame.duplicated(['gene', 'aa_pos', 'x']).any():
            raise ValueError('Multiple rows for an AA position: provide one position-level test, not separate residue tests.')
        return frame
    cells = ['x', 'y']
    duplicates = frame.duplicated(cells, keep=False)
    if duplicates.any() and duplicate != "keep":
        if duplicate == 'error':
            raise ValueError(
                f"Multiple interactions share genomic coordinates ({int(frame.duplicated(cells).sum())} additional rows). "
                "Supply one test per cell or explicitly set duplicate='max' to retain the most significant per coordinate pair.")
        warnings.warn("duplicate='max' retains only the largest -log10(P) per exact genomic coordinate pair; this is visual aggregation, not a combined test.", UserWarning, stacklevel=3)
        frame = frame.sort_values('score', ascending=False, kind='stable').drop_duplicates(cells).sort_values('source_row')
    return frame.reset_index(drop=True)
