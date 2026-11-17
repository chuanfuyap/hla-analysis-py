"""Shared functions for creating and resolving amino-acid motif labels."""
import numpy as np


def motif_from_sequences(selected):
    """Create a motif label from reference rows with all selected states."""
    rows = np.asarray(selected)
    if rows.ndim != 2 or not len(rows):
        return None
    rows = rows[(rows != "?").all(axis=1)]
    if not len(rows):
        return None
    candidates = sorted({"".join(row) for row in rows})
    return candidates[0] if len(candidates) == 1 else "{" + ";".join(candidates) + "}"


def representative_motif(label):
    """Select a supported representative motif using checkAAblock per position."""
    if label is None:
        return None
    label = label.replace("|", "")
    if not label.startswith("{"):
        return label
    from ..stats.adapters._aa_haplo import checkAAblock
    candidates = [tuple(motif) for motif in label[1:-1].split(";")]
    chosen = [checkAAblock(list(states)) for states in zip(*candidates)]
    if all(isinstance(state, str) and len(state) == 1 for state in chosen):
        if tuple(chosen) in candidates:
            return "".join(chosen)
    return label
