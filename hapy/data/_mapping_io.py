"""Shared QC and progress reports for mapped amino-acid dosages."""
from time import perf_counter

import numpy as np

from .HLAdat import _qc_prob


class MappingReport:
    """Time a mapping operation and report its filtered output."""

    def __init__(self, title, *, simpleQC, allele_filter, verbose, output):
        self.started = perf_counter()
        if output not in {"hladata", "dataframe"}:
            raise ValueError("output must be 'hladata' or 'dataframe'.")
        try:
            threshold = float(allele_filter)
        except (TypeError, ValueError):
            raise ValueError("allele_filter must be a finite number in [0, 0.5).") from None
        if not np.isfinite(threshold) or not 0 <= threshold < 0.5:
            raise ValueError("allele_filter must be a finite number in [0, 0.5).")
        self.simpleQC = simpleQC
        self.allele_filter = threshold
        self.verbose = verbose
        self.processing_started = None
        self._print("---------------------", title, "---------------------")

    def _print(self, *lines):
        if self.verbose:
            for line in lines:
                print(line, flush=True)

    def process(self, frame, n_hla):
        """Filter mapped states, preserving observed values and missingness."""
        self.processing_started = perf_counter()
        self._print(f"Mapping HLA variants:\t{n_hla}",
                    f"Elapsed time for mapping: {self.processing_started - self.started:.4f} seconds")
        if not self.simpleQC:
            return frame
        self._print("----------------------------------------------------",
                    f"PERFORMING SIMPLE MAF FILTER: dropping {100 * self.allele_filter:g}% allele frequency",
                    "----------------------------------------------------")
        result = _qc_prob(frame.T, self.allele_filter, preserve_missing=True).T
        result.attrs = dict(frame.attrs)
        return result

    def finish(self, frame, *, motifs=False):
        """Report retained groups, sample size and processing times."""
        if not self.verbose:
            return
        if motifs:
            count = int(len(frame.columns) > 0)
        else:
            from .hla_mapping import parse_aa_marker
            count = len({parse_aa_marker(marker)["AA_ID"] for marker in frame.columns})
        self._print("---------------------", f"Sample Size:\t {len(frame)}",
                    f"Number of AA variants:\t{count}")
        if motifs:
            self._print(f"Number of motif states:\t{len(frame.columns)}")
        ended = perf_counter()
        self._print("---------------------",
                    f"Elapsed time for processing: {ended - self.processing_started:.4f} seconds",
                    "", f"TOTAL TIME TAKEN: {ended - self.started:.4f} seconds")
