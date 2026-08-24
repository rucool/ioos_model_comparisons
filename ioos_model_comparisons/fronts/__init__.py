"""
Gulf Stream frontal analysis: north wall digitizing and ring detection.

Two data sources, deliberately:

    digitizer  — the north wall from satellite SST (GOES-19). Gradient +
                 Viterbi anchors inside a climatological corridor, which then
                 calibrate a per-longitude wall temperature; the wall itself
                 is traced as a 2-D isotherm contour so it stays continuous
                 through weak-gradient stretches and can wrap meanders.

    eddies     — warm/cold-core rings from satellite altimetry (CMEMS SLA),
                 NOT from SST. Rings are geostrophic features whose sea-level
                 signature survives summer surface heating; SST-based ring
                 detection was measured at 11% next-day persistence against
                 83% for altimetry, and is disabled by default.

Both write GeoJSON plus per-day QC properties. See scripts/fronts/ for the
runners.
"""

from pathlib import Path

# Repo-root outputs/ archive (gitignored), matching where the rest of the
# project keeps generated data. Resolved from the package location so it
# works for an editable install regardless of the working directory.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "gulf_stream_fronts"

__all__ = ["DEFAULT_OUTPUT_DIR"]
