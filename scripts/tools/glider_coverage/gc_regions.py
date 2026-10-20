"""
Regions for the glider-coverage analysis, shared by every script in this folder.

Each region has:
    name    display name
    bbox    [lon_min, lon_max, lat_min, lat_max]; also the model download box and
            the regular 1/12 deg analysis grid
    inside  f(lon, lat) -> bool array: which points belong to the region (land is
            removed separately by each model's own mask)

Outputs live in <workdir>/<region>/<model>/.
"""
import numpy as np

from ioos_model_comparisons.gulf_of_mexico import BOUNDING_BOX, gulf_mask


def _in_bbox(bbox):
    def inside(lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (lon >= bbox[0]) & (lon <= bbox[1]) & (lat >= bbox[2]) & (lat <= bbox[3])
    return inside


# Caribbean side south of Puerto Rico and Hispaniola: the northern edge (18.25N)
# sits between the islands' south coasts (~18.0N) and the Atlantic north of them
# (~18.4N+), and the eastern edge (62W) stays west of the Lesser Antilles arc.
_CARIB_PR = [-72.0, -62.0, 13.0, 18.25]

REGIONS = {
    "gulf": dict(name="Gulf of Mexico", bbox=list(BOUNDING_BOX), inside=gulf_mask),
    "caribbean_pr": dict(name="Caribbean south of Puerto Rico", bbox=_CARIB_PR, inside=_in_bbox(_CARIB_PR)),
}


def grid(region):
    """Regular 1/12 deg lon/lat vectors over the region's bbox."""
    lon_min, lon_max, lat_min, lat_max = REGIONS[region]["bbox"]
    return (np.round(np.arange(lon_min, lon_max + 1e-6, 1 / 12), 4),
            np.round(np.arange(lat_min, lat_max + 1e-6, 1 / 12), 4))


def center(region):
    lon_min, lon_max, lat_min, lat_max = REGIONS[region]["bbox"]
    return (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
