"""
Hand-built polygon approximating the Gulf of Mexico, used by the hurricane
history/histogram scripts to classify a storm position as "in the Gulf"
rather than the NW Caribbean or the Straits of Florida/Atlantic -- two water
bodies a simple lon/lat bounding box cannot separate out (a storm can dip
into the box at Cabo San Antonio, Cuba while still sitting in the Caribbean,
or sit off Key West in the Straits, without ever being in Gulf water).

No shapefile is used -- the boundary is strung between well-known
capes/points that mark the Gulf's natural "gates":

- Yucatan Channel (Gulf <-> Caribbean): Cabo Catoche, MX <-> Cabo San
  Antonio, Cuba.
- Straits of Florida (Gulf <-> Atlantic): Cabo San Antonio, Cuba <-> Key
  West, FL <-> Cape Sable, FL.
- The Yucatan Peninsula's Caribbean (east) coast, Cabo Catoche south to
  roughly Tulum, sits at nearly the same longitude as Cabo Catoche itself
  (the coastline runs N-S there) -- a plain vertical cut from Cabo Catoche
  would misclassify that Caribbean coast, and everything south of it, as
  Gulf water. Progreso and Campeche (both on the peninsula's actual Gulf
  coast) route the boundary around the peninsula instead.

Everything south/east of the Cabo Catoche -> Cabo San Antonio -> Key West ->
Cape Sable chain, or east of the Progreso -> Campeche line, is excluded.
Everything else inside the outer bounding box is kept, including the Bay of
Campeche, which dips south of the Yucatan Channel's latitude but is
unambiguously Gulf water.

This is not survey-grade -- it's a straight-line approximation good enough
to resolve the Caribbean/Straits edge cases the bounding box got wrong.
"""
import numpy as np
from matplotlib.path import Path

# Outer bounding box: [lon_min, lon_max, lat_min, lat_max]. This is also the
# map extent the scripts that import this module plot on -- the polygon
# below only tightens *classification*, not the view.
BOUNDING_BOX = [-98.25, -79.75, 17.75, 31.25]

_LON_MIN, _LON_MAX, _LAT_MIN, _LAT_MAX = BOUNDING_BOX

_CABO_CATOCHE = (-86.79, 21.62)      # NE tip of the Yucatan Peninsula, MX
_CABO_SAN_ANTONIO = (-84.95, 21.87)  # W tip of Cuba
_KEY_WEST = (-81.80, 24.55)
_CAPE_SABLE = (-81.09, 25.13)        # southern tip of the FL mainland
_PROGRESO = (-89.66, 21.28)          # Yucatan's N coast, where Gulf water begins
_CAMPECHE = (-90.53, 19.85)          # Yucatan's W (Gulf) coast

# (lon, lat) vertices, wound so the polygon interior is "the Gulf".
GULF_OF_MEXICO_POLYGON = [
    (_LON_MIN, _LAT_MAX),           # NW corner of the box
    (_LON_MAX, _LAT_MAX),           # NE corner of the box
    _CAPE_SABLE,
    _KEY_WEST,
    _CABO_SAN_ANTONIO,
    _CABO_CATOCHE,
    _PROGRESO,                      # around the peninsula's N coast
    _CAMPECHE,                      # down its W (Gulf) coast
    (_CAMPECHE[0], _LAT_MIN),       # straight down to the box's south edge
    (_LON_MIN, _LAT_MIN),           # SW corner of the box
]

_PATH = Path(GULF_OF_MEXICO_POLYGON)


def gulf_mask(lons, lats):
    """Return a boolean array: True where (lon, lat) falls inside the Gulf polygon."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    points = np.column_stack([lons, lats])
    return _PATH.contains_points(points)


def bounding_box_mask(lons, lats):
    """Return a boolean array: True where (lon, lat) falls inside BOUNDING_BOX.

    This is the coarser rectangular check the polygon above replaced. It's
    kept around for storms that get a deliberate by-hand exception to the
    polygon (see each script's ``force_include_names``) -- e.g. Irma, whose
    only Cat 4 points sit in the Straits of Florida near Key West, which the
    polygon excludes but which some slides want kept in as part of the
    Gulf-adjacent Keys landfall.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    in_lon = (lons >= _LON_MIN) & (lons <= _LON_MAX)
    in_lat = (lats >= _LAT_MIN) & (lats <= _LAT_MAX)
    return in_lon & in_lat
