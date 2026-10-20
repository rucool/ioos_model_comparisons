"""
regions.py — per-region plot configuration (extent, folder, colorbar limits,
currents, figure size, ...).

The defaults now live in YAML, one file per region:
ioos_model_comparisons/config/regions/<key>.yml. To add or change a region,
edit/add that file — no code change. (The legacy 1,300-line function that
used to live here was converted by scripts/tools/export_regions_yaml.py, and
tests/test_region_parity.py holds this function to its exact output.)

At runtime most scripts layer MongoDB overrides on top via
db.apply_colorbar_overrides() — the /regions web editor and the colorbar
tuners write those.
"""

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Keys whose values regions.py historically returned as tuples.
_TUPLE_KEYS = {"figsize"}


def _restore_tuples(obj):
    if isinstance(obj, dict):
        return {k: (tuple(v) if k in _TUPLE_KEYS and isinstance(v, list) else _restore_tuples(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_tuples(v) for v in obj]
    return obj


def region_config(regions=None, model=None):
    """
    Return the configuration dict for a region.

    :param regions: a region key ('gom') or a list of keys (['gom']). With
        several keys, the one defined last (file `order`) wins, as before.
    :param model: unused; kept for call-site compatibility.
    :return: OrderedDict with name, folder, extent, figure, sea_surface_height,
        eez, variables and, when configured, currents / salinity_max /
        ocean_heat_content.
    """
    from ioos_model_comparisons.registry import load_registry

    reg = load_registry()
    keys = reg.region_keys()
    regions = regions or ["gom"]

    if isinstance(regions, str):
        if regions in keys:
            regions = [regions]
        else:
            # Legacy behaviour: `key in "string"` substring matching. Kept so an
            # odd caller doesn't break, but flagged — pass an exact key.
            hits = [k for k in keys if k in regions]
            if hits:
                logger.warning("region_config(%r): not a region key; matched %r by "
                               "substring (legacy behaviour) — pass an exact key", regions, hits[-1])
            regions = hits

    hits = [k for k in keys if k in regions]
    if not hits:
        raise ValueError(
            f"Unknown region {regions!r}: no matching region is defined in "
            f"ioos_model_comparisons/config/regions/ (check spelling; keys use "
            f"underscores, e.g. 'mexico_pacific')."
        )
    return OrderedDict(_restore_tuples(reg.region_defaults(hits[-1])))
