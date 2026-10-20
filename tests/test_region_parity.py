"""region_config() is now backed by config/regions/*.yml. It must return
exactly what the legacy 1,300-line regions.py function returned — values,
types (tuples for figsize, int keys for limits_by_depth) and key order — for
both call forms. tests/golden/region_config.json is that legacy output,
captured by scripts/tools/export_regions_yaml.py.

When you deliberately change a region's YAML, this test fails for that
region: update its entries in the golden file to match (they use the same
encoding as encode() below). That's the point — config changes stay visible
in review."""

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from ioos_model_comparisons.regions import region_config

GOLDEN = json.loads((Path(__file__).parent / "golden" / "region_config.json").read_text())


def encode(o):
    if isinstance(o, tuple):
        return {"__tuple__": [encode(v) for v in o]}
    if isinstance(o, list):
        return [encode(v) for v in o]
    if isinstance(o, dict):
        if any(not isinstance(k, str) for k in o):
            return {"__intkeys__": {str(k): encode(v) for k, v in o.items()}}
        return {k: encode(v) for k, v in o.items()}
    return o


def roundtrip(o):
    return json.loads(json.dumps(encode(o)))


@pytest.mark.parametrize("key", GOLDEN["keys"])
def test_list_call_matches_legacy(key):
    got = region_config([key])
    assert isinstance(got, OrderedDict)
    assert roundtrip(got) == GOLDEN["list_call"][key]
    assert list(got) == list(GOLDEN["list_call"][key]), "top-level key order changed"


@pytest.mark.parametrize("key", GOLDEN["keys"])
def test_string_call_matches_legacy(key):
    assert roundtrip(region_config(key)) == GOLDEN["string_call"][key]


def test_default_is_gom():
    assert region_config()["name"] == "Gulf of Mexico"


def test_unknown_region_raises():
    with pytest.raises(ValueError, match="Unknown region"):
        region_config(["atlantis"])


def test_returns_independent_copies():
    a = region_config("gom")
    a["extent"][0] = 0
    assert region_config("gom")["extent"][0] == -99


def test_region_catalog_lists_every_region():
    from ioos_model_comparisons.region_catalog import ALL_REGIONS
    assert ALL_REGIONS == GOLDEN["keys"]
    assert "sargasso" in ALL_REGIONS  # was missing from the old hand-typed list
