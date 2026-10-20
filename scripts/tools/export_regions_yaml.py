#!/usr/bin/env python
"""
One-time migration: dump every region defined in the legacy regions.py
`region_config()` into ioos_model_comparisons/config/regions/<key>.yml, and
write tests/golden/region_config.json — the exact legacy output that
tests/test_region_parity.py holds the YAML-backed region_config() to.

    python scripts/tools/export_regions_yaml.py --legacy path/to/legacy_regions.py

--legacy defaults to ioos_model_comparisons/regions.py; point it at a saved
copy (e.g. `git show <sha>:ioos_model_comparisons/regions.py > /tmp/r.py`)
once regions.py has been replaced by the YAML-backed wrapper.

Each YAML file's content is exactly region_config([key]) for that key, plus an
`order:` field (position in the legacy file — later regions win when a caller
passes several keys, as before).
"""

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_REGIONS = REPO / "ioos_model_comparisons" / "config" / "regions"
GOLDEN = REPO / "tests" / "golden" / "region_config.json"


def encode(o):
    """JSON-safe encoding that keeps tuples and int dict keys distinguishable."""
    if isinstance(o, tuple):
        return {"__tuple__": [encode(v) for v in o]}
    if isinstance(o, list):
        return [encode(v) for v in o]
    if isinstance(o, dict):
        if any(not isinstance(k, str) for k in o):
            return {"__intkeys__": {str(k): encode(v) for k, v in o.items()}}
        return {k: encode(v) for k, v in o.items()}
    return o


def _plain(o):
    """Tuples -> lists and OrderedDict -> dict, for yaml.safe_dump."""
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    return o


class _Dumper(yaml.SafeDumper):
    pass


def _repr_list(dumper, data):
    # Leaf lists ([min, max, stride], extents, figsize) inline; lists of
    # dicts (per-depth limits) as block sequences.
    flow = all(not isinstance(v, (dict, list)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _repr_list)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", default=str(REPO / "ioos_model_comparisons" / "regions.py"))
    ap.add_argument("--no-golden", action="store_true", help="only write the YAML files")
    args = ap.parse_args()

    legacy = Path(args.legacy)
    src = legacy.read_text()
    keys = re.findall(r'^\s{4}key\s*=\s*["\'](\w+)["\']', src, re.M)
    spec = importlib.util.spec_from_file_location("legacy_regions", legacy)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"

    CONFIG_REGIONS.mkdir(parents=True, exist_ok=True)
    golden = {"keys": keys, "list_call": {}, "string_call": {}}
    for i, key in enumerate(keys):
        cfg = mod.region_config([key])
        golden["list_call"][key] = encode(cfg)
        golden["string_call"][key] = encode(mod.region_config(key))
        doc = {"order": i, **_plain(cfg)}
        header = (f"# Region '{key}'. Exported from legacy regions.py (git {sha}) by\n"
                  f"# scripts/tools/export_regions_yaml.py. Edit freely — this file is now the\n"
                  f"# source of truth for the defaults; MongoDB region_configs (the /regions\n"
                  f"# editor) holds runtime overrides on top. Validated by regions_store.validate().\n")
        body = yaml.dump(doc, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
        (CONFIG_REGIONS / f"{key}.yml").write_text(header + body)
    print(f"wrote {len(keys)} region files to {CONFIG_REGIONS}")

    if not args.no_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
        print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
