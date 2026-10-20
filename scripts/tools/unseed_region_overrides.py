#!/usr/bin/env python
"""
One-time cleanup: make hurricanes.region_configs hold only real overrides.

seed_region_configs.py used to copy every region's defaults (extent, folder,
name, figure, currents, ...) into MongoDB. Because apply_colorbar_overrides()
replaces whole top-level keys, those copies hide the YAML defaults — editing
config/regions/<key>.yml would have no effect for any seeded key. This
removes ($unset) every top-level key whose stored value equals the YAML
default, leaving deliberate overrides (tuned colorbar limits, editor changes)
in place. Plotting output does not change: the removed values are identical
to what the YAML now provides.

    python scripts/tools/unseed_region_overrides.py            # dry run: report only
    python scripts/tools/unseed_region_overrides.py --apply    # back up, then write

--apply first writes a JSON backup of the whole collection (restore with
mongoimport, or re-insert the documents).
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from ioos_model_comparisons.db import get_client
from ioos_model_comparisons.env import load_env
from ioos_model_comparisons.regions_store import DB_NAME, LIVE_COLL, defaults_for, normalize


def plan(coll):
    out = []
    for doc in coll.find({}, {"_id": 0}):
        region = doc.get("region")
        defaults = defaults_for(region)
        if not defaults:
            out.append((region, [], "not in config/regions — left untouched"))
            continue
        stored = normalize(doc)
        same = sorted(k for k, v in stored.items()
                      if k != "region" and k in defaults and defaults[k] == v)
        kept = sorted(k for k in stored if k != "region" and k not in same)
        out.append((region, same, f"keeps overrides: {', '.join(kept) or '(none)'}"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--backup-dir", default=".", help="where --apply writes the JSON backup")
    args = ap.parse_args(argv)

    load_env()
    client = get_client()
    if client is None:
        sys.exit("MongoDB unavailable (MONGODB_URI unset or unreachable)")
    coll = client[DB_NAME][LIVE_COLL]

    actions = plan(coll)
    for region, unset, note in actions:
        print(f"{region:28s} unset: {', '.join(unset) or '-':60s} {note}")
    total = sum(len(u) for _r, u, _n in actions)
    print(f"\n{total} key(s) equal to their YAML default across {len(actions)} region document(s)")

    if not args.apply:
        print("dry run — re-run with --apply to write")
        return
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = Path(args.backup_dir) / f"region_configs_backup_{stamp}.json"
    backup.write_text(json.dumps(list(coll.find({}, {"_id": 0})), indent=1, default=str))
    print(f"backup written to {backup}")
    for region, unset, _note in actions:
        if unset:
            coll.update_one({"region": region}, {"$unset": {k: "" for k in unset}})
    print("done")


if __name__ == "__main__":
    main()
