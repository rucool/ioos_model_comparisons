#!/usr/bin/env python3
"""
fronts_preflight.py — check a host is ready for the fronts/regions stack.

Run this on the digitizer host and on the web host BEFORE the first real
deployment. It only reads, plus one index creation on a scratch collection it
then drops — nothing in the real collections is touched.

    python scripts/tools/fronts_preflight.py
    python scripts/tools/fronts_preflight.py --role web       # web host checks
    python scripts/tools/fronts_preflight.py --role digitizer # cron host checks

Exit code is 0 if everything needed for that role passes, 1 otherwise, so it
can gate a deploy script.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ioos_model_comparisons.env import load_env          # noqa: E402

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
_status = {"fail": 0, "warn": 0}


def report(level, label, detail=""):
    if level is FAIL:
        _status["fail"] += 1
    elif level is WARN:
        _status["warn"] += 1
    print(f"[{level}] {label}" + (f"\n         {detail}" if detail else ""))


def check_env(role):
    uri = os.getenv("MONGODB_URI")
    report(OK if uri else FAIL, "MONGODB_URI is set",
           "" if uri else "export it, or set it in the systemd unit's Environment=")
    if role in ("web", "all"):
        key = os.getenv("FLASK_SECRET_KEY")
        report(OK if key else FAIL, "FLASK_SECRET_KEY is set",
               "" if key else
               "without it the editor returns 503 and nobody can sign in; the "
               "public dashboard still serves. It must be IDENTICAL in every "
               "gunicorn worker, so set it in the unit, not per process.")
        if key and len(key) < 32:
            report(WARN, "FLASK_SECRET_KEY is short",
                   "use at least 32 chars: python -c \"import secrets;print(secrets.token_hex(32))\"")
    return bool(uri)


def check_mongo(role):
    from ioos_model_comparisons.db import get_client
    client = get_client()
    if client is None:
        report(FAIL, "MongoDB reachable",
               "get_client() returned None. From a laptop the production host "
               "needs an SSH tunnel; on the server check the URI and firewall.")
        return None
    try:
        info = client.server_info()
        report(OK, f"MongoDB reachable (server {info.get('version','?')})")
    except Exception as exc:
        report(FAIL, "MongoDB reachable", str(exc)[:160])
        return None

    db = client["hurricanes"]
    # read permission on the collection plotting already depends on
    try:
        n = db["region_configs"].count_documents({})
        report(OK if n else WARN, f"read hurricanes.region_configs ({n} docs)",
               "" if n else "empty — run scripts/tools/seed_region_configs.py")
    except Exception as exc:
        report(FAIL, "read hurricanes.region_configs", str(exc)[:160])

    # write + index permission, on a scratch collection we clean up
    scratch = db["_preflight_scratch"]
    try:
        scratch.insert_one({"probe": True})
        scratch.create_index("probe", background=True)
        scratch.drop()
        report(OK, "write + createIndex permission",
               "this is the capability that could not be tested locally")
    except Exception as exc:
        report(FAIL, "write + createIndex permission",
               f"{str(exc)[:160]}\n         the app can read but not create "
               f"collections/indexes; ask for readWrite on `hurricanes`")

    for coll in ("front_walls", "front_rings", "front_overlays",
                 "front_users", "region_config_versions"):
        try:
            n = db[coll].count_documents({})
            report(OK, f"hurricanes.{coll}: {n} docs")
        except Exception as exc:
            report(FAIL, f"hurricanes.{coll}", str(exc)[:120])

    if role in ("web", "all"):
        try:
            n = db["front_users"].count_documents({"active": True})
            report(OK if n else FAIL, f"active editor accounts: {n}",
                   "" if n else "create one: python scripts/tools/manage_front_users.py add --username you")
        except Exception:
            pass
    return client


def check_files(role):
    if role not in ("digitizer", "all"):
        return
    from ioos_model_comparisons.fronts import DEFAULT_OUTPUT_DIR
    d = Path(DEFAULT_OUTPUT_DIR)
    if not d.is_dir():
        report(WARN, f"output dir {d} missing",
               "created on the first digitizer run; nothing to backfill yet")
        return
    walls = list(d.glob("gulf_stream_north_wall_*.geojson"))
    overlays = list(d.glob("gulf_stream_overlay_*_*.png"))
    report(OK, f"{len(walls)} wall file(s), {len(overlays)} overlay PNG(s) in {d.name}/",
           "run backfill_fronts_to_mongo.py to load these" if walls else "")


def check_imports(role):
    mods = ["pymongo", "numpy", "matplotlib"]
    if role in ("web", "all"):
        mods += ["flask", "flask_login"]
    if role in ("digitizer", "all"):
        mods += ["xarray", "scipy", "contourpy", "copernicusmarine"]
    import importlib.util as u
    missing = [m for m in mods if u.find_spec(m) is None]
    report(OK if not missing else FAIL,
           f"python packages for role '{role}'",
           "" if not missing else f"missing: {', '.join(missing)} — pip install -r requirements.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["web", "digitizer", "all"], default="all")
    args = ap.parse_args()
    load_env()
    print(f"preflight for role: {args.role}\n" + "-" * 60)
    check_imports(args.role)
    if check_env(args.role):
        check_mongo(args.role)
    check_files(args.role)
    print("-" * 60)
    print(f"{_status['fail']} failure(s), {_status['warn']} warning(s)")
    sys.exit(1 if _status["fail"] else 0)


if __name__ == "__main__":
    main()
