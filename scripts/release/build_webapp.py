#!/usr/bin/env python
"""
Build the deployable web app zip: dist/model_comparisons-<version>.zip

    python scripts/release/build_webapp.py                # needs a clean webapp tree
    python scripts/release/build_webapp.py --allow-dirty  # local experiments only

Steps: verify the vendored package (sync_vendor --check), take the version
from `git describe`, and zip exactly the files git knows about under
webapps/model_comparisons/ (tracked, plus untracked-but-not-ignored with
--allow-dirty) — so stray local files, __pycache__, .DS_Store and tests never
ship. A VERSION file is added (/healthz reports it, which is how the installer
confirms the new release is live), plus install_release.sh, the server-side
installer.

The zip unpacks to model_comparisons-<version>/.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP_REL = "webapps/model_comparisons"
DIST = REPO / "dist"
EXCLUDE_DIRS = {"tests", "dev_checks", "__pycache__"}
EXCLUDE_NAMES = {"pytest.ini", ".DS_Store", ".env"}


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout


def app_files(include_untracked):
    args = ["ls-files", "--cached"]
    if include_untracked:
        args += ["--others", "--exclude-standard"]
    files = []
    for rel in git(*args, "--", APP_REL).splitlines():
        p = Path(rel)
        parts = p.relative_to(APP_REL).parts
        if set(parts[:-1]) & EXCLUDE_DIRS or p.name in EXCLUDE_NAMES or p.suffix in (".pyc", ".zip"):
            continue
        if (REPO / rel).is_file():
            files.append(rel)
    return sorted(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from uncommitted/untracked files (never deploy these)")
    ap.add_argument("--out", default=str(DIST))
    args = ap.parse_args(argv)

    if subprocess.run([sys.executable, str(REPO / "scripts/release/sync_vendor.py"), "--check"]).returncode:
        sys.exit("vendored package check failed — run scripts/release/sync_vendor.py")

    dirty = git("status", "--porcelain", "--", APP_REL, "ioos_model_comparisons").strip()
    if dirty and not args.allow_dirty:
        sys.exit("uncommitted changes under the web app or package — commit them, or "
                 "use --allow-dirty for a local test build:\n" + dirty)

    version = git("describe", "--tags", "--always", "--dirty").strip()
    files = app_files(include_untracked=args.allow_dirty)
    required = {f"{APP_REL}/app.py", f"{APP_REL}/wsgi.py",
                f"{APP_REL}/requirements.txt", f"{APP_REL}/ioos_model_comparisons/registry.py"}
    missing = required - set(files)
    if missing:
        sys.exit(f"not tracked by git (commit them first): {sorted(missing)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    top = f"model_comparisons-{version}"
    zpath = out / f"{top}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(REPO / rel, f"{top}/{Path(rel).relative_to(APP_REL).as_posix()}")
        z.writestr(f"{top}/VERSION", version + "\n")
        # The server-side installer travels with every release (see its header).
        info = zipfile.ZipInfo(f"{top}/install_release.sh")
        info.external_attr = 0o755 << 16
        z.writestr(info, (REPO / "scripts/release/install_release.sh").read_text())
    print(f"built {zpath} ({len(files)} files, version {version})")
    return zpath


if __name__ == "__main__":
    main()
