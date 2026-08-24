#!/usr/bin/env python3
"""
manage_front_users.py — accounts for the Gulf Stream front editor.

There is deliberately no signup page and no password-reset flow: for a small
team this CLI is the entire user-management story, and every extra web-facing
auth surface is another thing to get wrong.

Passwords are always typed at a prompt, never passed as an argument — argv
lands in shell history and is visible in `ps` on a shared server.

Requires MONGODB_URI. The production Mongo host is not reachable from a
laptop: run this on the server, or open your SSH tunnel and point MONGODB_URI
at localhost.

Usage
-----
    python scripts/tools/manage_front_users.py add     --username msmith --name "Mike Smith"
    python scripts/tools/manage_front_users.py add     --username jdoe --role editor --role admin
    python scripts/tools/manage_front_users.py passwd  --username msmith
    python scripts/tools/manage_front_users.py disable --username msmith
    python scripts/tools/manage_front_users.py enable  --username msmith
    python scripts/tools/manage_front_users.py list
"""

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ioos_model_comparisons import users            # noqa: E402
from ioos_model_comparisons.env import load_env     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="create a user")
    a.add_argument("--username", required=True)
    a.add_argument("--name")
    a.add_argument("--email")
    a.add_argument("--role", action="append", default=None,
                   help="repeatable; defaults to 'editor'")

    for name, help_ in (("passwd", "change a password (logs that user out everywhere)"),
                        ("disable", "deactivate a user"),
                        ("enable", "reactivate a user")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--username", required=True)

    sub.add_parser("list", help="list users")
    return p.parse_args()


def _prompt_password():
    """Read a password twice from the tty and confirm they match."""
    for _ in range(3):
        pw = getpass.getpass("Password: ")
        if pw != getpass.getpass("Confirm: "):
            print("  passwords did not match, try again")
            continue
        try:
            users.hash_password(pw)     # validate length/bytes before we commit
        except ValueError as exc:
            print(f"  {exc}")
            continue
        return pw
    print("giving up after 3 attempts")
    sys.exit(1)


def main():
    args = parse_args()
    load_env()

    if not os.getenv("MONGODB_URI"):
        logger.error("MONGODB_URI is not set. Run this on the server, or open "
                     "your SSH tunnel and point MONGODB_URI at localhost.")
        sys.exit(1)

    if args.cmd == "list":
        rows = users.list_users()
        if not rows:
            print("no users (or MongoDB is unreachable — check the warning above)")
            return
        print(f"{'username':<16} {'name':<22} {'roles':<18} {'active':<7} {'last login'}")
        for u in rows:
            print(f"{u.get('username',''):<16} {(u.get('name') or ''):<22} "
                  f"{','.join(u.get('roles') or []):<18} "
                  f"{str(bool(u.get('active', True))):<7} "
                  f"{u.get('last_login_at') or '-'}")
        return

    username = args.username.strip().lower()

    if args.cmd == "add":
        if users.get_user(username):
            logger.error(f"user {username!r} already exists — use `passwd` to "
                         f"change their password")
            sys.exit(1)
        pw = _prompt_password()
        roles = args.role or ["editor"]
        if users.create_user(username, pw, name=args.name, email=args.email,
                             roles=roles):
            print(f"created {username} with roles {roles}")
        else:
            logger.error("could not create the user (is MongoDB reachable?)")
            sys.exit(1)
        return

    if users.get_user(username) is None:
        logger.error(f"no such user {username!r} (or MongoDB is unreachable)")
        sys.exit(1)

    if args.cmd == "passwd":
        pw = _prompt_password()
        if users.set_password(username, pw):
            print(f"password changed for {username}; all their existing "
                  f"sessions are now invalid")
        else:
            sys.exit(1)
    elif args.cmd in ("disable", "enable"):
        want = args.cmd == "enable"
        if users.set_active(username, want):
            print(f"{username} {'enabled' if want else 'disabled'}; existing "
                  f"sessions invalidated")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
