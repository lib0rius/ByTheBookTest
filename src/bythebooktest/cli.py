"""The ``bythebooktest`` command.

Usage::

    bythebooktest generate --config configs/level1.toml --out out/level1
    bythebooktest prune --source out/level3 --noise-ratio 0.2 --out out/level3-noise20
    bythebooktest run --config configs/d1-2b.toml
    bythebooktest status runs/d1-2b/status.json
    bythebooktest metrics runs/d1-2b
    bythebooktest analysis runs/d1-2b
    bythebooktest probe build | analyze ...

Run ``bythebooktest <command> --help`` for the options of each command.
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from . import __version__, analysis, generate, metrics, probe, prune, run, status


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bythebooktest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")
    for module in (generate, prune, run, status, metrics, analysis, probe):
        module.add_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValidationError as e:
        # Configuration errors: show which key is wrong, without a traceback.
        print(f"bythebooktest {args.command}: invalid configuration\n{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
