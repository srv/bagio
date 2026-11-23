"""Command line: the INS-clock correction's provenance of each mission.

``python -m bagio.ins_clock_survey <mission folder or bags>...`` prints, as
JSON, :meth:`bagio.ins_clock.InsClock.provenance` of each mission's main-bag
clock: each folder argument is a mission (its ``*.bag``), and the bag-file
arguments together form one more. A mission that cannot be read is reported
with its error and the others go on; the exit status is 1 if any failed.

The command lives in its own module, which the package does not import, so
running it does not load :mod:`bagio.ins_clock` twice.
"""

import argparse
import json
import sys
from pathlib import Path

from .ins_clock import InsClock

__all__ = ["main"]


def _mission_bags(args):
    """``(label, bags)`` per mission: each folder argument is a mission (its
    ``*.bag``); the bag-file arguments together form one more. Repeated
    labels get a ``#k`` suffix."""
    missions = []
    loose = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            missions.append((str(p), sorted(p.glob("*.bag"))))
        else:
            loose.append(p)
    if loose:
        missions.append((" ".join(str(p) for p in loose), loose))
    seen = {}
    labelled = []
    for label, bags in missions:
        seen[label] = seen.get(label, 0) + 1
        labelled.append((label if seen[label] == 1 else f"{label} #{seen[label]}",
                         bags))
    return labelled


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m bagio.ins_clock_survey",
        description="Print, as JSON, the INS-clock correction's provenance of "
                    "each mission (folder of bags, or the bag files given).")
    parser.add_argument("missions", nargs="+",
                        help="mission folders, or the bag files of one mission")
    args = parser.parse_args(argv)
    out = {}
    failed = False
    for label, bags in _mission_bags(args.missions):
        try:
            out[label] = InsClock.from_mission(bags).provenance()
        except Exception as exc:   # report and go on to the next mission
            out[label] = {"error": f"{type(exc).__name__}: {exc}"}
            failed = True
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
