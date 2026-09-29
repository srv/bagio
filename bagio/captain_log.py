"""The mission captain's log on ``/rosout``: parks and trigger-service calls.

The COLA2 captain node executes a mission step by step and logs each step on
``/rosout``. Two kinds of entry are read here:

- **Parks** — surface holds, logged from ``Executing mission park wait for N
  seconds`` to ``Mission park wait finalized``. The logged interval is the
  only reliable delimiter of a park: the GPS receiver keeps publishing while
  the vehicle is at the surface, and no timing gap separates park fixes from
  the transit or recovery around them.
- **Trigger-service calls** — mission actions, logged as ``Calling trigger
  service with id <service>`` when the step carrying the action completes
  (e.g. a sensor's ``enable`` or ``start_record_xtf``).

Only messages from a node whose name ends in ``/captain`` count, so the
vehicle namespace (``/sparus2/...``) is not hardcoded. Stamps are the
messages' header stamps, falling back to the bag record time when a stamp is
unset (see :func:`bagio.typestore.header_stamp_ns`).
"""

import re

from .bag_reader import BagSet, bags_with_topics
from .typestore import header_stamp_ns

__all__ = [
    "CAPTAIN_SUFFIX",
    "PARK_DURATION_RE",
    "PARK_END_PREFIX",
    "PARK_START_PREFIX",
    "ROSOUT_SUFFIX",
    "TRIGGER_PREFIX",
    "CaptainLogError",
    "park_intervals",
    "trigger_calls",
]

ROSOUT_SUFFIX = "/rosout"
# Matched by suffix so the vehicle namespace (``/sparus2/...``) is not
# hardcoded.
CAPTAIN_SUFFIX = "/captain"
PARK_START_PREFIX = "Executing mission park wait"
PARK_END_PREFIX = "Mission park wait finalized"
PARK_DURATION_RE = re.compile(r"for\s+(\d+(?:\.\d+)?)\s+seconds")
TRIGGER_PREFIX = "Calling trigger service with id "


class CaptainLogError(RuntimeError):
    """The captain's log is missing or cannot be read reliably."""


def _captain_messages(bag_paths, what):
    """Yield ``(stamp_ns, text)`` for every captain message on ``/rosout``.

    Messages come in bag-record order. Raises :class:`CaptainLogError` when no
    bag carries ``/rosout``, naming ``what`` the caller needed from it.
    """
    bags = bags_with_topics(bag_paths, (ROSOUT_SUFFIX,))
    if not bags:
        raise CaptainLogError(
            f"no bag carries {ROSOUT_SUFFIX} — {what} are recorded there and "
            "cannot be recovered without it")
    with BagSet(bags) as bag_set:
        for t_ns, _topic, msg in bag_set.messages(
                topic_suffixes=(ROSOUT_SUFFIX,)):
            if not str(getattr(msg, "name", "")).endswith(CAPTAIN_SUFFIX):
                continue
            text = getattr(msg, "msg", None)
            if not isinstance(text, str):
                continue
            yield header_stamp_ns(msg, t_ns), text


def park_intervals(bag_paths, log=print):
    """Every ``(t_start_ns, t_end_ns, declared_s)`` park the captain logged.

    A park is a matched ``park wait`` start and ``finalized`` pair. A second
    start before any finalize supersedes the pending one — that park never
    completed, and leaving it pending would let it swallow the next park's end
    marker. ``declared_s`` is the duration the captain announced (``None`` when
    the start line announces none), kept for the record; the interval used is
    the logged pair, so a park cut short is honoured.

    The result is validated to be strictly ascending and non-overlapping.
    Messages arrive in bag-record order while the intervals carry header
    stamps — two different clocks — so a violation means the marker pairing is
    not trustworthy, and selecting a park from it would silently anchor a
    placement on the wrong window. Raises :class:`CaptainLogError` on such a
    violation and when no bag carries ``/rosout``.
    """
    parks, pending, declared = [], None, None
    for stamp, text in _captain_messages(bag_paths, "the park intervals"):
        if text.startswith(PARK_START_PREFIX):
            if pending is not None:
                log("[bagio] warning: a park wait started before the "
                    "previous one finalized; the previous is discarded")
            pending = stamp
            found = PARK_DURATION_RE.search(text)
            declared = float(found.group(1)) if found else None
        elif pending is not None and text.startswith(PARK_END_PREFIX):
            if stamp > pending:
                parks.append((pending, stamp, declared))
            else:
                log(f"[bagio] warning: park finalized at {stamp} is "
                    f"not after its start {pending}; pair discarded")
            pending, declared = None, None
    for earlier, later in zip(parks, parks[1:]):
        if later[0] <= earlier[1]:
            raise CaptainLogError(
                f"the captain's park intervals overlap or are out of order "
                f"({earlier[:2]} then {later[:2]}) — the start/finalize "
                "markers cannot be paired reliably")
    return parks


def trigger_calls(bag_paths, log=print):
    """Every ``(t_ns, service)`` trigger-service call the captain logged.

    ``service`` is the full service id as logged (e.g.
    ``/sparus2/marinesonic_scoutmkii_sidescan/start_record_xtf``); match it by
    suffix, so the vehicle namespace is never hardcoded. The calls are
    returned in stamp order (stable for equal stamps); ``[]`` when the log
    holds none. A call line whose id is empty or is not a single
    whitespace-free name is skipped with a warning rather than reported under
    a guessed name. Raises :class:`CaptainLogError` when no bag carries
    ``/rosout``.
    """
    calls = []
    for stamp, text in _captain_messages(bag_paths, "the trigger calls"):
        if not text.startswith(TRIGGER_PREFIX):
            continue
        tokens = text[len(TRIGGER_PREFIX):].split()
        if len(tokens) != 1:
            log(f"[bagio] warning: trigger call at {stamp} has no single "
                f"service id ({text!r}); skipped")
            continue
        calls.append((stamp, tokens[0]))
    calls.sort(key=lambda call: call[0])
    return calls
