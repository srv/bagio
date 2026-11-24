"""Per-message correction of the INS clock in the navigation stamps.

The INS stamps ``navigator/navigation`` (``cola2_msgs/NavSts``), the
navigator's ``/tf`` edge to the vehicle frame, its raw data and the
navigator's other publications of the same message (odometry, throttled
copies) with its own validity time. Every other sensor stream (cameras,
sonars, DVL, pressure, GPS) is stamped on the vehicle PC's clock. When the INS
clock is not disciplined by a PPS pulse, it drifts from the PC clock by tens
to hundreds of milliseconds within a mission, and wrong resynchronisations
step it by ~0.1 s; the navigation stamps then sit behind every sensor stamp
by that error, which displaces each sample paired with the navigation by
speed × error.

The bag record time is on the PC clock, so for each navigation message

    lag = record time - header stamp = INS clock error + transport,

with a transport of 3.4-4.3 ms per message (3.8 ms typical). The correction is
measured per message on the main recorder's navigation (the recorder that
also carries ``/rosout``):

    lag_f = median of the lag over the 2m+1 messages centred on each one
    corrected stamp = header stamp + (lag_f - TRANSPORT_NS)

with m the number of messages in half of ``WINDOW_S`` (50 at 20 Hz). The
median keeps every clock segment longer than half the window and removes
recorder stalls (late record times) shorter than half the window. The window
is counted in messages, not in record time: it always holds an odd number of
messages, so its median is one of them and a clock step that happens between
two messages stays between those two messages. Near the ends of the recording
the window shrinks symmetrically, so the first and last messages are
filtered less (the very first and last one not at all); the provenance
reports how far they depart from the first and last full-window values.

Messages recorded elsewhere — the same navigation in another recorder, or
another stream on the INS clock — take the correction of the main-bag message
with the identical header stamp, else of the main-bag message nearest in
record time; never an interpolation over header stamps, which step back where
the INS clock steps back. A header stamp that the main bag itself carries
more than once (a clock step of a whole number of message periods re-issues
earlier stamps) is resolved by record time among the messages carrying it,
and refused when the nearest of them is not clearly nearer than the next. A
message not found by stamp and farther than ``MAX_HOLD_S`` from the main
bag's navigation in record time is refused: its correction would be a guess,
and most likely the clock belongs to another recording. The readers
(:meth:`bagio.NavSeries.from_records`, :class:`bagio.TfBuffer`,
:func:`bagio.load_mission_nav`) use ``MAX_HOLD_S``; another hold is passed to
:meth:`InsClock.correct` directly.

All times are integer nanoseconds; floats are refused rather than truncated,
and a lag beyond ``MAX_LAG_NS`` (stamps or record times in other units,
mispaired, or unset) is refused.

Every product records the state of the correction under
``PROVENANCE_KEY``: :meth:`InsClock.provenance` when applied, a copy of
``NOT_APPLIED`` otherwise; :func:`ins_clock_applied` reads it back, and a
record without the key reads as not applied. A GeoTIFF that carries the state
in its own metadata holds the block in its ``INS_CLOCK_TAG`` tag, as the text
:func:`ins_clock_tag_text` gives; it is read back with
``ins_clock_applied({PROVENANCE_KEY: json.loads(text)})``, never by passing
the raster's tags, which do not hold ``PROVENANCE_KEY`` and would read as not
applied.

``python -m bagio.ins_clock_survey <mission folder or bags>...`` prints the
provenance of each mission's clock as JSON (:mod:`bagio.ins_clock_survey`).
"""

import json
import math
import operator
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .bag_reader import BagSet
from .captain_log import ROSOUT_SUFFIX
from .surface_gps import NAV_SUFFIX
from .typestore import stamp_to_ns

__all__ = [
    "GAP_S",
    "INS_CLOCK_TAG",
    "MAX_HOLD_S",
    "MAX_LAG_NS",
    "MAX_REPEATED_SHARE",
    "NOT_APPLIED",
    "PROVENANCE_KEY",
    "RESIDUAL_MS",
    "STEP_MS",
    "TF_CHILD_SUFFIXES",
    "TRANSPORT_NS",
    "WINDOW_S",
    "InsClock",
    "ins_clock_applied",
    "ins_clock_tag_text",
    "main_bags",
]

# Typical transport from the INS to the recorder: the lag measured while the
# INS clock is synchronised by the PPS pulse.
TRANSPORT_NS = 3_800_000
# Running-median window, seconds of record time.
WINDOW_S = 5.0
# Provenance key under which every product records the correction's state.
PROVENANCE_KEY = "ins_clock_correction"
# GeoTIFF tag holding the correction's state as text (``ins_clock_tag_text``),
# for a raster that carries it in its own metadata.
INS_CLOCK_TAG = "INS_CLOCK_CORRECTION"
# The state of a product made without the correction. Read-only: write a copy
# (``dict(NOT_APPLIED)``), which is also what ``json`` can serialise.
NOT_APPLIED = MappingProxyType({"applied": False})
# Dynamic ``/tf`` edges stamped on the INS clock, by child-frame suffix: the
# navigator's edge to the vehicle frame (child ``<robot>/base_link``).
TF_CHILD_SUFFIXES = ("base_link",)
# A jump between consecutive filtered lags larger than this is a clock step.
STEP_MS = 10.0
# A message whose lag departs from the filtered lag by more than this counts
# in the residual.
RESIDUAL_MS = 5.0
# One recorder's navigation repeats a header stamp only at a clock step of a
# whole number of message periods; two recorders' navigation repeat half of
# them. A larger share of repeated stamps refuses.
MAX_REPEATED_SHARE = 0.01
# Largest |record time - header stamp| accepted on any navigation message.
# The INS driver replaces its time by the PC time beyond a 1 s difference and
# recorder stalls last seconds, so a larger lag is not a clock error.
MAX_LAG_NS = 10_000_000_000
# Largest record-time distance, in seconds, from a message not found by stamp
# to the clock's nearest navigation message.
MAX_HOLD_S = 30.0
# Record-time gaps in the clock's navigation longer than this, in seconds, are
# listed in the provenance.
GAP_S = 1.0

_METHOD = "per_message_running_median"
# A transport of a second or more is not a transport.
_MAX_TRANSPORT_NS = 1_000_000_000
# rosbag's split suffix: ``<name>_<index>.bag``; index 0 starts a recording.
_SPLIT_INDEX_RE = re.compile(r"_(\d+)\.bag$")
# Rows of the sliding-window matrix medianed at once (bounds memory).
_MEDIAN_CHUNK = 8192


def _as_int64(values, what):
    """``values`` as int64, refusing non-integer input instead of truncating
    (empty input passes, whatever its dtype)."""
    a = np.asarray(values)
    if a.size == 0:
        return a.astype(np.int64)
    if a.dtype.kind not in "iu":
        raise TypeError(
            f"{what} must be integer nanoseconds, got dtype {a.dtype}")
    return a.astype(np.int64, copy=False)


def _positive_seconds(value, what):
    """``value`` as a positive finite float; bools and non-numbers refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{what} must be a number of seconds, got {value!r}")
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{what} must be positive and finite, got {value!r}")
    return value


def _half_window_messages(record_t_ns, window_s):
    """m: the number of messages in half of ``window_s``, from the median
    interval between consecutive record times."""
    interval_s = float(np.median(np.diff(record_t_ns))) / 1e9
    if interval_s <= 0:
        raise ValueError(
            "InsClock: most consecutive navigation messages share their "
            "record time, so no message interval can be measured")
    m = int(round(window_s / 2.0 / interval_s))
    if m < 1:
        raise ValueError(
            f"InsClock: window_s={window_s} s holds no message on either side "
            f"at a {interval_s * 1e3:.1f} ms message interval")
    return m


def _symmetric_median(values, m):
    """Median of the ``2h+1`` values centred on each, ``h = min(m, i, n-1-i)``.

    The window always holds an odd number of values, so each output is one of
    the inputs; near the ends it shrinks symmetrically instead of becoming
    one-sided, which would move a step there away from its own message.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = values.copy()
    if m <= 0 or n < 3:
        return out
    full = 2 * m + 1
    if n >= full:
        windows = np.lib.stride_tricks.sliding_window_view(values, full)
        for a in range(0, len(windows), _MEDIAN_CHUNK):
            b = min(a + _MEDIAN_CHUNK, len(windows))
            out[m + a:m + b] = np.median(windows[a:b], axis=1)
        edges = list(range(0, m)) + list(range(n - m, n))
    else:
        edges = range(n)
    for i in edges:
        h = min(m, i, n - 1 - i)
        out[i] = np.median(values[i - h:i + h + 1])
    return out


class InsClock:
    """The INS clock error of one recording, per navigation message.

    Build it with :meth:`from_records`, :meth:`from_bags` or
    :meth:`from_mission`; the constructor takes arrays already ordered by
    record time. The per-message arrays, in that order, are exposed read-only
    for inspection: ``record_t_ns``, ``stamp_ns``, ``lag_ns``
    (record - stamp), ``lag_filtered_ns`` and ``corrections_ns``
    (``lag_filtered_ns - transport_ns``); ``half_window_messages`` is the
    median's m and ``repeated_stamps`` the number of messages whose header
    stamp another message also carries.
    """

    def __init__(self, record_t_ns, stamp_ns, *, transport_ns=TRANSPORT_NS,
                 window_s=WINDOW_S, source=None):
        record_t_ns = _as_int64(record_t_ns, "record_t_ns").copy()
        stamp_ns = _as_int64(stamp_ns, "stamp_ns").copy()
        if record_t_ns.ndim != 1 or record_t_ns.shape != stamp_ns.shape:
            raise ValueError(
                "InsClock: record_t_ns and stamp_ns must be 1-D arrays of the "
                f"same length, got shapes {record_t_ns.shape} and "
                f"{stamp_ns.shape}")
        if len(record_t_ns) < 2:
            raise ValueError(
                "InsClock needs at least two navigation messages, got "
                f"{len(record_t_ns)}")
        if np.any(np.diff(record_t_ns) < 0):
            raise ValueError(
                "InsClock: record_t_ns must be in ascending order "
                "(InsClock.from_records orders them)")
        if np.any(stamp_ns <= 0):
            raise ValueError(
                "InsClock: a navigation header stamp is zero or negative "
                "(unset)")
        transport_ns = operator.index(transport_ns)
        if not 0 <= transport_ns < _MAX_TRANSPORT_NS:
            raise ValueError(
                f"InsClock: transport_ns={transport_ns} is not a plausible "
                "transport in nanoseconds")
        if isinstance(source, (str, bytes, os.PathLike)):
            raise TypeError(
                "InsClock: source must be a sequence of bag names, not a "
                f"single {type(source).__name__}")
        self.transport_ns = transport_ns
        self.window_s = _positive_seconds(window_s, "window_s")
        self.source = None if source is None else [str(s) for s in source]
        self.record_t_ns = record_t_ns
        self.stamp_ns = stamp_ns
        self.lag_ns = record_t_ns - stamp_ns
        worst = int(np.argmax(np.abs(self.lag_ns)))
        if abs(int(self.lag_ns[worst])) > MAX_LAG_NS:
            raise ValueError(
                f"InsClock: message {worst} lags its record time by "
                f"{self.lag_ns[worst] / 1e9:.6g} s, beyond the "
                f"{MAX_LAG_NS / 1e9:g} s any clock error reaches: are the "
                "record times and stamps both integer nanoseconds of the "
                "same messages?")
        unique, first, counts = np.unique(
            stamp_ns, return_index=True, return_counts=True)
        repeated = 1.0 - len(unique) / len(stamp_ns)
        if repeated > MAX_REPEATED_SHARE:
            raise ValueError(
                f"InsClock: {repeated:.1%} of the navigation header stamps are "
                "repeated; the records hold the navigation of more than one "
                "recorder. Pass the main bag's navigation only.")
        self.half_window_messages = _half_window_messages(
            record_t_ns, self.window_s)
        filtered = _symmetric_median(self.lag_ns, self.half_window_messages)
        self.lag_filtered_ns = np.rint(filtered).astype(np.int64)
        self.corrections_ns = self.lag_filtered_ns - self.transport_ns
        # Exact-stamp table: stamps carried by one message only. A repeated
        # stamp resolves by record time among the messages carrying it.
        single = counts == 1
        self._stamps_sorted = unique[single]
        self._corrections_by_stamp = self.corrections_ns[first[single]]
        self.repeated_stamps = int(counts[~single].sum())
        self._shared_stamps = unique[~single]
        by_stamp = np.argsort(stamp_ns, kind="stable")
        sorted_stamps = stamp_ns[by_stamp]
        self._carriers = {
            int(s): by_stamp[np.searchsorted(sorted_stamps, s, "left"):
                             np.searchsorted(sorted_stamps, s, "right")]
            for s in self._shared_stamps}
        for name in ("record_t_ns", "stamp_ns", "lag_ns", "lag_filtered_ns",
                     "corrections_ns"):
            getattr(self, name).setflags(write=False)

    @classmethod
    def from_records(cls, records, *, transport_ns=TRANSPORT_NS,
                     window_s=WINDOW_S, source=None):
        """Build from ``(record_t_ns, stamp_ns)`` integer pairs of one
        recorder's ``navigator/navigation``, in any order.

        The pairs are ordered by record time. Raises ``TypeError`` on
        non-integer times and ``ValueError`` on fewer than two messages,
        anything but pairs, a non-positive window, an unset stamp, a lag
        beyond ``MAX_LAG_NS``, or a share of repeated stamps above
        ``MAX_REPEATED_SHARE`` (the navigation of more than one recorder).
        ``source`` names the bags the records come from, for the provenance.
        """
        _positive_seconds(window_s, "window_s")
        pairs = list(records)
        if len(pairs) < 2:
            raise ValueError(
                f"InsClock needs at least two navigation messages, got {len(pairs)}")
        arr = _as_int64(pairs, "records")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "InsClock: records must be (record_t_ns, stamp_ns) pairs, got "
                f"an array of shape {arr.shape}")
        order = np.argsort(arr[:, 0], kind="stable")
        record_t, stamp = arr[order, 0], arr[order, 1]
        return cls(record_t, stamp, transport_ns=transport_ns,
                   window_s=window_s, source=source)

    @classmethod
    def from_bags(cls, bag_paths, *, nav_topic_suffix=NAV_SUFFIX, **kwargs):
        """Build from the navigation topic of ``bag_paths`` (the parts of one
        recording). Reads that topic only. ``source`` defaults to the bags'
        file names."""
        paths = sorted(Path(p) for p in bag_paths)
        records = []
        with BagSet(paths) as bags:
            for t_ns, _topic, msg in bags.messages(
                    topic_suffixes=(nav_topic_suffix,)):
                records.append((t_ns, stamp_to_ns(msg.header.stamp)))
        kwargs.setdefault("source", [p.name for p in paths])
        return cls.from_records(records, **kwargs)

    @classmethod
    def from_mission(cls, bag_paths, *, nav_topic_suffix=NAV_SUFFIX, **kwargs):
        """Build from the main bag among a mission's bags
        (:func:`main_bags`, with the same navigation topic)."""
        return cls.from_bags(
            main_bags(bag_paths, nav_topic_suffix=nav_topic_suffix),
            nav_topic_suffix=nav_topic_suffix, **kwargs)

    def __len__(self):
        return len(self.record_t_ns)

    def correction_ns(self, stamp_ns, record_t_ns, *, max_hold_s=MAX_HOLD_S):
        """Correction (int64 ns) of INS-clock messages, to add to their stamps.

        Vectorised over ``stamp_ns`` and ``record_t_ns`` (integer ns,
        broadcast). A stamp carried by exactly one of this clock's navigation
        messages takes that message's correction. A stamp carried by several
        takes the correction of the one nearest in record time;
        ``ValueError`` when that one is not clearly nearer (farther than half
        the distance to the next). Any other stamp takes the correction of the
        navigation message nearest in record time, the earlier one on a tie;
        ``ValueError`` when it lies more than ``max_hold_s`` from every
        navigation message in record time. Scalar inputs return an ``int``.
        """
        hold_ns = int(_positive_seconds(max_hold_s, "max_hold_s") * 1e9)
        stamp = _as_int64(stamp_ns, "stamp_ns")
        record_t = _as_int64(record_t_ns, "record_t_ns")
        stamp, record_t = np.broadcast_arrays(stamp, record_t)
        n = len(self.record_t_ns)
        n_exact = len(self._stamps_sorted)
        if n_exact:
            i = np.clip(np.searchsorted(self._stamps_sorted, stamp), 0,
                        n_exact - 1)
            exact = self._stamps_sorted[i] == stamp
            by_stamp = self._corrections_by_stamp[i]
        else:
            exact = np.zeros(stamp.shape, dtype=bool)
            by_stamp = np.zeros(stamp.shape, dtype=np.int64)
        j = np.searchsorted(self.record_t_ns, record_t)
        before = np.clip(j - 1, 0, n - 1)
        after = np.clip(j, 0, n - 1)
        d_before = np.abs(record_t - self.record_t_ns[before])
        d_after = np.abs(self.record_t_ns[after] - record_t)
        nearest = np.where(d_before <= d_after, before, after)
        shared = (np.isin(stamp, self._shared_stamps) if self._carriers
                  else np.zeros(stamp.shape, dtype=bool))
        far = ~exact & ~shared & (np.minimum(d_before, d_after) > hold_ns)
        if np.any(far):
            raise ValueError(
                f"InsClock: {int(far.sum())} message(s) not found by stamp lie "
                f"more than {max_hold_s:g} s from this clock's navigation in "
                f"record time (clock source: {self.source}); is the clock "
                "from another recording?")
        out = np.where(exact, by_stamp, self.corrections_ns[nearest])
        if np.any(shared):
            flat_out = out.reshape(-1)
            flat_stamp = stamp.reshape(-1)
            flat_record = record_t.reshape(-1)
            for k in np.flatnonzero(shared):
                flat_out[k] = self._shared_correction(int(flat_stamp[k]),
                                                      int(flat_record[k]))
        return int(out) if out.ndim == 0 else out

    def _shared_correction(self, stamp, record_t):
        """Correction for a stamp several navigation messages carry: that of
        the carrier nearest in record time, when clearly nearest."""
        carriers = self._carriers[stamp]
        dist = np.abs(self.record_t_ns[carriers] - record_t)
        order = np.argsort(dist, kind="stable")
        nearest, runner_up = dist[order[0]], dist[order[1]]
        if 2 * nearest > runner_up:
            raise ValueError(
                f"InsClock: header stamp {stamp} is carried by {len(carriers)} "
                f"navigation messages, {nearest / 1e6:.1f} ms and "
                f"{runner_up / 1e6:.1f} ms from this message in record time; "
                "which one it copies is unclear")
        return int(self.corrections_ns[carriers[order[0]]])

    def correct(self, stamp_ns, record_t_ns, *, max_hold_s=MAX_HOLD_S):
        """Corrected stamps (int64 ns): ``stamp_ns + correction_ns(...)``."""
        corrected = (_as_int64(stamp_ns, "stamp_ns")
                     + self.correction_ns(stamp_ns, record_t_ns,
                                          max_hold_s=max_hold_s))
        return int(corrected) if np.ndim(corrected) == 0 else corrected

    def provenance(self):
        """JSON-ready record of this correction, for ``PROVENANCE_KEY``.

        - ``steps``: every jump larger than ``STEP_MS`` between consecutive
          filtered lags (record time of the later message, signed size); a
          recorder stall longer than half the window shows as a pair of
          opposite steps.
        - ``residual``: the largest departure of the lag from the filtered
          lag, and the count of messages departing by more than
          ``RESIDUAL_MS`` (stalls the median removed).
        - ``end_departure_ms``: how far the filtered lags of the first and of
          the last m messages, filtered over shrinking windows, depart from
          the first and the last full-window value (the m-th from each end;
          for a series shorter than 2m+1, its last and first value); a
          recorder backlog at the start or end of the recording shows here,
          as does a real clock step within m messages of an end.
        - ``repeated_stamps``: messages whose header stamp another message
          also carries.
        - ``gaps``: record-time gaps longer than ``GAP_S`` between
          consecutive navigation messages (record time after the gap, length).
        """
        from . import __version__

        n, m = len(self), self.half_window_messages
        lag_f_ms = self.lag_filtered_ns / 1e6
        corr_ms = self.corrections_ns / 1e6
        jumps_ms = np.diff(self.lag_filtered_ns) / 1e6
        at = np.flatnonzero(np.abs(jumps_ms) > STEP_MS)
        resid_ms = np.abs(self.lag_ns - self.lag_filtered_ns) / 1e6
        ref_start, ref_end = min(m, n - 1), max(n - 1 - m, 0)
        gaps_ns = np.diff(self.record_t_ns)
        gap_at = np.flatnonzero(gaps_ns > GAP_S * 1e9)
        return {
            "applied": True,
            "method": _METHOD,
            "transport_ms": self.transport_ns / 1e6,
            "window_s": self.window_s,
            "window_messages": 2 * m + 1,
            "source": None if self.source is None else list(self.source),
            "bagio_version": __version__,
            "messages": n,
            "correction_ms": {
                "min": round(float(corr_ms.min()), 3),
                "p50": round(float(np.median(corr_ms)), 3),
                "max": round(float(corr_ms.max()), 3),
            },
            "steps": [{"record_t_ns": int(self.record_t_ns[k + 1]),
                       "size_ms": round(float(jumps_ms[k]), 3)} for k in at],
            "residual": {
                "max_abs_ms": round(float(resid_ms.max()), 3),
                "count_above_threshold": int(np.sum(resid_ms > RESIDUAL_MS)),
                "threshold_ms": RESIDUAL_MS,
            },
            "end_departure_ms": {
                "start": round(float(np.abs(
                    lag_f_ms[:ref_start + 1] - lag_f_ms[ref_start]).max()), 3),
                "end": round(float(np.abs(
                    lag_f_ms[ref_end:] - lag_f_ms[ref_end]).max()), 3),
            },
            "repeated_stamps": self.repeated_stamps,
            "gaps": [{"record_t_ns": int(self.record_t_ns[k + 1]),
                      "length_s": round(float(gaps_ns[k]) / 1e9, 3)}
                     for k in gap_at],
        }


def ins_clock_applied(record):
    """Whether the INS-clock correction was applied to a product.

    ``record`` is a product's metadata, an artifact or a placement slot (a
    mapping). ``False`` when ``PROVENANCE_KEY`` is absent; ``ValueError`` when
    its block is not a mapping with a boolean ``applied``.
    """
    if not isinstance(record, Mapping):
        raise TypeError(
            f"ins_clock_applied: expected a mapping, got {type(record).__name__}")
    if PROVENANCE_KEY not in record:
        return False
    block = record[PROVENANCE_KEY]
    if not isinstance(block, Mapping) or not isinstance(block.get("applied"), bool):
        raise ValueError(
            f"Malformed {PROVENANCE_KEY!r} block: expected a mapping with a "
            f"boolean 'applied', got {block!r}")
    return block["applied"]


def ins_clock_tag_text(block):
    """``block`` as the canonical JSON text of the ``INS_CLOCK_TAG`` tag.

    ASCII, with sorted keys and no whitespace between tokens, so the text does
    not depend on the order of the keys. ``block`` is what
    :meth:`InsClock.provenance` returns, or ``dict(NOT_APPLIED)``; it is not
    checked here (:func:`ins_clock_applied` refuses a malformed one when it is
    read back). ``TypeError`` on anything ``json`` cannot serialise,
    ``NOT_APPLIED`` itself (a read-only mapping) included; ``ValueError`` on a
    NaN or an infinity.
    """
    return json.dumps(block, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def main_bags(bag_paths, *, nav_topic_suffix=NAV_SUFFIX):
    """The parts of the main bag among a mission's bags, sorted.

    The main bag is the recording that carries both the navigation and
    ``/rosout`` (checked by topic). Its parts are told apart from another
    recording by rosbag's split suffix: a part named ``…_0.bag``, or a bag
    without a split suffix, starts a recording. Raises ``ValueError`` when
    no bag carries both topics, or when they belong to more than one
    recording (e.g. a folder holding several missions).
    """
    paths = sorted(Path(p) for p in bag_paths)
    candidates = []
    for path in paths:
        with BagSet([path]) as bags:
            topics = [c.topic for _, c in bags.connections()]
        if (any(t.endswith(nav_topic_suffix) for t in topics)
                and any(t.endswith(ROSOUT_SUFFIX) for t in topics)):
            candidates.append(path)
    if not candidates:
        raise ValueError(
            f"No main bag among {len(paths)} bag(s): none carries both "
            f"'{nav_topic_suffix}' and '{ROSOUT_SUFFIX}'")
    starts = []
    for path in candidates:
        m = _SPLIT_INDEX_RE.search(path.name)
        if m is None or int(m.group(1)) == 0:
            starts.append(path.name)
    if len(starts) != 1:
        raise ValueError(
            f"Expected one main-bag recording, found {len(starts)} "
            f"(first parts: {starts}; main-bag parts: "
            f"{[p.name for p in candidates]})")
    return candidates
