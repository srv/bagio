"""Synthetic tests of the INS-clock correction (no dataset needed).

A synthetic mission is a 20 Hz navigation stream. Each message is valid at a
true PC-clock instant ``truth``; the INS stamps it ``truth - err`` (its clock
behind by ``err``) and the recorder records it at ``truth + transport``
(3.8 ms plus jitter) plus any stall delay. A correct correction turns every
stamp back into ``truth``.
"""

import json
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest

import bagio
from bagio.ins_clock import (
    MAX_LAG_NS,
    NOT_APPLIED,
    PROVENANCE_KEY,
    TF_CHILD_SUFFIXES,
    TRANSPORT_NS,
    InsClock,
    _symmetric_median,
    ins_clock_applied,
    main_bags,
)
from bagio.ins_clock_survey import main
from bagio.nav_series import NavSeries
from bagio.surface_gps import load_mission_nav
from bagio.tf_buffer import TfBuffer

from conftest import (BASE_NS, GPS_NED_TOPIC, NAV_TOPIC, ROSOUT_TOPIC,
                      gps_ned, header, log_msg, navsts)

DT_NS = 50_000_000          # 20 Hz
MS = 1_000_000
S = 1_000_000_000


def synth(err_ms, jitter_ms=0.0, delay_ms=None, seed=0):
    """``(record_t, stamp, truth)`` int64 arrays for per-message INS clock
    errors ``err_ms``, uniform transport jitter of ``±jitter_ms`` around
    3.8 ms, and optional per-message recorder delays ``delay_ms``."""
    err = np.asarray(err_ms, dtype=np.float64)
    n = len(err)
    rng = np.random.default_rng(seed)
    truth = BASE_NS + np.arange(n, dtype=np.int64) * DT_NS
    transport = TRANSPORT_NS + np.rint(
        rng.uniform(-jitter_ms, jitter_ms, n) * MS).astype(np.int64)
    delay = (np.zeros(n, dtype=np.int64) if delay_ms is None
             else np.rint(np.asarray(delay_ms) * MS).astype(np.int64))
    stamp = truth - np.rint(err * MS).astype(np.int64)
    return truth + transport + delay, stamp, truth


def pairs_of(record_t, stamp):
    return list(zip(record_t.tolist(), stamp.tolist()))


def clock_of(record_t, stamp, **kwargs):
    return InsClock.from_records(pairs_of(record_t, stamp), **kwargs)


def error_ms(clock, record_t, stamp, truth):
    return (clock.correct(stamp, record_t) - truth) / MS


def three_segments(first, second, n=1200, a=130.0, b=0.0, c=120.0):
    """INS errors of ``a`` ms before message ``first``, ``b`` until
    ``second``, ``c`` after: a dive and a surfacing."""
    err = np.full(n, b)
    err[:first] = a
    err[second:] = c
    return err


def brute_median(values, m):
    """Reference: median over ``h = min(m, i, n-1-i)`` on each side."""
    n = len(values)
    return np.array([np.median(values[i - min(m, i, n - 1 - i):
                                      i + min(m, i, n - 1 - i) + 1])
                     for i in range(n)])


# --------------------------------------------------------------------------
# The filter


def test_constant_lag_corrects_every_message_by_lag_minus_transport():
    record_t, stamp, truth = synth(np.full(2000, 50.0))
    clock = clock_of(record_t, stamp)

    lag_ns = 50 * MS + TRANSPORT_NS
    assert np.all(clock.lag_filtered_ns == lag_ns)
    assert np.all(clock.corrections_ns == lag_ns - TRANSPORT_NS)
    assert np.array_equal(clock.correct(stamp, record_t), truth)


def test_drift_with_jitter_is_followed_per_message():
    minutes = 55
    n = minutes * 60 * 20
    t_min = np.arange(n) * DT_NS / 6e10
    record_t, stamp, truth = synth(16.0 + 0.715 * t_min, jitter_ms=0.45)
    clock = clock_of(record_t, stamp)

    err = error_ms(clock, record_t, stamp, truth)
    assert np.abs(err).max() <= 0.5
    minute = np.floor(t_min).astype(int)
    per_minute = [err[minute == m].mean() for m in range(minutes)]
    assert np.abs(per_minute).max() <= 0.1
    # The jitter is filtered out: unfiltered, its spread would be ~0.26 ms.
    assert np.std(err) < 0.1
    assert clock.provenance()["steps"] == []


@pytest.mark.parametrize("first, second", [(400, 800), (20, 1180)],
                         ids=["mid-series", "near-both-ends"])
def test_each_clock_step_lands_at_its_own_message(first, second):
    record_t, stamp, truth = synth(three_segments(first, second),
                                   jitter_ms=0.45)
    # The second step (the clock falling behind by 120 ms) makes the
    # recorded stamps step back.
    assert stamp[second] < stamp[second - 1]
    clock = clock_of(record_t, stamp)

    err = error_ms(clock, record_t, stamp, truth)
    assert np.abs(err).max() <= 0.5
    assert np.all(np.diff(clock.correct(stamp, record_t)) > 0)
    steps = clock.provenance()["steps"]
    assert [s["record_t_ns"] for s in steps] == [int(record_t[first]),
                                                 int(record_t[second])]
    assert steps[0]["size_ms"] == pytest.approx(-130.0, abs=1.0)
    assert steps[1]["size_ms"] == pytest.approx(120.0, abs=1.0)


def test_a_step_of_whole_message_periods_keeps_every_message():
    # A 150 ms step at 20 Hz re-issues the stamps of the three messages
    # before it; each must still resolve to its own message.
    record_t, stamp, truth = synth(three_segments(400, 800, c=150.0),
                                   jitter_ms=0.45)
    assert np.array_equal(stamp[800:803], stamp[797:800])
    clock = clock_of(record_t, stamp)

    assert clock.repeated_stamps == 6
    assert clock.provenance()["repeated_stamps"] == 6
    assert np.abs(error_ms(clock, record_t, stamp, truth)).max() <= 0.5
    nav = NavSeries.from_records(nav_records(record_t, stamp), clock)
    assert len(nav) == len(stamp)


@pytest.fixture
def whole_period_step():
    """A 150 ms step at message 800: 797-799 and 800-802 share stamps."""
    record_t, stamp, truth = synth(three_segments(400, 800, c=150.0),
                                   jitter_ms=0.45)
    return clock_of(record_t, stamp), record_t, stamp, truth


@pytest.mark.parametrize("offset_ms", [30, -30])
def test_copies_of_shared_stamps_resolve_to_their_own_message(
        whole_period_step, offset_ms):
    clock, record_t, stamp, truth = whole_period_step
    k = np.arange(797, 803)
    copies = clock.correct(stamp[k], record_t[k] + offset_ms * MS)
    assert np.array_equal(copies, stamp[k] + clock.corrections_ns[k])
    # Another recorder's copies, merged with the main bag's, collapse onto
    # the main bag's corrected stamps.
    records = nav_records(record_t, stamp) + [
        (int(r) + offset_ms * MS, navsts(int(s), float(j), 0.0))
        for j, r, s in zip(k, record_t[k], stamp[k])]
    nav = NavSeries.from_records(records, clock)
    assert len(nav) == len(stamp)
    assert np.abs((nav.t_ns - truth) / MS).max() <= 0.5


def test_a_copy_between_two_carriers_is_refused(whole_period_step):
    clock, record_t, stamp, _ = whole_period_step
    # 70 ms after 797 and 80 ms before 800, which carry the same stamp.
    with pytest.raises(ValueError, match="unclear"):
        clock.correct(int(stamp[797]), int(record_t[797]) + 70 * MS)


def _ramp(n_stall):
    """Recorder delays of a stall: 200 ms on its first message falling to
    30 ms on its last, as a buffer flushing."""
    return np.linspace(200.0, 30.0, n_stall)


def test_stalls_shorter_than_half_the_window_are_removed():
    n = 2400
    delay = np.zeros(n)
    delay[400:420] = _ramp(20)        # 1 s
    delay[1200:1240] = _ramp(40)      # 2 s
    record_t, stamp, truth = synth(np.full(n, 40.0), jitter_ms=0.45,
                                   delay_ms=delay)
    clock = clock_of(record_t, stamp)

    assert np.abs(error_ms(clock, record_t, stamp, truth)).max() <= 0.5
    prov = clock.provenance()
    assert prov["steps"] == []
    assert prov["residual"]["count_above_threshold"] == 60


def test_a_long_stall_shows_as_a_pair_of_opposite_steps():
    n = 2400
    delay = np.zeros(n)
    delay[1000:1080] = _ramp(80)      # 4 s, longer than half the window
    record_t, stamp, _ = synth(np.full(n, 40.0), jitter_ms=0.45,
                               delay_ms=delay)
    steps = clock_of(record_t, stamp).provenance()["steps"]

    assert len(steps) == 2
    assert steps[0]["size_ms"] > 10 and steps[1]["size_ms"] < -10


def test_a_backlog_at_the_start_shows_as_an_end_departure():
    # The recorder holds the first 1.4 s and writes it at once, in order.
    n = 1200
    record_t, stamp, truth = synth(np.full(n, 40.0), jitter_ms=0.45)
    held = int(np.sum(truth < truth[0] + 1400 * MS))
    record_t[:held] = truth[0] + 1400 * MS + np.arange(held) * 100_000
    assert np.all(np.diff(record_t) > 0)
    ends = clock_of(record_t, stamp).provenance()["end_departure_ms"]
    assert ends["start"] > 100.0
    assert ends["end"] < 1.0


@pytest.mark.parametrize("n", [2, 7, 100, 9000])
def test_the_median_matches_a_brute_force_reference(n):
    lag = np.random.default_rng(n).normal(40.0, 5.0, n)
    assert np.array_equal(_symmetric_median(lag, 50), brute_median(lag, 50))


def test_records_in_any_order_give_the_same_clock():
    record_t, stamp, _ = synth(three_segments(400, 800), jitter_ms=0.45)
    ordered = clock_of(record_t, stamp)
    shuffled = pairs_of(record_t, stamp)
    np.random.default_rng(1).shuffle(shuffled)
    clock = InsClock.from_records(shuffled)
    assert np.array_equal(clock.corrections_ns, ordered.corrections_ns)
    assert np.array_equal(clock.record_t_ns, ordered.record_t_ns)


def test_record_time_gaps_are_listed():
    record_t, stamp, _ = synth(np.full(400, 40.0))
    record_t[200:] += 3 * S
    gaps = clock_of(record_t, stamp).provenance()["gaps"]
    assert gaps == [{"record_t_ns": int(record_t[200]),
                     "length_s": pytest.approx(3.05, abs=1e-3)}]


# --------------------------------------------------------------------------
# The lookup of other INS-clock messages


@pytest.fixture
def stepped():
    record_t, stamp, truth = synth(three_segments(400, 800), jitter_ms=0.45)
    return clock_of(record_t, stamp), record_t, stamp


def test_a_known_stamp_takes_its_own_correction(stepped):
    clock, record_t, stamp = stepped
    # 800 and 801 are stepped-back stamps, below those of 798 and 799; the
    # lookup is by identity, whatever record time comes with the stamp.
    for k in (100, 800, 801):
        assert clock.correction_ns(int(stamp[k]), int(record_t[0])) == int(
            clock.corrections_ns[k])


def test_an_unknown_stamp_takes_the_nearest_record_time(stepped):
    clock, record_t, stamp = stepped
    absent = int(stamp[800]) + 1
    assert clock.correction_ns(absent, int(record_t[600])) == int(
        clock.corrections_ns[600])
    # Beyond the clock's span, within the hold, the first or last message.
    assert clock.correction_ns(absent, int(record_t[0]) - 10 * S) == int(
        clock.corrections_ns[0])
    assert clock.correction_ns(absent, int(record_t[-1]) + 10 * S) == int(
        clock.corrections_ns[-1])


def test_a_record_time_tie_goes_to_the_earlier_message():
    # Without jitter, consecutive record times are exactly 50 ms apart.
    record_t, stamp, _ = synth(three_segments(400, 800))
    clock = clock_of(record_t, stamp)
    tie = int(record_t[399]) + DT_NS // 2
    assert tie - int(record_t[399]) == int(record_t[400]) - tie
    assert clock.correction_ns(1, tie) == int(clock.corrections_ns[399])


def test_a_message_far_from_the_clock_is_refused(stepped):
    clock, record_t, stamp = stepped
    absent = int(stamp[0]) + 1
    clock.correction_ns(absent, int(record_t[-1]) + 29 * S)
    with pytest.raises(ValueError, match="another recording"):
        clock.correction_ns(absent, int(record_t[-1]) + 31 * S)
    # A known stamp needs no record time near the clock.
    assert clock.correction_ns(int(stamp[5]), int(record_t[-1]) + 10**6 * S) == int(
        clock.corrections_ns[5])
    # The hold is a parameter.
    clock.correction_ns(absent, int(record_t[-1]) + 31 * S, max_hold_s=40.0)


def test_lookup_is_vectorised_and_scalar_gives_int(stepped):
    clock, record_t, stamp = stepped
    out = clock.correction_ns(stamp[:6].reshape(2, 3), record_t[:6].reshape(2, 3))
    assert out.shape == (2, 3) and out.dtype == np.int64
    assert np.array_equal(out.ravel(), clock.corrections_ns[:6])
    # A scalar record time broadcasts against an array of stamps.
    assert clock.correction_ns(stamp[:4], int(record_t[0])).shape == (4,)
    assert type(clock.correction_ns(int(stamp[3]), int(record_t[3]))) is int
    assert type(clock.correct(int(stamp[3]), int(record_t[3]))) is int


# --------------------------------------------------------------------------
# Inputs refused


def test_float_times_are_refused_not_truncated(stepped):
    clock, record_t, stamp = stepped
    seconds = (record_t / 1e9).tolist()
    with pytest.raises(TypeError, match="integer nanoseconds"):
        InsClock.from_records(list(zip(seconds, stamp.tolist())))
    with pytest.raises(TypeError, match="integer nanoseconds"):
        clock.correction_ns(stamp[:3], record_t[:3] / 1e9)
    with pytest.raises(TypeError, match="integer nanoseconds"):
        clock.correct(stamp[:3].astype(float), record_t[:3])


def test_unset_stamps_and_impossible_lags_are_refused():
    record_t, stamp, _ = synth(np.full(100, 40.0))
    zeroed = stamp.copy()
    zeroed[10] = 0
    with pytest.raises(ValueError, match="zero or negative"):
        clock_of(record_t, zeroed)
    shifted = stamp.copy()
    shifted[10] -= MAX_LAG_NS
    with pytest.raises(ValueError, match="integer nanoseconds of the same"):
        clock_of(record_t, shifted)


def test_stamps_in_whole_seconds_are_blamed_on_their_units():
    record_t, stamp, _ = synth(np.full(100, 40.0))
    with pytest.raises(ValueError, match="integer nanoseconds of the same"):
        clock_of(record_t, stamp // S)


def test_empty_lookups_return_empty(stepped):
    clock, _, _ = stepped
    assert clock.correction_ns([], []).shape == (0,)
    assert clock.correct([], []).dtype == np.int64


def test_records_must_be_pairs():
    record_t, stamp, _ = synth(np.full(100, 40.0))
    triples = [(r, s, 0) for r, s in pairs_of(record_t, stamp)]
    with pytest.raises(ValueError, match="pairs"):
        InsClock.from_records(triples)


def test_from_records_refuses_two_recorders():
    record_t, stamp, _ = synth(np.full(400, 50.0))
    pairs = pairs_of(record_t, stamp)
    resent = [(r + 100_000, s) for r, s in pairs]
    with pytest.raises(ValueError, match="more than one recorder"):
        InsClock.from_records(pairs + resent)


@pytest.mark.parametrize("window_s, error", [
    (0.0, ValueError), (-5.0, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (0.01, ValueError), (True, TypeError),
    ("5", TypeError)])
def test_from_records_refuses_an_unusable_window(window_s, error):
    record_t, stamp, _ = synth(np.full(10, 50.0))
    with pytest.raises(error, match="window_s"):
        clock_of(record_t, stamp, window_s=window_s)


@pytest.mark.parametrize("n", [0, 1])
def test_from_records_needs_two_messages(n):
    record_t, stamp, _ = synth(np.full(max(n, 1), 50.0))
    with pytest.raises(ValueError, match="at least two"):
        InsClock.from_records(pairs_of(record_t, stamp)[:n])


def test_the_constructor_validates_its_arrays():
    record_t, stamp, _ = synth(np.full(100, 40.0))
    with pytest.raises(ValueError, match="ascending"):
        InsClock(record_t[::-1], stamp[::-1])
    with pytest.raises(TypeError, match="sequence of bag names"):
        InsClock(record_t, stamp, source="main.bag")
    with pytest.raises(TypeError):
        InsClock(record_t, stamp, transport_ns=3.8)
    with pytest.raises(ValueError, match="plausible transport"):
        InsClock(record_t, stamp, transport_ns=S)
    with pytest.raises(ValueError, match="same length"):
        InsClock(record_t, stamp[:-1])
    both = np.concatenate([record_t, record_t + 100_000])
    order = np.argsort(both, kind="stable")
    with pytest.raises(ValueError, match="more than one recorder"):
        InsClock(both[order], np.concatenate([stamp, stamp])[order])


# --------------------------------------------------------------------------
# NavSeries.from_records


def nav_records(record_t, stamp):
    return [(int(r), navsts(int(s), float(k), 0.0))
            for k, (r, s) in enumerate(zip(record_t, stamp))]


_NAV_FIELDS = ("t_ns", "north", "east", "depth", "altitude", "roll", "pitch",
               "yaw", "yaw_rate", "vel_x", "lat", "lon")


def test_nav_from_records_without_clock_equals_from_messages(stepped):
    _, record_t, stamp = stepped
    records = nav_records(record_t, stamp)
    a = NavSeries.from_messages([m for _, m in records])
    b = NavSeries.from_records(records)

    for field in _NAV_FIELDS:
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert (a.origin_lat, a.origin_lon) == (b.origin_lat, b.origin_lon)
    assert a.ins_clock is None and b.ins_clock is None


def test_nav_from_records_with_clock_corrects_sorts_and_dedups(stepped):
    clock, record_t, stamp = stepped
    records = nav_records(record_t, stamp)
    # A resend of message 10, recorded later with another value: dropped.
    records.append((int(record_t[10]) + MS, navsts(int(stamp[10]), 999.0, 0.0)))
    plain = NavSeries.from_records(records)
    nav = NavSeries.from_records(records, clock)

    assert len(nav) == len(stamp)
    assert np.array_equal(nav.t_ns, np.sort(clock.correct(stamp, record_t)))
    # Corrected, the messages keep their order (north = message index);
    # sorted on the recorded stamps, the stepped-back ones jump ahead.
    assert np.array_equal(nav.north, np.arange(len(stamp), dtype=float))
    assert not np.all(np.diff(plain.north) > 0)
    assert nav.ins_clock == clock.provenance()
    assert nav.ins_clock["applied"] is True


def test_nav_from_records_with_clock_refuses_empty_and_float_input(stepped):
    clock, record_t, stamp = stepped
    with pytest.raises(ValueError, match="No NavSts"):
        NavSeries.from_records([], clock)
    floats = [(float(r) / 1e9, m) for r, m in nav_records(record_t, stamp)]
    with pytest.raises(TypeError):
        NavSeries.from_records(floats, clock)


def test_nav_series_accepts_only_an_applied_provenance(stepped):
    clock, record_t, stamp = stepped
    samples = NavSeries.from_records(nav_records(record_t, stamp)).samples
    for bad in (True, dict(NOT_APPLIED), {"applied": "yes"}):
        with pytest.raises(ValueError, match="ins_clock"):
            NavSeries(list(samples), 39.5, 2.4, ins_clock=bad)
    with pytest.raises(TypeError, match="JSON"):
        NavSeries(list(samples), 39.5, 2.4,
                  ins_clock=types.MappingProxyType(clock.provenance()))
    prov = clock.provenance()
    nav = NavSeries(list(samples), 39.5, 2.4, ins_clock=prov)
    prov["messages"] = -1        # the series keeps its own copy
    assert nav.ins_clock == clock.provenance()


# --------------------------------------------------------------------------
# TfBuffer


def tf_msg(t_ns, parent, child, x):
    ns = types.SimpleNamespace
    return ns(transforms=[ns(
        header=ns(stamp=header(t_ns).stamp, frame_id=parent),
        child_frame_id=child,
        transform=ns(translation=ns(x=x, y=0.0, z=0.0),
                     rotation=ns(x=0.0, y=0.0, z=0.0, w=1.0)))])


@pytest.fixture
def tf_case():
    """A 50 ms INS error; the vehicle edge moves 1 m per message, the
    world -> world_ned edge (PC clock) holds 1000 + k."""
    record_t, stamp, truth = synth(np.full(400, 50.0))
    clock = clock_of(record_t, stamp)
    rows = []
    for k in range(400):
        rows.append((int(record_t[k]), False, tf_msg(
            int(stamp[k]), "world_ned", "sparus2/base_link", float(k))))
        rows.append((int(record_t[k]), False, tf_msg(
            int(truth[k]), "world", "world_ned", 1000.0 + k)))
    rows.append((int(record_t[0]), True, tf_msg(
        0, "sparus2/base_link", "sparus2/camera", 0.25)))
    return clock, rows, record_t, stamp, truth


def build_tf(rows, clock=None, **kwargs):
    buf = TfBuffer(clock=clock, **kwargs)
    for record_t, static, msg in rows:
        buf.add_tf_message(msg, static=static,
                           record_t_ns=None if clock is None else record_t)
    return buf


def x_of(buf, target, source, t_ns):
    return buf.lookup(target, source, int(t_ns))[0, 3]


def test_tf_listed_edge_is_corrected(tf_case):
    clock, rows, _, stamp, truth = tf_case
    plain, corrected = build_tf(rows), build_tf(rows, clock)
    k = 200
    assert x_of(plain, "world_ned", "sparus2/base_link", stamp[k]) == pytest.approx(k)
    # The pose valid at truth[k] is message k's once corrected; uncorrected
    # the lookup lands one 50 ms message later.
    assert x_of(corrected, "world_ned", "sparus2/base_link", truth[k]) == pytest.approx(k)
    assert x_of(plain, "world_ned", "sparus2/base_link", truth[k]) == pytest.approx(k + 1)


def test_tf_stamp_unknown_to_the_clock_is_corrected_by_record_time(tf_case):
    clock, _, record_t, stamp, truth = tf_case
    buf = TfBuffer(clock=clock)
    for k in range(400):
        # 1 ns off every navigation stamp: found by record time instead.
        buf.add_tf_message(tf_msg(int(stamp[k]) + 1, "world_ned",
                                  "sparus2/base_link", float(k)),
                           record_t_ns=int(record_t[k]))
    assert x_of(buf, "world_ned", "sparus2/base_link", truth[200] + 1) == pytest.approx(200)


def test_tf_unlisted_dynamic_edge_is_unchanged(tf_case):
    clock, rows, _, _, truth = tf_case
    plain, corrected = build_tf(rows), build_tf(rows, clock)
    for k in (10, 200, 390):
        t = int(truth[k])
        assert np.array_equal(plain.lookup("world", "world_ned", t),
                              corrected.lookup("world", "world_ned", t))


def test_tf_no_listed_suffix_means_no_correction(tf_case):
    clock, rows, _, stamp, _ = tf_case
    buf = build_tf(rows, clock, clock_child_suffixes=())
    assert x_of(buf, "world_ned", "sparus2/base_link", stamp[200]) == pytest.approx(200)


def test_tf_clock_needs_an_integer_record_time_for_dynamic_messages(tf_case):
    clock, rows, record_t, _, _ = tf_case
    buf = TfBuffer(clock=clock)
    buf.add_tf_message(rows[-1][2], static=True)          # static: not needed
    with pytest.raises(ValueError, match="record_t_ns"):
        buf.add_tf_message(rows[0][2])
    with pytest.raises(TypeError):
        buf.add_tf_message(rows[0][2], record_t_ns=float(record_t[0]))


def test_tf_suffixes_must_not_be_a_bare_string():
    assert TF_CHILD_SUFFIXES == ("base_link",)
    with pytest.raises(TypeError, match="sequence of suffixes"):
        TfBuffer(clock_child_suffixes="base_link")


# --------------------------------------------------------------------------
# Readers on the fake bag registry

MAIN = "/m/sparus2_2026-07-02-10-52-41_0.bag"
CAMERA = "/m/sparus2_camera_2026-07-02-10-52-41_0.bag"
STEREO = "/m/sparus2_stereo_camera_2026-07-02-10-52-41_0.bag"


def register_mission(fake_bags, record_t, stamp, paths=(MAIN, CAMERA, STEREO)):
    nav = nav_records(record_t, stamp)
    main_path, *sensor_paths = paths
    fake_bags(main_path, {
        NAV_TOPIC: nav,
        ROSOUT_TOPIC: [(int(record_t[0]), log_msg("start", int(record_t[0])))],
        GPS_NED_TOPIC: [(int(record_t[5]), gps_ned(int(record_t[5]) - MS, 1.0, 2.0))],
    })
    for path in sensor_paths:
        # The sensor recorders record the same messages 0.1 ms later.
        fake_bags(path, {NAV_TOPIC: [(r + 100_000, m) for r, m in nav]})
    return list(paths)


def test_load_mission_nav_corrects_the_navigation_only(fake_bags):
    record_t, stamp, truth = synth(np.full(200, 50.0))
    clock = clock_of(record_t, stamp)
    register_mission(fake_bags, record_t, stamp)

    nav0, gps_t0, gps_ne0 = load_mission_nav([MAIN])
    nav1, gps_t1, gps_ne1 = load_mission_nav([MAIN], clock=clock)

    assert np.array_equal(nav0.t_ns, stamp)
    assert np.array_equal(nav1.t_ns, truth)
    assert nav0.ins_clock is None and nav1.ins_clock["applied"] is True
    assert np.array_equal(gps_t0, gps_t1) and np.array_equal(gps_ne0, gps_ne1)


def test_load_mission_nav_over_several_recorders_with_a_clock(fake_bags):
    record_t, stamp, truth = synth(three_segments(400, 800, n=1000),
                                   jitter_ms=0.45)
    clock = clock_of(record_t, stamp)
    bags = register_mission(fake_bags, record_t, stamp)

    nav, _, _ = load_mission_nav(bags, clock=clock)
    # Each message recorded three times collapses onto one corrected stamp.
    assert len(nav) == len(stamp)
    assert np.abs((nav.t_ns - truth) / MS).max() <= 0.5


def test_load_mission_nav_refuses_another_missions_clock(fake_bags):
    record_t, stamp, _ = synth(np.full(200, 50.0))
    register_mission(fake_bags, record_t, stamp)
    other = clock_of(record_t + 3600 * S, stamp + 3600 * S)
    with pytest.raises(ValueError, match="another recording"):
        load_mission_nav([MAIN], clock=other)


def test_main_bags_picks_the_one_recording(fake_bags):
    record_t, stamp, _ = synth(np.full(50, 50.0))
    bags = register_mission(fake_bags, record_t, stamp)
    assert [p.name for p in main_bags(bags)] == [MAIN.split("/")[-1]]


def test_main_bags_returns_every_part_of_a_split_recording(fake_bags):
    record_t, stamp, _ = synth(np.full(50, 50.0))
    bags = register_mission(fake_bags, record_t, stamp)
    part1 = "/m/sparus2_2026-07-02-11-14-52_1.bag"
    fake_bags(part1, {NAV_TOPIC: [], ROSOUT_TOPIC: []})
    assert [p.name for p in main_bags(bags + [part1])] == [
        MAIN.split("/")[-1], part1.split("/")[-1]]


def test_main_bags_refuses_none_and_two_recordings(fake_bags):
    record_t, stamp, _ = synth(np.full(50, 50.0))
    bags = register_mission(fake_bags, record_t, stamp)
    with pytest.raises(ValueError, match="No main bag"):
        main_bags([CAMERA, STEREO])
    other = "/m/sparus2_2026-07-02-11-52-22_0.bag"
    fake_bags(other, {NAV_TOPIC: [], ROSOUT_TOPIC: []})
    with pytest.raises(ValueError, match="found 2"):
        main_bags(bags + [other])


def test_from_mission_reads_the_main_bag_only(fake_bags):
    record_t, stamp, _ = synth(np.full(200, 50.0))
    bags = register_mission(fake_bags, record_t, stamp)
    clock = InsClock.from_mission(bags)
    assert len(clock) == 200
    assert clock.source == [MAIN.split("/")[-1]]


def test_from_mission_selects_and_reads_with_the_same_topic(fake_bags):
    record_t, stamp, _ = synth(np.full(200, 50.0))
    nav = nav_records(record_t, stamp)
    fake_bags(MAIN, {"/x/navigation": nav,
                     ROSOUT_TOPIC: [(int(record_t[0]), log_msg("s", int(record_t[0])))]})
    clock = InsClock.from_mission([MAIN], nav_topic_suffix="x/navigation")
    assert len(clock) == 200


# --------------------------------------------------------------------------
# Provenance and its reader


def test_provenance_round_trips_through_json(stepped):
    clock, _, _ = stepped
    prov = clock.provenance()
    assert json.loads(json.dumps(prov)) == prov
    assert prov["applied"] is True
    assert prov["transport_ms"] == 3.8
    assert prov["window_s"] == 5.0
    assert prov["window_messages"] == 101
    assert prov["messages"] == len(clock)
    assert prov["bagio_version"] == bagio.__version__
    assert prov["source"] is None
    assert prov["repeated_stamps"] == 0
    assert prov["gaps"] == []
    assert set(prov["end_departure_ms"]) == {"start", "end"}


def test_provenance_copies_its_source(stepped):
    clock, _, _ = stepped
    named = clock_of(clock.record_t_ns, clock.stamp_ns, source=["a.bag"])
    prov = named.provenance()
    assert prov["source"] == ["a.bag"]
    prov["source"].append("b.bag")
    assert named.source == ["a.bag"]


def test_ins_clock_applied_reads_every_state(stepped):
    clock, _, _ = stepped
    assert ins_clock_applied({}) is False
    assert ins_clock_applied({PROVENANCE_KEY: dict(NOT_APPLIED)}) is False
    assert ins_clock_applied({PROVENANCE_KEY: NOT_APPLIED}) is False
    assert ins_clock_applied({PROVENANCE_KEY: clock.provenance()}) is True


def test_not_applied_is_read_only():
    assert dict(NOT_APPLIED) == {"applied": False}
    with pytest.raises(TypeError):
        NOT_APPLIED["applied"] = True


@pytest.mark.parametrize("block", [True, None, {}, {"applied": "yes"},
                                   {"applied": 1}, ["applied"]])
def test_ins_clock_applied_refuses_a_malformed_block(block):
    with pytest.raises(ValueError, match=PROVENANCE_KEY):
        ins_clock_applied({PROVENANCE_KEY: block})


@pytest.mark.parametrize("record", [None, [], "applied"])
def test_ins_clock_applied_needs_a_mapping(record):
    with pytest.raises(TypeError, match="mapping"):
        ins_clock_applied(record)


# --------------------------------------------------------------------------
# Package exports, version and the command line


def test_ins_clock_api_is_exported_from_package():
    assert bagio.InsClock is InsClock
    assert bagio.ins_clock_applied is ins_clock_applied
    assert bagio.main_bags is main_bags


def test_package_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        assert tomllib.load(f)["project"]["version"] == bagio.__version__


def test_cli_prints_each_missions_provenance(fake_bags, capsys):
    record_t, stamp, _ = synth(np.full(200, 50.0))
    bags = register_mission(fake_bags, record_t, stamp)

    assert main(bags) == 0
    out = json.loads(capsys.readouterr().out)
    (prov,) = out.values()
    assert prov["applied"] is True and prov["messages"] == 200


def test_cli_reads_a_mission_folder(fake_bags, capsys, tmp_path):
    record_t, stamp, _ = synth(np.full(200, 50.0))
    paths = [tmp_path / Path(p).name for p in (MAIN, CAMERA)]
    for p in paths:
        p.touch()
    register_mission(fake_bags, record_t, stamp, paths=[str(p) for p in paths])

    assert main([str(tmp_path), str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out) == [str(tmp_path), f"{tmp_path} #2"]
    assert all(v["messages"] == 200 for v in out.values())


def test_cli_reports_a_mission_without_main_bag(fake_bags, capsys):
    record_t, stamp, _ = synth(np.full(50, 50.0))
    register_mission(fake_bags, record_t, stamp)

    assert main([CAMERA]) == 1
    (entry,) = json.loads(capsys.readouterr().out).values()
    assert "No main bag" in entry["error"]
