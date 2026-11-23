"""INS-clock correction against recorded missions.

These tests read (never write) mission bags and are skipped when the dataset
is not mounted, so the suite stays runnable on any checkout; full runs happen
inside the py311 container where ``/home/rosuser/dataset`` exists. Override
the dataset root with the ``BAGIO_TEST_DATASET`` environment variable.

Reference values come from an independent measurement on the same bags: the
per-minute medians of (record time - header stamp) of the main bag's
navigation and their linear fit, in minutes from the first navigation record
(of the main bag for the Andratx missions; of all the recorders read for the
Pollensa ones, whose recordings start in the same second, so the two origins
differ by under a second, i.e. under ~0.012 ms of lag at ~0.72 ms/min, far
inside the 0.3 ms tolerance). Without the PPS pulse (Andratx 2026-07-02
``12_52_41``, ``13_52_22``; Pollensa 2026-06-18 ``11_56_49``, ``12_25_02``)
the lag drifts by ~0.72 ms/min; with it (``10_44_27``, ``15_46_16``) it is
flat at the transport. The 2026-10-08 mission dives once, with the INS clock
wrong at the surface and right while submerged.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pytest

from bagio import BagSet, InsClock, main_bags, stamp_to_ns

DATASET = Path(os.environ.get("BAGIO_TEST_DATASET", "/home/rosuser/dataset"))
ORAT = DATASET / "2026" / "orat"
OCT08 = DATASET / "orat_snapshots" / "2026_10_08" / "bags"
OCT08_MISSION = "2026-10-08-10-51-22"
NAV = ("navigator/navigation",)
MS = 1e6

pytestmark = pytest.mark.skipif(
    not DATASET.is_dir(), reason=f"dataset not mounted at {DATASET}")


def mission_bags(rel):
    folder = ORAT / rel
    if not folder.is_dir():
        pytest.skip(f"mission not present: {folder}")
    return sorted(folder.glob("*.bag"))


@functools.lru_cache(maxsize=None)
def clock_of(rel):
    return InsClock.from_mission(mission_bags(rel))


def per_minute_fit(clock):
    """``(slope_ms_per_min, max_dev_ms, line)`` of the per-minute medians of
    the filtered lag (correction + transport)."""
    t_min = (clock.record_t_ns - clock.record_t_ns[0]) / 6e10
    lag_ms = (clock.corrections_ns + clock.transport_ns) / MS
    minute = np.floor(t_min).astype(int)
    rows = np.array([(np.median(t_min[minute == m]),
                      np.median(lag_ms[minute == m]))
                     for m in np.unique(minute)])
    slope, intercept = np.polyfit(rows[:, 0], rows[:, 1], 1)
    dev = np.abs(rows[:, 1] - (intercept + slope * rows[:, 0])).max()
    return slope, dev, lambda t: intercept + slope * t


def nav_pairs(bags):
    """``(record_t, stamp)`` int64 arrays of the navigation in ``bags``."""
    rec, st = [], []
    with BagSet(bags) as bs:
        for t, _, m in bs.messages(topic_suffixes=NAV):
            rec.append(t)
            st.append(stamp_to_ns(m.header.stamp))
    return np.array(rec, dtype=np.int64), np.array(st, dtype=np.int64)


# --------------------------------------------------------------------------
# Missions without the pulse: the drifting lag

# mission: (slope ms/min, [(minute, lag ms), ...] on the reference line)
DRIFT = {
    "Andratx/2026_07_02/12_52_41": (0.71523, [(1.80142, 16.0678),
                                              (53.28956, 52.8935)]),
    "Andratx/2026_07_02/13_52_22": (0.71587, [(1.41728, 57.5411),
                                              (34.40935, 81.1592)]),
    "Pollensa/2026_06_18/11_56_49": (0.72272, [(0.0, 122.0719)]),
    "Pollensa/2026_06_18/12_25_02": (0.73142, [(0.0, 141.9556)]),
}


@pytest.mark.parametrize("rel", sorted(DRIFT))
def test_drifting_lag_matches_the_reference_measurement(rel):
    clock = clock_of(rel)
    slope_ref, points = DRIFT[rel]
    slope, dev, line = per_minute_fit(clock)

    assert slope == pytest.approx(slope_ref, abs=0.005)
    for minute, lag_ms in points:
        assert line(minute) == pytest.approx(lag_ms, abs=0.3)
    assert clock.provenance()["steps"] == []
    if rel.endswith("12_52_41"):
        assert dev <= 0.8


# --------------------------------------------------------------------------
# Missions with the pulse


@pytest.mark.parametrize("rel", ["Andratx/2026_07_02/10_44_27",
                                 "Pollensa/2026_06_18/15_46_16"])
def test_pulse_missions_need_no_correction(rel):
    clock = clock_of(rel)
    corr_ms = clock.corrections_ns / MS
    assert abs(np.median(corr_ms)) <= 0.5
    t_min = (clock.record_t_ns - clock.record_t_ns[0]) / 6e10
    minute = np.floor(t_min).astype(int)
    medians = [np.median(corr_ms[minute == m]) for m in np.unique(minute)]
    assert max(medians) - min(medians) <= 0.5


# --------------------------------------------------------------------------
# The other recorders' navigation


@pytest.mark.parametrize("rel", ["Andratx/2026_07_02/12_52_41",
                                 "Andratx/2026_07_02/13_52_22"])
@pytest.mark.parametrize("recorder", ["sparus2_stereo_camera",
                                      "sparus2_multibeam", "sparus2_sidescan"])
def test_sensor_bag_navigation_takes_the_main_bag_correction(rel, recorder):
    clock = clock_of(rel)
    bags = [b for b in mission_bags(rel)
            if b.name.startswith(recorder + "_2")]
    assert bags, f"no {recorder} bag in {rel}"
    rec, st = nav_pairs(bags)

    known = np.isin(st, clock.stamp_ns)
    assert known.mean() >= 0.999
    by_stamp = np.argsort(clock.stamp_ns)
    idx = np.searchsorted(clock.stamp_ns, st[known], sorter=by_stamp)
    main_rec = clock.record_t_ns[by_stamp][idx]
    offset_ms = (rec[known] - main_rec) / MS
    assert abs(np.median(offset_ms)) <= 0.1
    # A stamp the main bag carried twice would be resolved by record time
    # between carriers at least 50 ms apart; copies stay far closer.
    assert np.abs(offset_ms).max() <= 10.0
    # Every message, the unmatched ones beyond the main bag's span included,
    # is corrected within the hold, and in record order the corrected
    # stamps keep increasing.
    corrected = clock.correct(st, rec)[np.argsort(rec, kind="stable")]
    assert np.all(np.diff(corrected) > 0)


# --------------------------------------------------------------------------
# Clock steps at sea


@pytest.fixture(scope="module")
def oct08():
    bags = sorted(OCT08.glob(f"*{OCT08_MISSION}_0.bag")) if OCT08.is_dir() else []
    if not bags:
        pytest.skip(f"2026-10-08 mission {OCT08_MISSION} not present in {OCT08}")
    main = main_bags(bags)
    clock = InsClock.from_bags(main)
    rec, depth = [], []
    with BagSet(main) as bs:
        for t, _, m in bs.messages(topic_suffixes=NAV):
            rec.append(t)
            depth.append(m.position.depth)
    order = np.argsort(rec, kind="stable")
    return clock, np.array(rec)[order], np.array(depth)[order]


def test_the_submergence_step_lands_at_one_message(oct08):
    clock, rec, depth = oct08
    steps = [s for s in clock.provenance()["steps"] if abs(s["size_ms"]) > 100]
    assert [s["size_ms"] < 0 for s in steps] == [True, False]   # dive, surfacing
    dive = steps[0]
    k = int(np.searchsorted(clock.record_t_ns, dive["record_t_ns"]))
    raw_jump = (clock.lag_ns[k] - clock.lag_ns[k - 1]) / MS
    assert dive["size_ms"] == pytest.approx(raw_jump, abs=2.0)
    leaves = rec[np.argmax(depth > 0.5)]
    assert 2.0 <= (dive["record_t_ns"] - leaves) / 1e9 <= 11.0


def test_corrected_stamps_increase_where_recorded_ones_step_back(oct08):
    clock, _, _ = oct08
    assert np.sum(np.diff(clock.stamp_ns) <= 0) >= 1
    corrected = clock.correct(clock.stamp_ns, clock.record_t_ns)
    assert np.all(np.diff(corrected) > 0)


# --------------------------------------------------------------------------
# Per message


def test_correction_follows_the_lag_per_message():
    clock = clock_of("Andratx/2026_07_02/12_52_41")
    corrected = clock.correct(clock.stamp_ns, clock.record_t_ns)
    dev_ms = np.abs(corrected - (clock.record_t_ns - clock.transport_ns)) / MS
    assert np.percentile(dev_ms, 95) <= 0.8
    assert dev_ms.max() <= 5.0
