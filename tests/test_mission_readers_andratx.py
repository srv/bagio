"""Mission readers against the Andratx 2026-07-02 recordings.

These tests read (never write) the main bags of three sidescan missions of
that day. They are skipped when the dataset is not mounted, so the suite stays
runnable on any checkout; full runs happen inside the py311 container where
``/home/rosuser/dataset`` exists. Override the day folder with the
``BAGIO_TEST_ANDRATX_DAY`` environment variable.

The expectations are intervals and counts, not absolute instants: each
sidescan mission flew one survey line and ended with a single 32 s park.
"""

import os
from pathlib import Path

import pytest

from bagio import load_mission_nav, park_intervals, trigger_calls

DAY = Path(os.environ.get(
    "BAGIO_TEST_ANDRATX_DAY",
    "/home/rosuser/dataset/2026/orat/Andratx/2026_07_02",
))

pytestmark = pytest.mark.skipif(
    not DAY.is_dir(), reason=f"validation day not mounted at {DAY}")

S = 1e9
TOL_S = 0.05
QUIET = {"log": lambda *a: None}

# mission: (start_record_xtf -> stop_record_xtf [s],
#           stop_record_xtf -> park start [s], fixes inside the park)
SIDESCAN = {
    "10_39_47": (62.6, 25.0, 23),
    "10_44_27": (111.0, 24.5, 23),
    "11_34_44": (112.0, 28.0, 21),
}
PARK_DECLARED_S = 32.0
PARK_LENGTH_S = 32.5
TRIGGER_SEQUENCE = ["enable", "start_record_xtf", "stop_record_xtf",
                    "disable", "disable_logs"]


def main_bag(mission):
    """``sparus2_<stamp>_0.bag``: the bag carrying the captain log and GPS."""
    bags = sorted((DAY / mission).glob("sparus2_2*_0.bag"))
    assert len(bags) == 1, f"expected one main bag in {mission}, got {bags}"
    return [bags[0]]


@pytest.mark.parametrize("mission", sorted(SIDESCAN))
def test_sidescan_mission_park_triggers_and_fixes(mission):
    record_s, stop_to_park_s, n_fixes = SIDESCAN[mission]
    bags = main_bag(mission)

    parks = park_intervals(bags, **QUIET)
    calls = trigger_calls(bags, **QUIET)
    _nav, gps_t, _gps_ne = load_mission_nav(bags)

    assert len(parks) == 1
    t_start, t_end, declared = parks[0]
    assert declared == PARK_DECLARED_S
    assert (t_end - t_start) / S == pytest.approx(PARK_LENGTH_S, abs=TOL_S)

    assert [service.rsplit("/", 1)[-1] for _, service in calls] \
        == TRIGGER_SEQUENCE
    t_by_action = {service.rsplit("/", 1)[-1]: t for t, service in calls}
    t_p0 = t_by_action["start_record_xtf"]
    t_plast = t_by_action["stop_record_xtf"]
    assert (t_plast - t_p0) / S == pytest.approx(record_s, abs=TOL_S)
    assert (t_start - t_plast) / S == pytest.approx(stop_to_park_s,
                                                    abs=TOL_S)

    in_park = (gps_t >= t_start) & (gps_t <= t_end)
    assert int(in_park.sum()) == n_fixes

