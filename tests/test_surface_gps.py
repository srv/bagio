"""Synthetic tests of the navigation + surface-GPS readers."""

import numpy as np

import bagio
from bagio.surface_gps import gps_quality_reader, load_mission_nav

from conftest import (BASE_NS, GPS_NED_TOPIC, GPS_TOPIC, NAV_TOPIC, S,
                      FakeBagSet, gps_ned, navsatfix, navsts)


def nav_log(seconds):
    return [(BASE_NS + k * S, navsts(BASE_NS + k * S, float(k), 0.5 * k))
            for k in seconds]


def test_readers_are_exported_from_package():
    assert bagio.load_mission_nav is load_mission_nav
    assert bagio.gps_quality_reader is gps_quality_reader


# --------------------------------------------------------------------------
# load_mission_nav


def test_load_mission_nav_reads_nav_and_fixes(fake_bags):
    bags = fake_bags("main.bag", {
        NAV_TOPIC: nav_log(range(5)),
        GPS_NED_TOPIC: [(BASE_NS + S, gps_ned(BASE_NS + S, 10.0, -2.0))],
    })

    nav, gps_t, gps_ne = load_mission_nav(bags)

    assert len(nav) == 5
    assert np.array_equal(nav.north, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert gps_t.dtype == np.int64
    assert gps_t.tolist() == [BASE_NS + S]
    # N/E come from pose.position.x / .y, in that order.
    assert gps_ne.tolist() == [[10.0, -2.0]]


def test_load_mission_nav_sorts_fixes_by_header_stamp(fake_bags):
    """Record order is not stamp order; the fixes follow their stamps."""
    fixes = [(1, gps_ned(BASE_NS + 3 * S, 3.0, 30.0)),
             (2, gps_ned(BASE_NS + 1 * S, 1.0, 10.0)),
             (3, gps_ned(BASE_NS + 2 * S, 2.0, 20.0))]
    bags = fake_bags("main.bag", {NAV_TOPIC: nav_log(range(5)),
                                  GPS_NED_TOPIC: fixes})

    _, gps_t, gps_ne = load_mission_nav(bags)

    assert gps_t.tolist() == [BASE_NS + k * S for k in (1, 2, 3)]
    assert gps_ne.tolist() == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_load_mission_nav_without_fixes_yields_empty_arrays(fake_bags):
    bags = fake_bags("main.bag", {NAV_TOPIC: nav_log(range(3))})

    nav, gps_t, gps_ne = load_mission_nav(bags)

    assert len(nav) == 3
    assert gps_t.shape == (0,) and gps_t.dtype == np.int64
    assert gps_ne.shape == (0, 2)


def test_load_mission_nav_merges_several_bags(fake_bags):
    bags = (fake_bags("nav.bag", {NAV_TOPIC: nav_log(range(4))})
            + fake_bags("gps.bag", {GPS_NED_TOPIC: [
                (BASE_NS + 2 * S, gps_ned(BASE_NS + 2 * S, 5.0, 6.0))]}))

    nav, gps_t, gps_ne = load_mission_nav(bags)

    assert len(nav) == 4
    assert gps_t.tolist() == [BASE_NS + 2 * S]
    assert gps_ne.tolist() == [[5.0, 6.0]]


def test_load_mission_nav_reads_gps_ned_not_the_receiver_topic(fake_bags):
    """``navigator/gps`` sits next to ``navigator/gps_ned``; only the NED
    fixes are positions in the mission frame."""
    bags = fake_bags("main.bag", {
        NAV_TOPIC: nav_log(range(3)),
        GPS_NED_TOPIC: [(BASE_NS + S, gps_ned(BASE_NS + S, 1.0, 2.0))],
        GPS_TOPIC: [(BASE_NS + S, navsatfix(BASE_NS + S, 1, 0.5))],
    })

    _, gps_t, gps_ne = load_mission_nav(bags)

    assert gps_t.tolist() == [BASE_NS + S]
    assert gps_ne.tolist() == [[1.0, 2.0]]


# --------------------------------------------------------------------------
# gps_quality_reader


def quality_bags(fake_bags, fixes):
    return fake_bags("main.bag", {GPS_TOPIC: list(fixes)})


def at(k):
    return BASE_NS + k * S


def test_gps_quality_is_scoped_to_the_window(fake_bags):
    bags = quality_bags(fake_bags, [(at(k), navsatfix(at(k), 1, float(k)))
                                    for k in (10, 20, 30)])
    read = gps_quality_reader(bags)

    assert read(at(15), at(35)) == {"cov_xx_mean": 25.0, "cov_xx_min": 20.0,
                                    "cov_xx_max": 30.0, "n": 2, "status": [1]}
    # Both window boundaries are inclusive.
    assert read(at(10), at(10))["n"] == 1


def test_gps_quality_is_none_on_an_empty_window(fake_bags):
    bags = quality_bags(fake_bags, [(at(10), navsatfix(at(10), 1, 0.5))])

    assert gps_quality_reader(bags)(at(11), at(20)) is None


def test_gps_quality_skips_no_fix_and_non_finite_covariance(fake_bags):
    bags = quality_bags(fake_bags, [
        (at(1), navsatfix(at(1), -1, 0.1)),           # NO_FIX
        (at(2), navsatfix(at(2), 1, float("nan"))),   # non-finite
        (at(3), navsatfix(at(3), 1, float("inf"))),   # non-finite
        (at(4), navsatfix(at(4), 1, 0.4)),
    ])

    assert gps_quality_reader(bags)(at(0), at(10)) == {
        "cov_xx_mean": 0.4, "cov_xx_min": 0.4, "cov_xx_max": 0.4,
        "n": 1, "status": [1]}


def test_gps_quality_skips_malformed_messages(fake_bags):
    broken = navsatfix(at(1), 1, 0.2)
    broken.position_covariance = []
    bags = quality_bags(fake_bags, [(at(1), broken),
                                    (at(2), navsatfix(at(2), 1, 0.3))])

    assert gps_quality_reader(bags)(at(0), at(10))["n"] == 1


def test_gps_quality_rounds_and_reports_sorted_statuses(fake_bags):
    bags = quality_bags(fake_bags, [
        (at(1), navsatfix(at(1), 2, 0.12345678)),
        (at(2), navsatfix(at(2), 1, 0.12345678)),
        (at(3), navsatfix(at(3), 1, 0.12345678)),
    ])

    got = gps_quality_reader(bags)(at(0), at(10))

    assert got["cov_xx_mean"] == 0.123457
    assert got["status"] == [1, 2]


def test_gps_quality_falls_back_to_bag_time_on_an_unset_stamp(fake_bags):
    bags = quality_bags(fake_bags,
                        [(at(5), navsatfix(at(5), 1, 0.5, stamped=False))])

    assert gps_quality_reader(bags)(at(5), at(5))["n"] == 1


def test_gps_quality_reads_the_receiver_topic_not_gps_ned(fake_bags):
    bags = fake_bags("main.bag", {
        GPS_NED_TOPIC: [(at(1), gps_ned(at(1), 1.0, 2.0))],
        GPS_TOPIC: [(at(2), navsatfix(at(2), 1, 0.5))],
    })

    assert gps_quality_reader(bags)(at(0), at(10))["n"] == 1


def test_gps_quality_reads_only_the_bags_carrying_the_topic(fake_bags):
    bags = (fake_bags("camera.bag", {NAV_TOPIC: nav_log(range(2))})
            + fake_bags("main.bag", {GPS_TOPIC: [
                (at(1), navsatfix(at(1), 1, 0.5))]}))

    assert gps_quality_reader(bags)(at(0), at(10))["n"] == 1
    assert "camera.bag" not in FakeBagSet.read_paths


def test_gps_quality_is_none_without_the_topic(fake_bags):
    bags = fake_bags("camera.bag", {NAV_TOPIC: nav_log(range(2))})

    assert gps_quality_reader(bags)(at(0), at(10)) is None
