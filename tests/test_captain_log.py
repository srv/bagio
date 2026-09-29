"""Synthetic tests of the captain-log readers (hand-built ``/rosout``)."""

import pytest

import bagio
from bagio.captain_log import CaptainLogError, park_intervals, trigger_calls

from conftest import (BASE_NS, NAV_TOPIC, QUIET, ROSOUT_TOPIC, S, FakeBagSet,
                      log_msg, park_log)

START_PARK = (BASE_NS + 20 * S, BASE_NS + 80 * S, 60.0)
END_PARK = (BASE_NS + 300 * S, BASE_NS + 365 * S, 60.0)

SSS = "/sparus2/marinesonic_scoutmkii_sidescan"
TRIGGER_TEXT = "Calling trigger service with id "


def trigger(service, t_ns, **kwargs):
    return (t_ns, log_msg(f"{TRIGGER_TEXT}{service}", t_ns, **kwargs))


def test_readers_are_exported_from_package():
    assert bagio.park_intervals is park_intervals
    assert bagio.trigger_calls is trigger_calls
    assert bagio.CaptainLogError is CaptainLogError
    assert issubclass(CaptainLogError, RuntimeError)


# --------------------------------------------------------------------------
# park_intervals


def test_park_intervals_reads_every_complete_pair(fake_bags):
    rosout = (park_log(*START_PARK[:2], declared=60.0)
              + park_log(*END_PARK[:2], declared=120.0))
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: rosout})

    parks = park_intervals(bags, **QUIET)

    assert parks == [(START_PARK[0], START_PARK[1], 60.0),
                     (END_PARK[0], END_PARK[1], 120.0)]


def test_park_intervals_drops_a_start_without_its_finalized(fake_bags):
    """A park that never completed must not swallow the next park's end."""
    rosout = park_log(*START_PARK[:2])[:2] + park_log(*END_PARK[:2])
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: rosout})

    assert [p[:2] for p in park_intervals(bags, **QUIET)] == [END_PARK[:2]]


def test_park_intervals_drops_a_pair_that_ends_before_it_starts(fake_bags):
    rosout = [(START_PARK[1], log_msg("Executing mission park wait for 60 "
                                      "seconds", START_PARK[1])),
              (START_PARK[0], log_msg("Mission park wait finalized",
                                      START_PARK[0]))]
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: rosout})

    assert park_intervals(bags, **QUIET) == []


def test_park_intervals_warns_with_the_package_prefix(fake_bags):
    """Both discarded-pair cases are reported, never silent."""
    superseded = park_log(*START_PARK[:2])[:1]            # start, no finalize
    backwards = [(END_PARK[1], log_msg("Executing mission park wait for 60 "
                                       "seconds", END_PARK[1])),
                 (END_PARK[1] + 1, log_msg("Mission park wait finalized",
                                           END_PARK[0]))]
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: superseded + backwards})
    lines = []

    assert park_intervals(bags, log=lines.append) == []
    assert len(lines) == 2
    assert all(line.startswith("[bagio] warning:") for line in lines)


def test_park_intervals_refuses_intervals_out_of_order(fake_bags):
    """Messages arrive in record order but the intervals carry header stamps.

    When the two clocks disagree the pairing is not trustworthy, and silently
    selecting from it would anchor a placement on the wrong window.
    """
    start = "Executing mission park wait for 60 seconds"
    end = "Mission park wait finalized"
    rosout = [(1, log_msg(start, BASE_NS + 300 * S)),
              (2, log_msg(end, BASE_NS + 400 * S)),
              (3, log_msg(start, BASE_NS + 100 * S)),
              (4, log_msg(end, BASE_NS + 200 * S))]
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: rosout})

    with pytest.raises(CaptainLogError, match="overlap or are out of order"):
        park_intervals(bags, **QUIET)


def test_park_intervals_ignores_other_nodes(fake_bags):
    """Only the captain declares a park; a lookalike elsewhere must not count."""
    impostor = [(START_PARK[0],
                 log_msg("Executing mission park wait for 60 seconds",
                         START_PARK[0], name="/sparus2/pilot"))]
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: impostor})

    assert park_intervals(bags, **QUIET) == []


def test_park_intervals_falls_back_to_bag_time_on_an_unset_stamp(fake_bags):
    """A node that never fills header.stamp publishes zeros, not a 1970 park."""
    bags = fake_bags("nav.bag",
                     {ROSOUT_TOPIC: park_log(*START_PARK[:2], stamped=False)})

    assert park_intervals(bags, **QUIET)[0][:2] == START_PARK[:2]


def test_park_intervals_records_no_duration_when_none_is_declared(fake_bags):
    rosout = [(START_PARK[0], log_msg("Executing mission park wait",
                                      START_PARK[0])),
              (START_PARK[1], log_msg("Mission park wait finalized",
                                      START_PARK[1]))]
    bags = fake_bags("nav.bag", {ROSOUT_TOPIC: rosout})

    assert park_intervals(bags, **QUIET) == [(START_PARK[0], START_PARK[1],
                                              None)]


def test_park_intervals_pairs_markers_across_bag_parts(fake_bags):
    """A recording split into parts may split a park between two of them."""
    start, _request, end = park_log(*END_PARK[:2])
    bags = (fake_bags("mission_0.bag", {ROSOUT_TOPIC: [start]})
            + fake_bags("mission_1.bag", {ROSOUT_TOPIC: [end]}))

    assert park_intervals(bags, **QUIET) == [END_PARK]


def test_park_intervals_refuses_without_the_log_topic(fake_bags):
    bags = fake_bags("nav.bag", {NAV_TOPIC: []})

    with pytest.raises(CaptainLogError,
                       match="the park intervals are recorded there"):
        park_intervals(bags, **QUIET)


def test_park_intervals_reads_only_the_bags_carrying_the_log(fake_bags):
    """A mission passes all its bags; only the one with /rosout is read."""
    bags = (fake_bags("camera.bag", {NAV_TOPIC: []})
            + fake_bags("main.bag", {ROSOUT_TOPIC: park_log(*END_PARK[:2])}))

    assert [p[:2] for p in park_intervals(bags, **QUIET)] == [END_PARK[:2]]
    assert "camera.bag" not in FakeBagSet.read_paths


# --------------------------------------------------------------------------
# trigger_calls


def test_trigger_calls_returns_full_service_ids_in_stamp_order(fake_bags):
    """Record order and stamp order may differ; the stamps decide."""
    rosout = [(1, log_msg(f"{TRIGGER_TEXT}{SSS}/stop_record_xtf",
                          BASE_NS + 90 * S)),
              (2, log_msg(f"{TRIGGER_TEXT}{SSS}/enable", BASE_NS + 10 * S)),
              (3, log_msg(f"{TRIGGER_TEXT}{SSS}/start_record_xtf",
                          BASE_NS + 11 * S))]
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: rosout})

    assert trigger_calls(bags, **QUIET) == [
        (BASE_NS + 10 * S, f"{SSS}/enable"),
        (BASE_NS + 11 * S, f"{SSS}/start_record_xtf"),
        (BASE_NS + 90 * S, f"{SSS}/stop_record_xtf"),
    ]


def test_trigger_calls_keeps_record_order_for_equal_stamps(fake_bags):
    """Calls logged in one step share a stamp; their logged order is kept."""
    t = BASE_NS + 50 * S
    rosout = [(1, log_msg(f"{TRIGGER_TEXT}{SSS}/stop_record_xtf", t)),
              (2, log_msg(f"{TRIGGER_TEXT}{SSS}/disable", t))]
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: rosout})

    assert [s for _, s in trigger_calls(bags, **QUIET)] == [
        f"{SSS}/stop_record_xtf", f"{SSS}/disable"]


def test_trigger_calls_ignores_other_nodes_and_other_text(fake_bags):
    rosout = (park_log(*START_PARK[:2])
              + [trigger(f"{SSS}/enable", BASE_NS + 90 * S,
                         name="/sparus2/pilot"),
                 (BASE_NS + 95 * S, log_msg("Mission section finalized",
                                            BASE_NS + 95 * S)),
                 trigger(f"{SSS}/start_record_xtf", BASE_NS + 96 * S)])
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: rosout})

    assert trigger_calls(bags, **QUIET) == [(BASE_NS + 96 * S,
                                             f"{SSS}/start_record_xtf")]


@pytest.mark.parametrize("tail", ["", "   ", f"{SSS}/enable (timeout)"])
def test_trigger_calls_skips_a_call_without_a_single_id_and_warns(fake_bags,
                                                                   tail):
    rosout = [(BASE_NS, log_msg(f"{TRIGGER_TEXT}{tail}", BASE_NS)),
              trigger(f"{SSS}/enable", BASE_NS + S)]
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: rosout})
    lines = []

    assert trigger_calls(bags, log=lines.append) == [(BASE_NS + S,
                                                      f"{SSS}/enable")]
    assert len(lines) == 1 and lines[0].startswith("[bagio] warning:")


def test_trigger_calls_falls_back_to_bag_time_on_an_unset_stamp(fake_bags):
    rosout = [trigger(f"{SSS}/enable", BASE_NS + 10 * S, stamped=False)]
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: rosout})

    assert trigger_calls(bags, **QUIET) == [(BASE_NS + 10 * S,
                                             f"{SSS}/enable")]


def test_trigger_calls_is_empty_without_calls(fake_bags):
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: park_log(*END_PARK[:2])})

    assert trigger_calls(bags, **QUIET) == []


def test_trigger_calls_refuses_without_the_log_topic(fake_bags):
    bags = fake_bags("nav.bag", {NAV_TOPIC: []})

    with pytest.raises(CaptainLogError,
                       match="the trigger calls are recorded there"):
        trigger_calls(bags, **QUIET)
