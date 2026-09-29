"""Synthetic tests of the shared helpers: header stamps and bag filtering."""

import types

import pytest

import bagio
from bagio.bag_reader import bags_with_topics
from bagio.typestore import header_stamp_ns

from conftest import BASE_NS, GPS_TOPIC, NAV_TOPIC, ROSOUT_TOPIC, header


def test_helpers_are_exported_from_package():
    assert bagio.bags_with_topics is bags_with_topics
    assert bagio.header_stamp_ns is header_stamp_ns


# --------------------------------------------------------------------------
# header_stamp_ns


def test_header_stamp_ns_reads_a_set_stamp():
    msg = types.SimpleNamespace(header=header(BASE_NS + 5))
    assert header_stamp_ns(msg, fallback_ns=1) == BASE_NS + 5


def test_header_stamp_ns_falls_back_on_a_zero_stamp():
    msg = types.SimpleNamespace(header=header(0))
    assert header_stamp_ns(msg, fallback_ns=BASE_NS) == BASE_NS


def test_header_stamp_ns_falls_back_on_a_negative_stamp():
    stamp = types.SimpleNamespace(sec=-1, nanosec=0)
    msg = types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp))
    assert header_stamp_ns(msg, fallback_ns=BASE_NS) == BASE_NS


def test_header_stamp_ns_falls_back_without_a_header():
    msg = types.SimpleNamespace(data=b"")
    assert header_stamp_ns(msg, fallback_ns=BASE_NS) == BASE_NS


# --------------------------------------------------------------------------
# bags_with_topics


def test_bags_with_topics_keeps_the_bags_carrying_any_suffix(fake_bags):
    bags = (fake_bags("camera.bag", {"/stereo/image": []})
            + fake_bags("main.bag", {ROSOUT_TOPIC: [], GPS_TOPIC: []})
            + fake_bags("nav.bag", {NAV_TOPIC: []}))

    assert bags_with_topics(bags, ("/rosout", "navigator/navigation")) \
        == ["main.bag", "nav.bag"]
    assert bags_with_topics(bags, ("/missing",)) == []


def test_bags_with_topics_keeps_the_input_order(fake_bags):
    bags = (fake_bags("b.bag", {ROSOUT_TOPIC: []})
            + fake_bags("a.bag", {ROSOUT_TOPIC: []}))

    assert bags_with_topics(bags, ("/rosout",)) == ["b.bag", "a.bag"]


def test_bags_with_topics_refuses_a_bare_string(fake_bags):
    bags = fake_bags("main.bag", {ROSOUT_TOPIC: []})

    with pytest.raises(TypeError, match="not a string"):
        bags_with_topics(bags, "/rosout")
