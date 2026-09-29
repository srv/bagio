"""Shared test helpers: a fake bag registry and message builders.

``FakeBagSet`` stands in for :class:`bagio.BagSet` over an in-memory registry
``{path: {topic: [(t_ns, msg), ...]}}``, so the mission readers can be tested
without recorded bags. Messages are ``types.SimpleNamespace`` objects shaped
like the deserialized ROS messages the readers access.
"""

import types

import pytest

from bagio import bag_reader, captain_log, surface_gps

BASE_NS = 1_782_993_228_000_000_000  # 2026-epoch magnitude on purpose
S = 1_000_000_000

CAPTAIN = "/sparus2/captain"
ROSOUT_TOPIC = "/rosout"
NAV_TOPIC = "/sparus2/navigator/navigation"
GPS_NED_TOPIC = "/sparus2/navigator/gps_ned"
GPS_TOPIC = "/sparus2/navigator/gps"

QUIET = {"log": lambda *a: None}


class _Conn:
    def __init__(self, topic):
        self.topic = topic


class FakeBagSet:
    """BagSet stand-in over ``{path: {topic: [(t_ns, msg), ...]}}``.

    ``read_paths`` records every bag whose messages were requested, so a test
    can check which bags a reader actually read.
    """

    registry = {}
    read_paths = []

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def connections(self, topic_suffixes=None, topics=None):
        out = []
        for path in self.paths:
            for topic in self.registry.get(path, {}):
                if topics is not None:
                    if topic in topics:
                        out.append((path, _Conn(topic)))
                elif topic_suffixes is not None:
                    if any(topic.endswith(s) for s in topic_suffixes):
                        out.append((path, _Conn(topic)))
                else:
                    out.append((path, _Conn(topic)))
        return out

    def messages(self, topic_suffixes=None, topics=None):
        FakeBagSet.read_paths.extend(self.paths)
        keep = {c.topic for _, c in self.connections(topic_suffixes, topics)}
        rows = []
        for path in self.paths:
            for topic, msgs in self.registry.get(path, {}).items():
                if topic in keep:
                    rows.extend((t, topic, m) for t, m in msgs)
        rows.sort(key=lambda row: row[0])
        return rows


@pytest.fixture
def fake_bags(monkeypatch):
    """Install FakeBagSet and return a helper that registers one bag."""
    FakeBagSet.registry = {}
    FakeBagSet.read_paths = []
    for module in (bag_reader, captain_log, surface_gps):
        monkeypatch.setattr(module, "BagSet", FakeBagSet)

    def register(path, topics):
        FakeBagSet.registry[str(path)] = topics
        return [str(path)]

    return register


def stamp(t_ns):
    return types.SimpleNamespace(sec=t_ns // S, nanosec=t_ns % S)


def header(t_ns):
    return types.SimpleNamespace(stamp=stamp(t_ns))


def log_msg(text, t_ns, name=CAPTAIN, stamped=True):
    """A ``rosgraph_msgs/Log`` entry; ``stamped=False`` leaves the stamp zero."""
    return types.SimpleNamespace(name=name, msg=text,
                                 header=header(t_ns if stamped else 0))


def park_log(t_start_ns, t_end_ns, declared=60.0, step_no=7, stamped=True):
    return [
        (t_start_ns, log_msg(
            f"Executing mission park wait for {declared:.0f} seconds. "
            f"Step {step_no}", t_start_ns, stamped=stamped)),
        (t_start_ns, log_msg(
            "Send park request at [-88.0684, 5.00285] with depth 0. "
            f"Timeout is {declared:.0f} seconds", t_start_ns, stamped=stamped)),
        (t_end_ns, log_msg("Mission park wait finalized", t_end_ns,
                           stamped=stamped)),
    ]


def navsts(t_ns, north, east, lat=39.5, lon=2.4):
    """A ``cola2_msgs/NavSts`` with the fields NavSeries reads."""
    ns = types.SimpleNamespace
    return ns(
        header=header(t_ns),
        origin=ns(latitude=lat, longitude=lon),
        position=ns(north=north, east=east, depth=0.0),
        altitude=5.0,
        orientation=ns(roll=0.0, pitch=0.0, yaw=0.0),
        orientation_rate=ns(yaw=0.0),
        body_velocity=ns(x=0.0),
        global_position=ns(latitude=lat, longitude=lon),
    )


def gps_ned(t_ns, north, east):
    """A ``geometry_msgs/PoseStamped`` surface fix in the mission NED frame."""
    ns = types.SimpleNamespace
    return ns(header=header(t_ns),
              pose=ns(position=ns(x=north, y=east, z=0.0)))


def navsatfix(t_ns, status, cov_xx, stamped=True):
    """A ``sensor_msgs/NavSatFix`` with a fix status and an xx covariance."""
    ns = types.SimpleNamespace
    return ns(header=header(t_ns if stamped else 0),
              status=ns(status=status),
              position_covariance=[cov_xx] + [0.0] * 8)
