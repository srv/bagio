"""Message-type registration for decoding ROS1 bags without ROS.

Types are registered from each connection's embedded message definition
rather than from ``rosbags``' bundled ROS1 store: the bundled store lacks
some standard types (e.g. ``tf2_msgs/TFMessage``) and knows nothing about
``cola2_msgs``. Registering from the bag itself also stays correct if a
message definition ever changes between campaigns — each bag carries the
definition its messages were serialized with.
"""

from rosbags.typesys import Stores, get_types_from_msg, get_typestore


def build_typestore(readers):
    """Return a typestore able to decode every connection of ``readers``.

    Parameters
    ----------
    readers : iterable
        Open ``rosbags.rosbag1.Reader`` instances.
    """
    ts = get_typestore(Stores.ROS1_NOETIC)
    for reader in readers:
        register_connection_types(ts, reader.connections)
    return ts


def register_connection_types(ts, connections):
    """Register the embedded message definitions of ``connections`` in ``ts``.

    Re-registering a type with an identical definition is a no-op in
    ``rosbags``; a genuinely conflicting redefinition raises, which is the
    desired loud failure.
    """
    for c in connections:
        ts.register(get_types_from_msg(c.msgdef.data, c.msgtype))


def stamp_to_ns(stamp):
    """Convert a deserialized header stamp to integer nanoseconds.

    ``rosbags`` exposes ROS1 ``time`` fields with ROS2-style names
    (``sec``/``nanosec``); accept the rospy-style names too so helpers stay
    usable with dict-like fixtures in tests.
    """
    sec = getattr(stamp, "sec", None)
    if sec is None:
        sec = stamp.secs
    nanosec = getattr(stamp, "nanosec", None)
    if nanosec is None:
        nanosec = stamp.nsecs
    return int(sec) * 1_000_000_000 + int(nanosec)
