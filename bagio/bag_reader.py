"""Time-ordered message access over one or more ROS1 bag files.

A mission's data may span several bag parts (``*_0.bag``, ``*_1.bag``, …);
:class:`BagSet` opens them together and yields messages merged in timestamp
order, decoded on the fly. Image messages (raw ``sensor_msgs/Image`` and
``sensor_msgs/CompressedImage``) are converted to BGR numpy arrays by
:func:`decode_image`.
"""

import heapq
from pathlib import Path

import cv2
import numpy as np

from .typestore import build_typestore

# Raw-encoding conversions to BGR. cv_bridge's "bgr8" target conversion is
# reproduced for the encodings that appear in this project's cameras; unknown
# encodings raise so silent color corruption cannot happen.
_ENCODING_TO_BGR = {
    "bgr8": None,
    "rgb8": cv2.COLOR_RGB2BGR,
    "mono8": cv2.COLOR_GRAY2BGR,
    "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
    "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
    "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
    "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
}


def decode_image(msg, msgtype):
    """Return the message's image as a BGR uint8 array.

    Handles ``sensor_msgs/CompressedImage`` (JPEG/PNG payload) and raw
    ``sensor_msgs/Image`` with the encodings in ``_ENCODING_TO_BGR``.
    """
    if "CompressedImage" in msgtype:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed on CompressedImage payload "
                             f"(format={msg.format!r})")
        return img

    enc = msg.encoding
    if enc not in _ENCODING_TO_BGR:
        raise ValueError(f"Unsupported raw image encoding {enc!r}")
    channels = 3 if enc in ("bgr8", "rgb8") else 1
    arr = np.frombuffer(msg.data, np.uint8)
    step = int(msg.step)
    height, width = int(msg.height), int(msg.width)
    arr = arr.reshape(height, step)[:, : width * channels]
    arr = arr.reshape(height, width, channels) if channels == 3 else arr.reshape(height, width)
    code = _ENCODING_TO_BGR[enc]
    return arr.copy() if code is None else cv2.cvtColor(arr, code)


class BagSet:
    """Context manager over one or more ROS1 bags with a shared typestore.

    Usage::

        with BagSet(sorted(mission.glob('sparus2_stereo_camera_*.bag'))) as bags:
            for t_ns, topic, msg in bags.messages(topic_suffixes=('port/camera_info',)):
                ...
    """

    def __init__(self, bag_paths):
        paths = [Path(p) for p in bag_paths]
        if not paths:
            raise ValueError("BagSet needs at least one bag path")
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Bag file(s) not found: {missing}")
        self.paths = sorted(paths)
        self.readers = []
        self.typestore = None

    def __enter__(self):
        from rosbags.rosbag1 import Reader

        try:
            for p in self.paths:
                reader = Reader(p)
                reader.open()
                self.readers.append(reader)
            self.typestore = build_typestore(self.readers)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for reader in self.readers:
            try:
                reader.close()
            except Exception:
                pass
        self.readers = []
        return False

    def connections(self, topic_suffixes=None, topics=None):
        """Connections across all bags, filtered by exact topic or suffix."""
        out = []
        for reader in self.readers:
            for c in reader.connections:
                if topics is not None and c.topic in topics:
                    out.append((reader, c))
                elif topic_suffixes is not None and any(
                    c.topic.endswith(s) for s in topic_suffixes
                ):
                    out.append((reader, c))
                elif topics is None and topic_suffixes is None:
                    out.append((reader, c))
        return out

    def message_count(self, topic_suffixes=None, topics=None):
        return sum(c.msgcount for _, c in self.connections(topic_suffixes, topics))

    def messages(self, topic_suffixes=None, topics=None):
        """Yield ``(t_ns, topic, msg)`` merged across bags in time order.

        ``t_ns`` is the bag record timestamp in nanoseconds. Messages are
        deserialized with the shared typestore.
        """
        per_reader = {}
        for reader, conn in self.connections(topic_suffixes, topics):
            per_reader.setdefault(reader, []).append(conn)

        def stream(reader, conns):
            for conn, t_ns, raw in reader.messages(connections=conns):
                yield t_ns, conn.topic, conn.msgtype, raw

        merged = heapq.merge(
            *(stream(r, cs) for r, cs in per_reader.items()), key=lambda x: x[0]
        )
        for t_ns, topic, msgtype, raw in merged:
            yield t_ns, topic, self.typestore.deserialize_ros1(raw, msgtype)


def bags_with_topics(bag_paths, topic_suffixes):
    """Subset of ``bag_paths`` that actually carries any of the topics.

    Reading a mission's camera bags for navigation would cost minutes for
    nothing, so the connection index (cheap) decides which bags are opened for
    messages. Presence is checked by topic, never by filename convention. The
    input order is kept. ``topic_suffixes`` is a sequence of suffixes; a bare
    string raises ``TypeError``, as it would be matched character by character.
    """
    if isinstance(topic_suffixes, str):
        raise TypeError(
            "topic_suffixes must be a sequence of suffixes, not a string: "
            f"pass ({topic_suffixes!r},)")
    keep = []
    for path in bag_paths:
        with BagSet([path]) as bags:
            if bags.connections(topic_suffixes=topic_suffixes):
                keep.append(path)
    return keep
