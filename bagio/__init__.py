"""bagio — ROS1 bag access without ROS.

Shared foundation for the AI4MAP offline pipelines: reads recorded ``.bag``
files with the pure-Python ``rosbags`` library. Modules:

- :mod:`.typestore` — message-type registration from the bag's own embedded
  definitions.
- :mod:`.bag_reader` — time-ordered message iteration across multi-part bags
  and image decoding (raw and compressed).
- :mod:`.nav_series` — ``cola2_msgs/NavSts`` stream as time-indexed arrays
  with nearest-sample and interpolated access.
- :mod:`.tf_buffer` — transform tree built from ``/tf_static`` + ``/tf``,
  resolved at any timestamp by interpolation.
- :mod:`.captain_log` — the mission captain's ``/rosout`` log: surface parks
  and trigger-service calls.
- :mod:`.surface_gps` — navigation with the surface GPS fixes, and the
  receiver-reported fix quality.
- :mod:`.nav_frame` — exact conversion between the navigation frame
  (``NavSts`` north/east) and a map CRS.
"""

from .bag_reader import BagSet, bags_with_topics, decode_image
from .captain_log import CaptainLogError, park_intervals, trigger_calls
from .nav_frame import LocalAffine, NavFrame
from .nav_series import (
    PLACEMENT_LEVELS,
    PLACEMENT_METHOD_INS,
    PLACEMENT_METHOD_PARK,
    NavSeries,
    PlacementInfo,
    apply_trajectory_correction,
    load_placement,
    load_trajectory,
)
from .surface_gps import gps_quality_reader, load_mission_nav
from .tf_buffer import TfBuffer
from .typestore import (
    build_typestore,
    header_stamp_ns,
    register_connection_types,
    stamp_to_ns,
)

__version__ = "0.1.0"

__all__ = [
    "build_typestore",
    "register_connection_types",
    "stamp_to_ns",
    "header_stamp_ns",
    "BagSet",
    "bags_with_topics",
    "decode_image",
    "NavSeries",
    "load_trajectory",
    "apply_trajectory_correction",
    "PLACEMENT_LEVELS",
    "PLACEMENT_METHOD_PARK",
    "PLACEMENT_METHOD_INS",
    "PlacementInfo",
    "load_placement",
    "TfBuffer",
    "CaptainLogError",
    "park_intervals",
    "trigger_calls",
    "load_mission_nav",
    "gps_quality_reader",
    "NavFrame",
    "LocalAffine",
]
