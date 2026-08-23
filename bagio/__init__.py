"""bagio — ROS1 bag access without ROS.

Shared foundation for the AI4MAP offline pipelines (mosaicing, sonar
processing): reads recorded ``.bag`` files with the pure-Python ``rosbags``
library and replaces the live ROS machinery the original pipelines relied
on:

- :mod:`.typestore` — message-type registration from the bag's own embedded
  definitions.
- :mod:`.bag_reader` — time-ordered message iteration across multi-part bags
  and image decoding (raw and compressed).
- :mod:`.nav_series` — ``cola2_msgs/NavSts`` stream as time-indexed arrays
  with nearest-sample and interpolated access.
- :mod:`.tf_buffer` — transform tree built from ``/tf_static`` + ``/tf``,
  resolved at any timestamp by interpolation.
"""

from .bag_reader import BagSet, decode_image
from .nav_series import (
    NavSeries,
    PlacementInfo,
    apply_trajectory_correction,
    load_placement,
    load_trajectory,
)
from .tf_buffer import TfBuffer
from .typestore import build_typestore, register_connection_types, stamp_to_ns

__version__ = "0.1.0"

__all__ = [
    "build_typestore",
    "register_connection_types",
    "stamp_to_ns",
    "BagSet",
    "decode_image",
    "NavSeries",
    "load_trajectory",
    "apply_trajectory_correction",
    "PlacementInfo",
    "load_placement",
    "TfBuffer",
]
