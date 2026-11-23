"""Navigation and surface GPS fixes of a mission.

The vehicle is geolocalized only at the surface. The INS driver republishes
each accepted surface fix of the GPS receiver on ``navigator/gps_ned``
(``geometry_msgs/PoseStamped``) already converted into the mission's NED
frame — the same frame as ``navigator/navigation`` — so no geodetic
conversion is needed here. The fixes keep the receiver's ``navigator/gps``
stamps, on the vehicle PC's clock, whereas the navigation stamps are on the
INS clock unless corrected (:mod:`bagio.ins_clock`). The
recorded navigation is the INS solution: conditioned by GPS while the vehicle
is at the surface, purely inertial from the moment it submerges.

The receiver's own view of each fix (fix status, position covariance) is on
``navigator/gps`` (``sensor_msgs/NavSatFix``); :func:`gps_quality_reader`
summarizes it over a time window.
"""

import math

import numpy as np

from .bag_reader import BagSet, bags_with_topics
from .nav_series import NavSeries
from .typestore import header_stamp_ns, stamp_to_ns

__all__ = [
    "GPS_NED_SUFFIX",
    "GPS_STATUS_SUFFIX",
    "NAV_SUFFIX",
    "gps_quality_reader",
    "load_mission_nav",
]

NAV_SUFFIX = "navigator/navigation"
GPS_NED_SUFFIX = "navigator/gps_ned"
GPS_STATUS_SUFFIX = "navigator/gps"


def _round(value, digits=6):
    """Round for a JSON record; non-finite values become None (strict JSON has
    no Infinity/NaN)."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def load_mission_nav(bag_paths, nav_topic_suffix=NAV_SUFFIX,
                     gps_topic_suffix=GPS_NED_SUFFIX, clock=None):
    """One pass over the mission bags: nav series + surface-GPS fixes.

    Returns ``(nav, gps_t_ns, gps_ne)`` — a :class:`bagio.NavSeries` and
    the fixes' header stamps (int64 ns) and N/E positions (Nx2 float64),
    time-sorted. A mission without any ``gps_ned`` message yields empty
    fix arrays; one without any navigation message raises ``ValueError``.

    The fixes' stamps are the receiver's ``navigator/gps`` stamps, on the
    vehicle PC's clock. With ``clock`` (an :class:`~bagio.ins_clock.InsClock`)
    the navigation stamps are corrected to that clock too
    (:meth:`bagio.NavSeries.from_records`); without it they stay on the INS
    clock, as recorded. The fixes are never corrected.
    """
    nav_records = []
    gps = []
    with BagSet(list(bag_paths)) as bags:
        for t_ns, topic, msg in bags.messages(
                topic_suffixes=(nav_topic_suffix, gps_topic_suffix)):
            if topic.endswith(gps_topic_suffix):
                gps.append((stamp_to_ns(msg.header.stamp),
                            float(msg.pose.position.x),
                            float(msg.pose.position.y)))
            else:
                nav_records.append((t_ns, msg))
    nav = NavSeries.from_records(nav_records, clock)
    gps.sort(key=lambda g: g[0])
    gps_t = np.array([g[0] for g in gps], dtype=np.int64)
    gps_ne = (np.array([[g[1], g[2]] for g in gps], dtype=np.float64)
              if gps else np.zeros((0, 2)))
    return nav, gps_t, gps_ne


def gps_quality_reader(bag_paths):
    """A ``(t0_ns, t1_ns) -> dict|None`` probe of receiver-reported quality.

    Corroborating evidence only: the covariance does fall as the solution
    converges, but it is coarsely quantized and spans a far smaller range than
    the actual fix errors, so it is fit for a record rather than for selecting
    fixes. Returned as a callable so the caller can scope it to the window
    actually used — a whole-park average would describe fixes that were
    discarded. The topic is read once and the windows filter the cached
    samples, so several windows share a single pass over the bags.

    The dict holds the mean, min and max of the ``xx`` position covariance
    (``cov_xx_mean``/``cov_xx_min``/``cov_xx_max``), the sample count ``n``
    and the sorted set of fix statuses ``status``; ``None`` when the window
    holds no sample. NO_FIX samples (status < 0) and non-finite covariances
    are skipped.
    """
    samples = []
    for path in bags_with_topics(bag_paths, (GPS_STATUS_SUFFIX,)):
        with BagSet([path]) as bag_set:
            for t_ns, _topic, msg in bag_set.messages(
                    topic_suffixes=(GPS_STATUS_SUFFIX,)):
                stamp = header_stamp_ns(msg, t_ns)
                try:
                    fix_status = int(msg.status.status)
                    if fix_status < 0:      # NO_FIX: covariance is meaningless
                        continue
                    value = float(msg.position_covariance[0])
                except (AttributeError, IndexError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    samples.append((stamp, value, fix_status))

    def read(t0_ns, t1_ns):
        window = [s for s in samples if int(t0_ns) <= s[0] <= int(t1_ns)]
        if not window:
            return None
        cov = np.array([s[1] for s in window], dtype=np.float64)
        return {"cov_xx_mean": _round(float(cov.mean())),
                "cov_xx_min": _round(float(cov.min())),
                "cov_xx_max": _round(float(cov.max())),
                "n": len(window),
                "status": sorted({s[2] for s in window})}

    return read
