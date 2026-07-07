"""``cola2_msgs/NavSts`` stream as time-indexed arrays.

Replaces the original pipeline's ``message_filters.ApproximateTimeSynchronizer``
pairing of images with navigation messages: the whole navigation stream is
loaded once, and consumers ask for the sample nearest to (or interpolated at)
an image timestamp. Timestamps are the messages' own ``header.stamp`` (not
the bag record time), matching what the synchronizer matched on.
"""

from dataclasses import dataclass

import numpy as np

from .typestore import stamp_to_ns

_FIELDS = (
    "north", "east", "depth", "altitude",
    "roll", "pitch", "yaw",
    "yaw_rate", "vel_x", "lat", "lon",
)


@dataclass
class NavSample:
    """One navigation sample (units as recorded: meters, radians, m/s)."""

    t_ns: int
    north: float
    east: float
    depth: float
    altitude: float
    roll: float
    pitch: float
    yaw: float
    yaw_rate: float
    vel_x: float
    lat: float
    lon: float


class NavSeries:
    """Sorted arrays of the NavSts fields the pipeline consumes."""

    def __init__(self, samples, origin_lat, origin_lon):
        if not samples:
            raise ValueError("NavSeries needs at least one NavSts message")
        samples.sort(key=lambda s: s.t_ns)
        self.samples = samples
        self.t_ns = np.array([s.t_ns for s in samples], dtype=np.int64)
        for name in _FIELDS:
            setattr(self, name, np.array([getattr(s, name) for s in samples]))
        # Unwrapped copy for interpolation across the ±pi seam; the per-sample
        # `yaw` stays as recorded (what the YAML sidecar stores). Caveat: a
        # genuine >pi yaw change between consecutive samples (e.g. across a
        # nav dropout) is indistinguishable from wraparound — inherent to
        # unwrapping, acceptable at this stream's 10 Hz rate.
        self.yaw_unwrapped = np.unwrap(self.yaw)
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        # Interpolation runs on offsets from the first stamp: absolute
        # 2026-epoch nanoseconds exceed float64 integer precision (~256 ns
        # granularity), offsets are exact for any mission length.
        self._t0 = int(self.t_ns[0])
        self._t_rel = (self.t_ns - self._t0).astype(np.float64)

    @classmethod
    def from_messages(cls, messages):
        """Build from an iterable of deserialized NavSts messages."""
        samples = []
        origin = None
        for m in messages:
            if origin is None:
                origin = (m.origin.latitude, m.origin.longitude)
            samples.append(NavSample(
                t_ns=stamp_to_ns(m.header.stamp),
                north=m.position.north,
                east=m.position.east,
                depth=m.position.depth,
                altitude=m.altitude,
                roll=m.orientation.roll,
                pitch=m.orientation.pitch,
                yaw=m.orientation.yaw,
                yaw_rate=m.orientation_rate.yaw,
                vel_x=m.body_velocity.x,
                lat=m.global_position.latitude,
                lon=m.global_position.longitude,
            ))
        if origin is None:
            raise ValueError("No NavSts messages supplied")
        return cls(samples, *origin)

    def __len__(self):
        return len(self.samples)

    def nearest(self, t_ns):
        """The sample whose header stamp is closest to ``t_ns``."""
        i = int(np.searchsorted(self.t_ns, t_ns))
        if i <= 0:
            return self.samples[0]
        if i >= len(self.samples):
            return self.samples[-1]
        before, after = self.samples[i - 1], self.samples[i]
        return before if t_ns - before.t_ns <= after.t_ns - t_ns else after

    def interp(self, t_ns, field):
        """Linear interpolation of ``field`` at ``t_ns`` (clamped at ends).

        ``yaw`` interpolates on the unwrapped series and re-wraps to
        (-pi, pi] so the ±pi seam cannot produce a bogus midpoint.
        """
        values = self.yaw_unwrapped if field == "yaw" else getattr(self, field)
        out = float(np.interp(float(int(t_ns) - self._t0), self._t_rel, values))
        if field == "yaw":
            out = float(np.arctan2(np.sin(out), np.cos(out)))
        return out
