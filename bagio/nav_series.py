"""``cola2_msgs/NavSts`` stream as time-indexed arrays.

Replaces the original pipeline's ``message_filters.ApproximateTimeSynchronizer``
pairing of images with navigation messages: the whole navigation stream is
loaded once, and consumers ask for the sample nearest to (or interpolated at)
an image timestamp. Timestamps are the messages' own ``header.stamp`` (not
the bag record time), matching what the synchronizer matched on.
"""

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

from .typestore import stamp_to_ns

# Coarse metres-per-degree used only by the trajectory origin guard (a sanity
# net against a mis-paired artifact, not a survey-grade conversion).
_M_PER_DEG = 111_320.0

# What a placement translation was measured from: the surface park before the
# dive, the surface park after the final ascent, or the INS navigation as
# recorded (no GPS, nothing estimated).
PLACEMENT_LEVELS = ("start_park", "end_park", "ins_as_recorded")
_START_PARK, _END_PARK, _INS_AS_RECORDED = PLACEMENT_LEVELS

# Placement-slot ``method`` values from which the level is derived: a
# translation estimated from a surface park's GPS fixes, and one that keeps
# the recorded INS navigation.
PLACEMENT_METHOD_PARK = "park_navavg_median"
PLACEMENT_METHOD_INS = "ins_as_recorded"
_LEVEL_BY_PARK = {"start": _START_PARK, "end": _END_PARK}

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
        # Drop repeated header stamps, keeping the first occurrence (the sort
        # is stable, so "first" is first in message order). A zero-width
        # interval is poison for interpolation: np.interp returns whichever
        # duplicate lands adjacent to the query, so a bogus resend could
        # silently shadow the real sample.
        deduped = [samples[0]]
        for s in samples[1:]:
            if s.t_ns != deduped[-1].t_ns:
                deduped.append(s)
        self.samples = deduped
        samples = deduped
        self.t_ns = np.array([s.t_ns for s in samples], dtype=np.int64)
        for name in _FIELDS:
            setattr(self, name, np.array([getattr(s, name) for s in samples]))
        # Unwrapped copy for interpolation across the ±pi seam; the per-sample
        # `yaw` stays as recorded. Caveat: a
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

    def interp_values(self, t_ns, values):
        """Linear interpolation of a precomputed per-sample array at ``t_ns``.

        ``values`` must be aligned with ``t_ns``/``samples`` (one entry per
        sample, e.g. the output of :meth:`median_filtered`). Uses the same
        precision-safe stamp offsets as :meth:`interp`; clamped at the ends.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self._t_rel.shape:
            raise ValueError(
                "interp_values: `values` must have one entry per sample "
                f"({self._t_rel.shape[0]}), got shape {values.shape}"
            )
        return float(np.interp(
            float(int(t_ns) - self._t0), self._t_rel, values))

    def median_filtered(self, field, window_s):
        """Centered sliding-median copy of ``field``, aligned with ``t_ns``.

        For each sample, the median of all samples whose header stamp lies
        within ``±window_s / 2`` of its own stamp. The window is time-based
        (robust to irregular sampling) and always contains the sample itself,
        so a non-positive ``window_s`` returns the raw values unchanged.
        Deterministic: same series + same window -> same output.

        Intended for measurement channels with impulsive artifacts — e.g.
        DVL ``altitude`` bottom-lock plane hops. Refuses ``yaw``: a plain
        median across the ±pi seam is meaningless (use the unwrapped series
        for angular smoothing if ever needed).
        """
        if field == "yaw":
            raise ValueError(
                "median_filtered: `yaw` wraps at ±pi; a plain median across "
                "the seam is meaningless"
            )
        values = np.asarray(getattr(self, field), dtype=np.float64)
        if window_s <= 0:
            return values.copy()
        half_ns = int(round(window_s * 0.5e9))
        lo = np.searchsorted(self.t_ns, self.t_ns - half_ns, side="left")
        hi = np.searchsorted(self.t_ns, self.t_ns + half_ns, side="right")
        out = np.empty_like(values)
        for i in range(len(values)):
            out[i] = np.median(values[lo[i]:hi[i]])
        return out

    def with_correction(self, path, origin_tol_m=1.0):
        """A copy of this series corrected by the trajectory artifact at ``path``.

        Convenience wrapper: loads and validates the ``trajectory_refined.json``
        at ``path`` and applies it. See :func:`apply_trajectory_correction` for
        the correction semantics and ``origin_tol_m``.
        """
        return apply_trajectory_correction(
            self, load_trajectory(path), origin_tol_m=origin_tol_m)


_REQUIRED_KEYFRAME_FIELDS = ("t_ns", "delta_n", "delta_e", "delta_yaw")


def _validate_trajectory_header(artifact, source):
    """Raise ``ValueError`` unless ``artifact``'s header marks a usable,
    relative-frame correction with a self-consistent keyframe list.

    Shared by :func:`load_trajectory` and :func:`apply_trajectory_correction`
    so both public entry points are independently safe — a caller that
    hand-builds or hand-loads the dict cannot bypass the gate. ``source``
    (a path or a placeholder) is quoted in the messages.
    """
    version = artifact.get("schema_version")
    if version != 1:
        raise ValueError(
            f"trajectory {source}: unsupported schema_version {version!r} "
            "(expected 1)")
    status = artifact.get("status")
    if status != "ok":
        raise ValueError(
            f"trajectory {source}: status is {status!r}, expected 'ok' — the "
            "refine stage did not produce a usable correction")
    frame = artifact.get("frame") or {}
    frame_type = frame.get("type")
    if frame_type != "relative":
        raise ValueError(
            f"trajectory {source}: frame.type is {frame_type!r}, expected "
            "'relative'")
    keyframes = artifact.get("keyframes")
    if not keyframes:
        raise ValueError(f"trajectory {source}: no keyframes")
    n_declared = artifact.get("n_keyframes")
    if n_declared is not None and n_declared != len(keyframes):
        raise ValueError(
            f"trajectory {source}: n_keyframes ({n_declared}) does not match "
            f"the number of keyframes ({len(keyframes)}) — looks truncated")


def load_trajectory(path):
    """Load and validate a ``trajectory_refined.json`` correction artifact.

    Parses the JSON at ``path`` and checks the header fields that gate whether
    the correction is safe to consume: ``schema_version`` must be ``1``,
    ``status`` must be ``"ok"`` (the refine stage marks a failed solve
    otherwise), ``frame.type`` must be ``"relative"`` (the per-keyframe deltas
    are world-frame NED offsets, not an absolute placement), and the keyframe
    list must be present and match ``n_keyframes``. Returns the parsed dict
    unchanged — the caller uses both the ``keyframes`` (via
    :func:`apply_trajectory_correction`) and the header (e.g.
    ``refine_params_sha256`` for a provenance record). Raises ``ValueError`` on
    any failed check.
    """
    with open(path) as f:
        artifact = json.load(f)
    _validate_trajectory_header(artifact, path)
    return artifact


def _is_finite_number(value):
    """True for a real, finite JSON number (int or float), excluding ``bool``.

    ``bool`` is an ``int`` subclass, so a stray ``True``/``False`` would
    otherwise slip through a numeric check as ``1``/``0``. A Python int too
    large to fit a C double raises ``OverflowError`` in :func:`math.isfinite`;
    that is not a usable finite value either, so treat it as a rejection.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class PlacementInfo:
    """The rigid translation recorded in a trajectory artifact's ``placement``
    slot.

    Written by ``offline_mosaicing.placement``, which solves the per-mission
    rigid translation — measured at a surface park or taken from the recorded
    INS position — that places a mission's GPS-free (floating) products onto
    the ground. ``translation_ne_m`` is ``(north, east)`` in metres — **north
    first, east second**, the order the slot stores — and the placement is
    **translation only** (no rotation). ``method`` and ``refine_params_sha256``
    carry the slot's provenance, each ``None`` when the slot does not record it
    (a well-formed ``computed`` slot always names its ``method``; the reader
    reports what the slot holds rather than second-guessing the producer).
    ``level`` says what the translation was measured from — one of
    :data:`PLACEMENT_LEVELS` — or is ``None`` when the slot's fields do not
    determine it (see :func:`load_placement`); any other value raises
    ``ValueError``.
    """

    translation_ne_m: tuple[float, float]   # (north, east) metres, as stored
    method: str | None
    refine_params_sha256: str | None
    level: str | None = None

    def __post_init__(self):
        if self.level is not None and self.level not in PLACEMENT_LEVELS:
            raise ValueError(
                f"PlacementInfo.level must be one of {PLACEMENT_LEVELS} or "
                f"None, got {self.level!r}")


def _placement_level(slot):
    """The :data:`PLACEMENT_LEVELS` entry a placement slot's fields determine.

    A park placement (``method`` :data:`PLACEMENT_METHOD_PARK`) is
    ``start_park`` or ``end_park`` by its ``park.which``; a slot with
    ``method`` :data:`PLACEMENT_METHOD_INS` is ``ins_as_recorded``. Anything
    else — no method, another method, no ``park`` block, an unknown ``which``
    — is ``None``.
    """
    method = slot.get("method")
    if method == PLACEMENT_METHOD_INS:
        return _INS_AS_RECORDED
    if method == PLACEMENT_METHOD_PARK:
        park = slot.get("park")
        if isinstance(park, dict):
            which = park.get("which")
            if isinstance(which, str):
                return _LEVEL_BY_PARK.get(which)
    return None


def load_placement(artifact):
    """Read the ``placement`` slot of a trajectory refine artifact.

    ``offline_mosaicing.placement`` writes a ``placement`` block into
    ``trajectory_refined.json`` recording the per-mission rigid translation
    that places the mission's GPS-free (floating) products onto the ground.
    ``artifact`` may be the dict returned by :func:`load_trajectory` or a path
    to the JSON file — in the path case :func:`load_trajectory` is called first,
    reusing its header validation. Returns a :class:`PlacementInfo` with the
    ``(north, east)`` translation (order as stored), the slot's provenance and
    its placement level.

    The level is derived, not stored: ``method`` ``park_navavg_median`` with
    ``park.which`` ``"start"`` / ``"end"`` is ``start_park`` / ``end_park``;
    ``method`` ``ins_as_recorded`` is ``ins_as_recorded``; anything else is
    ``None``. The level is reported, never a reason to refuse the slot.

    The reader supports **pure translation only** and reports only what the
    translation is; it does not inspect ``applied_to_products`` — whether the
    products on disk have been shifted is the applier's concern, not the
    reader's. Raises ``ValueError`` — naming the artifact and the remedy — when:

    - there is no ``placement`` slot (the artifact predates placement, or was
      not produced by the placement estimator);
    - the slot ``status`` is not ``"computed"`` (``"unplaced"``/``"anchored"``
      carry no usable translation — run the placement estimator first);
    - ``translation_ne_m`` is not a 2-sequence of finite numbers;
    - ``rotation_deg`` is present and not exactly ``0.0`` (a rotating placement
      must arrive as an explicit API change, never a silently dropped term);
    - the slot's ``refine_params_sha256`` is present alongside the artifact
      header's and disagrees with it (the slot was carried over from a
      different solve).
    """
    if isinstance(artifact, dict):
        source = "<artifact>"
    elif isinstance(artifact, (str, os.PathLike)):
        source = str(artifact)
        artifact = load_trajectory(artifact)
    else:
        raise TypeError(
            "load_placement: artifact must be a placement dict or a path to "
            f"trajectory_refined.json, got {type(artifact).__name__}")

    placement = artifact.get("placement")
    if not isinstance(placement, dict):
        raise ValueError(
            f"placement {source}: artifact has no placement slot — it predates "
            "placement or was not produced by offline_mosaicing.placement")

    status = placement.get("status")
    if status != "computed":
        raise ValueError(
            f"placement {source}: status is {status!r}, expected 'computed' — "
            "run `python -m offline_mosaicing.placement <mission>` first to "
            "compute the translation")

    translation = placement.get("translation_ne_m")
    if (not isinstance(translation, (list, tuple))
            or len(translation) != 2
            or not all(_is_finite_number(v) for v in translation)):
        raise ValueError(
            f"placement {source}: translation_ne_m must be a 2-sequence of "
            f"finite [north, east] metres, got {translation!r}")

    rotation = placement.get("rotation_deg")
    if rotation is not None and (
            not _is_finite_number(rotation) or rotation != 0.0):
        raise ValueError(
            f"placement {source}: rotation_deg must be exactly 0.0 (the reader "
            f"supports pure translation only), got {rotation!r} — a rotating "
            "placement requires an explicit API change")

    slot_sha = placement.get("refine_params_sha256")
    header_sha = artifact.get("refine_params_sha256")
    if (slot_sha is not None and header_sha is not None
            and slot_sha != header_sha):
        raise ValueError(
            f"placement {source}: slot refine_params_sha256 {slot_sha!r} does "
            f"not match the artifact header {header_sha!r} — the placement slot "
            "was carried over from a different solve")

    return PlacementInfo(
        translation_ne_m=(float(translation[0]), float(translation[1])),
        method=placement.get("method"),
        refine_params_sha256=slot_sha,
        level=_placement_level(placement),
    )


def apply_trajectory_correction(nav, artifact, origin_tol_m=1.0):
    """Return a copy of ``nav`` corrected by a trajectory refine ``artifact``.

    There is one physical AUV trajectory per mission; the mosaicing refine
    stage solves a GPS-free correction to it from the camera pose-graph and
    exports it as ``trajectory_refined.json``. Every sensor that rode that same
    trajectory (here the multibeam) must consume the *identical* correction, so
    this rebuilds ``nav`` with the per-keyframe world-frame deltas applied.

    For each sample the keyframe deltas ``delta_n``/``delta_e``/``delta_yaw``
    are interpolated at the sample's stamp: linear between keyframes, held flat
    beyond the first/last keyframe (``np.interp`` clamps), matching the
    artifact's ``extrapolation`` contract (``start``/``end`` hold, ``interp``
    linear). The interpolation runs on nanosecond offsets from ``nav``'s first
    stamp — the same precision-safe basis :meth:`interp` uses — with the
    keyframe stamps rebased onto it (both share ``NavSts``' epoch timebase).

    ``delta_n``/``delta_e`` add to ``north``/``east``; ``delta_yaw`` is *always*
    added to ``yaw`` and re-wrapped to (-pi, pi] via ``arctan2(sin, cos)``. All
    other fields (including the recorded ``lat``/``lon``, which are not on the
    projection path) are copied verbatim. The result is a fresh
    :class:`NavSeries`, so ``yaw_unwrapped`` and the interpolation offsets are
    recomputed from the corrected yaw.

    The artifact header is re-validated here (schema/status/frame/keyframes; see
    :func:`load_trajectory`), so this is safe to call directly on a hand-built
    dict. As a guard against a mis-paired file — the one-trajectory-per-mission
    rule — the artifact's ``frame.ned_origin_lat``/``lon`` must be within
    ``origin_tol_m`` metres of ``nav``'s origin; a larger gap raises
    ``ValueError`` (it is the wrong mission's correction).
    """
    _validate_trajectory_header(artifact, "<artifact>")
    frame = artifact["frame"]
    art_lat = frame.get("ned_origin_lat")
    art_lon = frame.get("ned_origin_lon")
    if art_lat is None or art_lon is None:
        raise ValueError(
            "trajectory artifact frame is missing ned_origin_lat/ned_origin_lon")
    # Coarse great-circle-free distance between the two NED origins. Correctly
    # paired, they are the same mission datum (equal to JSON rounding, ~cm); a
    # different mission's origin is metres-to-kilometres away.
    d_north = (art_lat - nav.origin_lat) * _M_PER_DEG
    d_east = (art_lon - nav.origin_lon) * _M_PER_DEG * np.cos(
        np.deg2rad(nav.origin_lat))
    origin_gap_m = float(np.hypot(d_north, d_east))
    if origin_gap_m > origin_tol_m:
        raise ValueError(
            f"trajectory origin ({art_lat}, {art_lon}) is {origin_gap_m:.2f} m "
            f"from the nav origin ({nav.origin_lat}, {nav.origin_lon}), beyond "
            f"the {origin_tol_m} m tolerance — this is the wrong mission's "
            "correction")

    keyframes = artifact["keyframes"]  # header validation ensured non-empty
    for i, k in enumerate(keyframes):
        missing = [f for f in _REQUIRED_KEYFRAME_FIELDS if f not in k]
        if missing:
            raise ValueError(
                f"trajectory keyframe {i} is missing fields {missing}")
        # t_ns must stay an exact integer: a float-encoded stamp is already
        # ~256 ns-quantized at 2026-epoch magnitude, and the int64 cast below
        # would silently truncate it (bool is an int subclass — exclude it).
        if not isinstance(k["t_ns"], int) or isinstance(k["t_ns"], bool):
            raise ValueError(
                f"trajectory keyframe {i}: t_ns must be an integer nanosecond "
                f"stamp, got {type(k['t_ns']).__name__}")
    kf_t = np.array([k["t_ns"] for k in keyframes], dtype=np.int64)
    kf_dn = np.array([k["delta_n"] for k in keyframes], dtype=np.float64)
    kf_de = np.array([k["delta_e"] for k in keyframes], dtype=np.float64)
    kf_dyaw = np.array([k["delta_yaw"] for k in keyframes], dtype=np.float64)
    # np.interp assumes increasing sample points and does not validate them, so
    # sort defensively (the artifact stores keyframes in time order) and reject
    # ties: a zero-width interval makes np.interp return whichever duplicate
    # lands adjacent to the query — the same interpolation hazard NavSeries
    # guards against for repeated nav stamps.
    order = np.argsort(kf_t, kind="stable")
    kf_t, kf_dn, kf_de, kf_dyaw = (
        kf_t[order], kf_dn[order], kf_de[order], kf_dyaw[order])
    if np.any(np.diff(kf_t) == 0):
        raise ValueError(
            "trajectory keyframes contain duplicate t_ns stamps — the "
            "correction at that instant is ambiguous")

    # Rebase the keyframe stamps onto nav's first stamp (both are NavSts epoch
    # ns): the offsets stay small integers, exact in float64, unlike absolute
    # 2026-epoch ns. Query points are nav's own precomputed offsets.
    kf_rel = (kf_t - nav._t0).astype(np.float64)
    dn = np.interp(nav._t_rel, kf_rel, kf_dn)
    de = np.interp(nav._t_rel, kf_rel, kf_de)
    dyaw = np.interp(nav._t_rel, kf_rel, kf_dyaw)

    new_north = nav.north + dn
    new_east = nav.east + de
    corrected_yaw = nav.yaw + dyaw
    new_yaw = np.arctan2(np.sin(corrected_yaw), np.cos(corrected_yaw))

    new_samples = [
        replace(s, north=float(new_north[i]), east=float(new_east[i]),
                yaw=float(new_yaw[i]))
        for i, s in enumerate(nav.samples)
    ]
    return NavSeries(new_samples, nav.origin_lat, nav.origin_lon)
