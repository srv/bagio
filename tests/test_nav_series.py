"""Synthetic NavSeries tests (no dataset needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from bagio.nav_series import (
    PLACEMENT_LEVELS,
    NavSample,
    NavSeries,
    PlacementInfo,
    apply_trajectory_correction,
    load_placement,
    load_trajectory,
)

BASE_NS = 1_782_993_228_000_000_000  # 2026-epoch magnitude on purpose

FIXTURE = Path(__file__).parent / "fixtures" / "trajectory_refined.json"


def make_series(altitudes, dt_ns=100_000_000, **overrides):
    """NavSeries with the given altitude sequence at a fixed sample rate."""
    samples = []
    for k, alt in enumerate(altitudes):
        fields = dict(
            t_ns=BASE_NS + k * dt_ns,
            north=float(k), east=0.0, depth=5.0, altitude=float(alt),
            roll=0.0, pitch=0.0, yaw=0.0,
            yaw_rate=0.0, vel_x=0.5, lat=39.5, lon=2.4,
        )
        fields.update({key: val[k] for key, val in overrides.items()})
        samples.append(NavSample(**fields))
    return NavSeries(samples, origin_lat=39.5, origin_lon=2.4)


def test_median_filtered_removes_isolated_hop():
    # A single-sample bottom-lock hop must vanish while
    # the surrounding plateau is untouched.
    alt = [3.0] * 10 + [3.4] + [3.0] * 10
    nav = make_series(alt)
    filt = nav.median_filtered("altitude", window_s=0.5)
    assert np.allclose(filt, 3.0)


def test_median_filtered_preserves_sustained_change():
    # A genuine terrain step (longer than the window) must survive.
    alt = [3.0] * 10 + [3.5] * 10
    nav = make_series(alt)
    filt = nav.median_filtered("altitude", window_s=0.5)
    assert np.allclose(filt[:8], 3.0)
    assert np.allclose(filt[-8:], 3.5)


def test_median_filtered_zero_window_is_identity():
    alt = [3.0, 3.4, 2.9, 3.1]
    nav = make_series(alt)
    filt = nav.median_filtered("altitude", window_s=0.0)
    assert np.array_equal(filt, np.asarray(alt))
    # It is a copy, not a view of the series' array.
    filt[0] = -1.0
    assert nav.altitude[0] == 3.0


def test_median_filtered_irregular_sampling_uses_time_window():
    # Two dense clusters far apart in time: the window must not leak
    # across the gap even though the index distance is small.
    t = [0, 1, 2, 100, 101, 102]  # in 0.1 s units
    alt = [3.0, 3.0, 3.0, 5.0, 5.0, 5.0]
    samples = [
        NavSample(t_ns=BASE_NS + ti * 100_000_000,
                  north=0.0, east=0.0, depth=5.0, altitude=a,
                  roll=0.0, pitch=0.0, yaw=0.0,
                  yaw_rate=0.0, vel_x=0.5, lat=39.5, lon=2.4)
        for ti, a in zip(t, alt)
    ]
    nav = NavSeries(samples, origin_lat=39.5, origin_lon=2.4)
    filt = nav.median_filtered("altitude", window_s=1.0)
    assert np.allclose(filt[:3], 3.0)
    assert np.allclose(filt[3:], 5.0)


def test_median_filtered_is_deterministic():
    rng = np.random.default_rng(0)
    alt = 3.0 + 0.4 * rng.standard_normal(200)
    nav = make_series(alt)
    a = nav.median_filtered("altitude", window_s=2.0)
    b = nav.median_filtered("altitude", window_s=2.0)
    assert np.array_equal(a, b)


def test_median_filtered_rejects_yaw():
    nav = make_series([3.0, 3.0])
    with pytest.raises(ValueError, match="yaw"):
        nav.median_filtered("yaw", window_s=1.0)


def test_interp_values_matches_interp_on_raw_field():
    alt = [3.0, 3.2, 3.4, 3.6]
    nav = make_series(alt)
    t_query = BASE_NS + 150_000_000  # midway between samples 1 and 2
    assert nav.interp_values(t_query, nav.altitude) == pytest.approx(
        nav.interp(t_query, "altitude"))
    assert nav.interp_values(t_query, nav.altitude) == pytest.approx(3.3)


def test_interp_values_clamps_at_ends():
    nav = make_series([3.0, 4.0])
    assert nav.interp_values(BASE_NS - 10**9, nav.altitude) == 3.0
    assert nav.interp_values(BASE_NS + 10**10, nav.altitude) == 4.0


def test_interp_values_rejects_misaligned_array():
    nav = make_series([3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="one entry per sample"):
        nav.interp_values(BASE_NS, np.array([1.0, 2.0]))


def _sample(t_ns, altitude):
    return NavSample(t_ns=t_ns, north=0.0, east=0.0, depth=5.0,
                     altitude=altitude, roll=0.0, pitch=0.0, yaw=0.0,
                     yaw_rate=0.0, vel_x=0.5, lat=39.5, lon=2.4)


def test_duplicate_stamps_deduplicated_keeping_first():
    # A repeated stamp is a zero-width interval: interpolation would answer
    # with whichever duplicate lands adjacent. The first occurrence (message
    # order) must win and the duplicate must not distort interpolation.
    samples = [
        _sample(BASE_NS, 3.0),
        _sample(BASE_NS + 100_000_000, 4.0),
        _sample(BASE_NS + 100_000_000, 99.0),   # bogus resend
        _sample(BASE_NS + 200_000_000, 5.0),
    ]
    nav = NavSeries(samples, origin_lat=39.5, origin_lon=2.4)
    assert len(nav) == 3
    assert nav.altitude.tolist() == [3.0, 4.0, 5.0]
    assert nav.interp(BASE_NS + 150_000_000, "altitude") == pytest.approx(4.5)


def test_duplicate_stamps_dedup_survives_unsorted_input():
    # The sort is stable, so "first occurrence" means first in message order
    # even when the stream arrives out of order.
    samples = [
        _sample(BASE_NS + 200_000_000, 5.0),
        _sample(BASE_NS + 100_000_000, 4.0),
        _sample(BASE_NS, 3.0),
        _sample(BASE_NS + 100_000_000, 99.0),   # bogus resend, arrives later
    ]
    nav = NavSeries(samples, origin_lat=39.5, origin_lon=2.4)
    assert len(nav) == 3
    assert nav.altitude.tolist() == [3.0, 4.0, 5.0]


# --- Trajectory correction ------------------------------------------------
#
# Keyframes matching tests/fixtures/trajectory_refined.json, expressed in the
# make_series timebase: samples land at BASE_NS + k * 100 ms, keyframes at
# k = 2, 5, 8. Deltas chosen so the interpolated correction is exact at every
# integer k (see the expected arrays in test_with_correction_*).
_KF_K = (2, 5, 8)
_KF_DN = (1.0, 4.0, 4.0)
_KF_DE = (2.0, 2.0, -1.0)
_KF_DYAW = (0.1, 0.1, -0.2)


def make_trajectory(keyframes=None, ned_origin_lat=39.5, ned_origin_lon=2.4,
                    schema_version=1, status="ok", frame_type="relative"):
    """A trajectory_refined.json-shaped dict (only the fields the correction
    consumes)."""
    if keyframes is None:
        keyframes = [
            {"t_ns": BASE_NS + k * 100_000_000,
             "delta_n": dn, "delta_e": de, "delta_yaw": dyaw}
            for k, dn, de, dyaw in zip(_KF_K, _KF_DN, _KF_DE, _KF_DYAW)
        ]
    frame = {"type": frame_type}
    if ned_origin_lat is not None:
        frame["ned_origin_lat"] = ned_origin_lat
    if ned_origin_lon is not None:
        frame["ned_origin_lon"] = ned_origin_lon
    return {
        "schema_version": schema_version,
        "status": status,
        "frame": frame,
        "extrapolation": {"start": "hold", "end": "hold", "interp": "linear"},
        "refine_params_sha256": "deadbeefdeadbeef",
        "n_keyframes": len(keyframes),
        "keyframes": keyframes,
    }


def write_trajectory(tmp_path, artifact):
    path = tmp_path / "trajectory_refined.json"
    path.write_text(json.dumps(artifact))
    return str(path)


def test_load_trajectory_returns_validated_dict():
    art = load_trajectory(FIXTURE)
    assert art["schema_version"] == 1
    assert art["status"] == "ok"
    assert len(art["keyframes"]) == 3
    assert art["refine_params_sha256"] == "deadbeefdeadbeef"


def test_with_correction_applies_interpolated_deltas():
    # 11 samples at k = 0..10: north = k, east = 0, yaw = 0. The correction
    # holds before k=2 and after k=8, and is linear (in n) between keyframes.
    nav = make_series([3.0] * 11)  # altitude irrelevant here
    corrected = nav.with_correction(FIXTURE)

    assert isinstance(corrected, NavSeries)
    assert len(corrected) == 11
    # delta_n(k): hold 1.0 for k<=2, linear 1->4 for 2..5, flat 4 for k>=5.
    exp_north = np.array(
        [1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14], dtype=float)
    # delta_e(k): flat 2 for k<=5, linear 2->-1 for 5..8, hold -1 for k>=8.
    exp_east = np.array([2, 2, 2, 2, 2, 2, 1, 0, -1, -1, -1], dtype=float)
    # delta_yaw(k): flat .1 for k<=5, linear .1->-.2 for 5..8, hold -.2.
    exp_yaw = np.array(
        [.1, .1, .1, .1, .1, .1, 0, -.1, -.2, -.2, -.2], dtype=float)
    assert np.allclose(corrected.north, exp_north)
    assert np.allclose(corrected.east, exp_east)
    assert np.allclose(corrected.yaw, exp_yaw)


def test_apply_correction_holds_beyond_keyframe_span():
    # Samples before the first keyframe (k<2) hold the first delta; samples
    # after the last (k>8) hold the last delta — the artifact's start/end=hold.
    nav = make_series([3.0] * 11)
    corrected = apply_trajectory_correction(nav, make_trajectory())
    # k=0,1 held at (dn,de,dyaw) of keyframe 0 = (1, 2, .1).
    assert corrected.north[0] == pytest.approx(0 + 1.0)
    assert corrected.east[1] == pytest.approx(0 + 2.0)
    # k=9,10 held at keyframe 2 = (4, -1, -.2).
    assert corrected.north[10] == pytest.approx(10 + 4.0)
    assert corrected.east[9] == pytest.approx(0 - 1.0)
    assert corrected.yaw[10] == pytest.approx(-0.2)


def test_apply_correction_always_wraps_yaw_across_seam():
    # yaw is ALWAYS corrected; adding a positive delta near +pi must wrap into
    # (-pi, pi] via arctan2(sin, cos) rather than exceed pi.
    n = 5
    nav = make_series([3.0] * n, yaw=[3.10] * n)
    single_kf = [{"t_ns": BASE_NS, "delta_n": 0.0,
                  "delta_e": 0.0, "delta_yaw": 0.1}]
    corrected = apply_trajectory_correction(
        nav, make_trajectory(keyframes=single_kf))
    expected = np.arctan2(np.sin(3.10 + 0.1), np.cos(3.10 + 0.1))
    assert expected < 0  # 3.20 rad wrapped past +pi is negative
    assert np.all(corrected.yaw <= np.pi) and np.all(corrected.yaw > -np.pi)
    assert np.allclose(corrected.yaw, expected)


def test_apply_correction_copies_other_fields_and_leaves_input_unchanged():
    # Non-corrected fields are copied verbatim; the result is a fresh NavSeries
    # (origin preserved) and the input nav is not mutated.
    alts = [3.0, 3.5, 4.0, 3.2, 3.8, 3.1, 3.9, 3.3, 3.7, 3.0, 3.6]
    nav = make_series(alts)
    north_before = nav.north.copy()
    east_before = nav.east.copy()
    yaw_before = nav.yaw.copy()
    corrected = apply_trajectory_correction(nav, make_trajectory())
    assert corrected is not nav
    assert np.array_equal(corrected.altitude, nav.altitude)
    assert np.array_equal(corrected.depth, nav.depth)
    assert np.array_equal(corrected.lat, nav.lat)
    assert np.array_equal(corrected.lon, nav.lon)
    assert np.array_equal(corrected.vel_x, nav.vel_x)
    assert (corrected.origin_lat, corrected.origin_lon) == (39.5, 2.4)
    # The correction did shift the corrected series but left the input intact.
    assert not np.array_equal(corrected.north, nav.north)
    assert np.array_equal(nav.north, north_before)
    assert np.array_equal(nav.east, east_before)
    assert np.array_equal(nav.yaw, yaw_before)


def test_apply_correction_recomputes_yaw_unwrapped_from_corrected_yaw():
    # A fresh NavSeries recomputes yaw_unwrapped from the *corrected* yaw. With
    # a yaw ramp near +pi and a positive delta, consecutive corrected samples
    # straddle the seam: yaw itself jumps by ~2pi but yaw_unwrapped stays
    # continuous — impossible if yaw_unwrapped had been copied from the input.
    nav = make_series([3.0, 3.0], yaw=[3.0, 3.1])
    single_kf = [{"t_ns": BASE_NS, "delta_n": 0.0,
                  "delta_e": 0.0, "delta_yaw": 0.1}]
    corrected = apply_trajectory_correction(
        nav, make_trajectory(keyframes=single_kf))
    # Sample 0: wrap(3.1) = 3.1 (still <= pi); sample 1: wrap(3.2) < 0.
    assert corrected.yaw[0] == pytest.approx(3.1)
    assert corrected.yaw[1] < 0
    assert abs(corrected.yaw[1] - corrected.yaw[0]) > np.pi   # seam jump in yaw
    # ...but the unwrapped copy is continuous and reflects the corrected values.
    assert corrected.yaw_unwrapped[1] > np.pi
    assert np.allclose(corrected.yaw_unwrapped,
                       np.unwrap([3.1, np.arctan2(np.sin(3.2), np.cos(3.2))]))


def test_apply_correction_handles_unsorted_keyframes():
    # Keyframes supplied out of time order must be reordered before interp;
    # the corrected series must match the sorted-input result exactly.
    nav = make_series([3.0] * 11)
    sorted_art = make_trajectory()
    shuffled = list(reversed(sorted_art["keyframes"]))
    unsorted_art = make_trajectory(keyframes=shuffled)
    a = apply_trajectory_correction(nav, sorted_art)
    b = apply_trajectory_correction(nav, unsorted_art)
    assert np.array_equal(a.north, b.north)
    assert np.array_equal(a.east, b.east)
    assert np.array_equal(a.yaw, b.yaw)


def test_apply_correction_rejects_duplicate_keyframe_stamps():
    nav = make_series([3.0] * 4)
    dupes = [
        {"t_ns": BASE_NS, "delta_n": 1.0, "delta_e": 0.0, "delta_yaw": 0.0},
        {"t_ns": BASE_NS, "delta_n": 9.0, "delta_e": 0.0, "delta_yaw": 0.0},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        apply_trajectory_correction(nav, make_trajectory(keyframes=dupes))


def test_apply_correction_rejects_missing_keyframe_field():
    nav = make_series([3.0] * 4)
    incomplete = [{"t_ns": BASE_NS, "delta_n": 1.0, "delta_e": 0.0}]  # no dyaw
    with pytest.raises(ValueError, match="missing fields"):
        apply_trajectory_correction(nav, make_trajectory(keyframes=incomplete))


def test_apply_correction_rejects_float_t_ns():
    nav = make_series([3.0] * 4)
    floaty = [{"t_ns": float(BASE_NS), "delta_n": 1.0,
               "delta_e": 0.0, "delta_yaw": 0.0}]
    with pytest.raises(ValueError, match="integer nanosecond"):
        apply_trajectory_correction(nav, make_trajectory(keyframes=floaty))


def test_apply_correction_rejects_truncated_keyframe_list():
    # n_keyframes disagreeing with the actual list length is a truncation flag.
    nav = make_series([3.0] * 4)
    art = make_trajectory()
    art["n_keyframes"] = 99
    with pytest.raises(ValueError, match="truncated"):
        apply_trajectory_correction(nav, art)


def test_load_trajectory_rejects_bad_schema_version(tmp_path):
    path = write_trajectory(tmp_path, make_trajectory(schema_version=2))
    with pytest.raises(ValueError, match="schema_version"):
        load_trajectory(path)


def test_load_trajectory_rejects_non_ok_status(tmp_path):
    path = write_trajectory(tmp_path, make_trajectory(status="failed"))
    with pytest.raises(ValueError, match="status"):
        load_trajectory(path)


def test_load_trajectory_rejects_non_relative_frame(tmp_path):
    path = write_trajectory(tmp_path, make_trajectory(frame_type="absolute"))
    with pytest.raises(ValueError, match="frame.type"):
        load_trajectory(path)


def test_apply_correction_rejects_origin_mismatch():
    # A 0.1 deg latitude gap (~11 km) is the wrong mission's correction.
    nav = make_series([3.0] * 4)
    with pytest.raises(ValueError, match="wrong mission"):
        apply_trajectory_correction(nav, make_trajectory(ned_origin_lat=39.6))


def test_apply_correction_accepts_origin_within_tolerance():
    # ~0.56 m offset (5e-6 deg lat) is within the 1 m default tolerance.
    nav = make_series([3.0] * 11)
    corrected = apply_trajectory_correction(
        nav, make_trajectory(ned_origin_lat=39.5 + 5e-6))
    assert len(corrected) == 11


def test_apply_correction_rejects_missing_ned_origin():
    nav = make_series([3.0] * 4)
    art = make_trajectory(ned_origin_lat=None, ned_origin_lon=None)
    with pytest.raises(ValueError, match="ned_origin"):
        apply_trajectory_correction(nav, art)


def test_apply_correction_rejects_empty_keyframes():
    nav = make_series([3.0] * 4)
    with pytest.raises(ValueError, match="no keyframes"):
        apply_trajectory_correction(nav, make_trajectory(keyframes=[]))


# --- Placement reader -----------------------------------------------------
#
# `load_placement` reads the `placement` slot that offline_mosaicing.placement
# writes into trajectory_refined.json, reporting the (north, east) rigid
# translation that places a mission's floating products. It reuses
# load_trajectory for the path case; its own validation is placement-only.

PLACED_FIXTURE = (
    Path(__file__).parent / "fixtures" / "trajectory_refined_placed.json")

_DROP = object()  # an override value of `_DROP` removes that key entirely


def placement_slot(**overrides):
    """A `computed` placement slot; an override of `_DROP` drops that key."""
    slot = {
        "status": "computed",
        "method": "park_navavg_median",
        "translation_ne_m": [0.8887, -1.5396],   # [north, east]
        "rotation_deg": 0.0,
        "orientation_source": "ins",
        "refine_params_sha256": "deadbeefdeadbeef",
    }
    for key, value in overrides.items():
        if value is _DROP:
            slot.pop(key, None)
        else:
            slot[key] = value
    return slot


def artifact_with(slot, header_sha="deadbeefdeadbeef"):
    """A valid trajectory dict carrying `slot` as its `placement` block.

    `slot` or `header_sha` set to `_DROP` omits that piece.
    """
    art = make_trajectory()
    if header_sha is _DROP:
        art.pop("refine_params_sha256", None)
    else:
        art["refine_params_sha256"] = header_sha
    if slot is not _DROP:
        art["placement"] = slot
    return art


def test_placement_reader_is_exported_from_package():
    import bagio
    assert bagio.load_placement is load_placement
    assert bagio.PlacementInfo is PlacementInfo


def test_load_placement_reads_computed_slot_from_file():
    info = load_placement(PLACED_FIXTURE)
    assert isinstance(info, PlacementInfo)
    assert info.translation_ne_m == (0.8887, -1.5396)
    assert info.method == "park_navavg_median"
    assert info.refine_params_sha256 == "deadbeefdeadbeef"


def test_load_placement_accepts_path_and_dict_alike():
    from_path = load_placement(PLACED_FIXTURE)
    from_dict = load_placement(load_trajectory(PLACED_FIXTURE))
    assert from_path == from_dict


def test_load_placement_trusts_dict_without_revalidating_header():
    # The dict case is trusted (it is meant to come from load_trajectory) and
    # is NOT re-validated for the trajectory header — a bad header on a dict
    # still reads the placement slot. This locks that design in against a
    # future refactor silently adding header re-validation to the dict branch.
    art = artifact_with(placement_slot())
    art["schema_version"] = 2   # would fail load_trajectory; dict is trusted
    art["status"] = "failed"
    info = load_placement(art)
    assert info.translation_ne_m == (0.8887, -1.5396)


def test_load_placement_reports_null_method_as_none():
    # A computed slot missing its method violates the producer's invariant,
    # but the reader reports provenance as-is (method may be None) rather than
    # adding a refusal beyond the documented ones.
    info = load_placement(artifact_with(placement_slot(method=None)))
    assert info.method is None
    assert info.translation_ne_m == (0.8887, -1.5396)


@pytest.mark.parametrize("bad_arg", [3, None, [1, 2], 3.5])
def test_load_placement_rejects_non_dict_non_path_argument(bad_arg):
    # Anything that is neither a dict nor a path must fail with a clear
    # TypeError naming the received type — not fall through to open() (where
    # an int would be misread as a file descriptor).
    with pytest.raises(TypeError, match="must be a placement dict or a path"):
        load_placement(bad_arg)


def test_load_placement_reports_translation_in_stored_order():
    # north first, east second: the reader must not swap the pair.
    info = load_placement(artifact_with(
        placement_slot(translation_ne_m=[10.0, -3.0])))
    assert info.translation_ne_m == (10.0, -3.0)


def test_load_placement_rejects_missing_slot():
    with pytest.raises(ValueError, match="no placement slot"):
        load_placement(artifact_with(_DROP))


def test_load_placement_rejects_non_dict_slot():
    # A `placement` value that is present but not a dict (e.g. a stray list)
    # is as unusable as an absent slot.
    with pytest.raises(ValueError, match="no placement slot"):
        load_placement(artifact_with([1, 2]))


def test_load_placement_rejects_unplaced_status():
    # The stock fixture carries an `unplaced` slot: refuse it, and name both
    # the actual status and the remedy (the computed status).
    with pytest.raises(ValueError, match="unplaced") as exc:
        load_placement(FIXTURE)
    assert "computed" in str(exc.value)


def test_load_placement_rejects_anchored_status():
    with pytest.raises(ValueError, match="anchored") as exc:
        load_placement(artifact_with(placement_slot(status="anchored")))
    assert "computed" in str(exc.value)


@pytest.mark.parametrize("bad", [
    None,
    [1.0],
    [1.0, 2.0, 3.0],
    [float("nan"), 2.0],
    [1.0, float("inf")],
    ["1.0", "2.0"],
    [True, False],
    [10**400, 2.0],   # int too large for a C double -> not a usable number
])
def test_load_placement_rejects_malformed_translation(bad):
    with pytest.raises(ValueError, match="translation_ne_m"):
        load_placement(artifact_with(placement_slot(translation_ne_m=bad)))


@pytest.mark.parametrize("bad_rot", [5.0, -0.5, "0.0", float("nan"), True, False])
def test_load_placement_rejects_bad_rotation(bad_rot):
    # Only an exact numeric zero is a pure translation; a nonzero angle, NaN,
    # a bool, or a non-numeric value must all be refused.
    with pytest.raises(ValueError, match="rotation"):
        load_placement(artifact_with(placement_slot(rotation_deg=bad_rot)))


@pytest.mark.parametrize("ok_rot", [0.0, -0.0, 0, _DROP])
def test_load_placement_accepts_zero_or_absent_rotation(ok_rot):
    # 0.0/-0.0/int 0 are all pure translation; absent (dropped) is too.
    info = load_placement(artifact_with(placement_slot(rotation_deg=ok_rot)))
    assert info.translation_ne_m == (0.8887, -1.5396)


def test_load_placement_rejects_sha_mismatch():
    with pytest.raises(ValueError, match="refine_params_sha256"):
        load_placement(artifact_with(
            placement_slot(refine_params_sha256="0000feed0000feed")))


def test_load_placement_allows_absent_slot_sha():
    # With no slot sha to compare, the mismatch guard does not fire and the
    # reported provenance is None.
    info = load_placement(
        artifact_with(placement_slot(refine_params_sha256=_DROP)))
    assert info.refine_params_sha256 is None


def test_load_placement_path_case_reuses_trajectory_validation(tmp_path):
    # A bad header must be caught by load_trajectory in the path case, before
    # the placement slot is ever inspected.
    art = artifact_with(placement_slot())
    art["schema_version"] = 2
    path = write_trajectory(tmp_path, art)
    with pytest.raises(ValueError, match="schema_version"):
        load_placement(path)


def test_computed_slot_is_transparent_to_trajectory_correction():
    # schema_version stays 1: a slot-bearing artifact still validates and
    # applies through the trajectory path exactly as a slot-free one would.
    nav = make_series([3.0] * 11)
    placed = apply_trajectory_correction(nav, load_trajectory(PLACED_FIXTURE))
    plain = apply_trajectory_correction(nav, load_trajectory(FIXTURE))
    assert np.array_equal(placed.north, plain.north)
    assert np.array_equal(placed.east, plain.east)
    assert np.array_equal(placed.yaw, plain.yaw)


# --- Placement level ------------------------------------------------------
#
# The level says what a placement translation was measured from: the start
# park, the end park, or the INS navigation as recorded. load_placement
# derives it from the slot's `method` and `park.which`; it is reported, never
# a reason to refuse the slot.

REAL_SLOTS = Path(__file__).parent / "fixtures" / "placement_slots_start_park.json"


def test_placement_vocabulary_is_exported_from_package():
    import bagio
    assert bagio.PLACEMENT_LEVELS is PLACEMENT_LEVELS
    assert PLACEMENT_LEVELS == ("start_park", "end_park", "ins_as_recorded")
    assert bagio.PLACEMENT_METHOD_PARK == "park_navavg_median"
    assert bagio.PLACEMENT_METHOD_INS == "ins_as_recorded"


@pytest.mark.parametrize("which, level", [("start", "start_park"),
                                          ("end", "end_park")])
def test_load_placement_reports_the_park_level(which, level):
    info = load_placement(artifact_with(placement_slot(park={"which": which})))
    assert info.level == level


def test_load_placement_reports_the_ins_as_recorded_level():
    # A computed slot whose method is ins_as_recorded, with a finite
    # translation and no rotation, is accepted as it stands.
    info = load_placement(artifact_with(placement_slot(
        method="ins_as_recorded", translation_ne_m=[0.0, 0.0])))
    assert info.level == "ins_as_recorded"
    assert info.translation_ne_m == (0.0, 0.0)


@pytest.mark.parametrize("overrides", [
    {},                                                   # no park block
    {"park": None},
    {"park": "start"},                                    # not a block
    {"park": {}},                                         # no which
    {"park": {"which": "middle"}},                        # unknown which
    {"park": {"which": ["start"]}},                       # not a string
    {"method": None, "park": {"which": "start"}},         # no method
    {"method": "other", "park": {"which": "start"}},      # unknown method
])
def test_load_placement_reports_an_undetermined_level_as_none(overrides):
    info = load_placement(artifact_with(placement_slot(**overrides)))
    assert info.level is None
    assert info.translation_ne_m == (0.8887, -1.5396)


def test_load_placement_level_of_the_stock_placed_fixture_is_none():
    # The fixture's slot records its method but no park block.
    assert load_placement(PLACED_FIXTURE).level is None


@pytest.mark.parametrize("mission, translation", [
    ("12_52_41", (0.6027, -2.3969)),
    ("13_52_22", (0.8887, -1.5396)),
])
def test_load_placement_reads_the_level_of_real_park_slots(mission,
                                                           translation):
    # Two complete slots written by the park estimator from the start park.
    # Each also carries a cross_check block whose own park window is the end
    # park: only the top-level park decides the level.
    real = json.loads(REAL_SLOTS.read_text())[mission]
    assert real["placement"]["cross_check"]["park_window"]["which"] == "end"

    info = load_placement(real)

    assert info.level == "start_park"
    assert info.translation_ne_m == translation
    assert info.method == "park_navavg_median"


def test_placement_info_level_is_optional():
    info = PlacementInfo(translation_ne_m=(0.8887, -1.5396),
                         method="park_navavg_median",
                         refine_params_sha256=None)
    assert info.level is None


@pytest.mark.parametrize("level", PLACEMENT_LEVELS)
def test_placement_info_accepts_every_level(level):
    info = PlacementInfo((0.0, 0.0), "m", None, level=level)
    assert info.level == level


@pytest.mark.parametrize("bad", ["start", "START_PARK", "", 1])
def test_placement_info_refuses_a_level_outside_the_vocabulary(bad):
    with pytest.raises(ValueError, match="PlacementInfo.level"):
        PlacementInfo((0.0, 0.0), "m", None, level=bad)
