"""Synthetic NavSeries tests (no dataset needed)."""

import numpy as np
import pytest

from bagio.nav_series import NavSample, NavSeries

BASE_NS = 1_782_993_228_000_000_000  # 2026-epoch magnitude on purpose


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
    # A single-sample bottom-lock hop (the E1 signature) must vanish while
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
