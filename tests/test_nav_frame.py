"""NavFrame: the navigation frame <-> geodetic <-> map conversions.

The reference for the geodetic part is independent of PROJ
(``geodesy_reference``: a numpy port of ``cola2_lib``'s C++ ``NED``, the
conversion the vehicle applies to produce ``NavSts`` north/east, inverted by
fixed-point iteration). The map step of the reference is pyproj's own
geodetic -> map transformation.

The tests reading recorded navigation are skipped when the dataset is not
mounted (override the day folder with ``BAGIO_TEST_ANDRATX_DAY``); when it is
mounted, a missing bag fails rather than skips.
"""

import math
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from geodesy_reference import (cola2_geodetic2ned, cola2_ned2geodetic,
                               geodetic_gap_m, reference_geodetic)
from pyproj import Proj, Transformer

from bagio import BagSet, LocalAffine, NavFrame, NavSeries, PlacementInfo
from bagio.nav_series import NavSample

# The NED origin of the Andratx 2026-07-02 missions (UTM zone 31N).
LAT0, LON0 = 39.5437387, 2.3779181
UTM31 = "EPSG:32631"

UM = 1e-6
MM = 1e-3


def grid(half, k=21):
    t = np.linspace(-half, half, k)
    n, e = np.meshgrid(t, t, indexing="ij")
    return n.ravel(), e.ravel()


@pytest.fixture(scope="module")
def frame():
    return NavFrame(LAT0, LON0, UTM31)


# ------------------------------------------------- independent reference

@pytest.mark.parametrize("origin, crs", [
    ((LAT0, LON0), UTM31),
    ((-33.9, 18.4), "EPSG:32734"),     # southern hemisphere, UTM 34S
    ((60.2, 24.9), "EPSG:32635"),      # high latitude, UTM 35N
])
def test_to_map_matches_independent_reference(origin, crs):
    f = NavFrame(*origin, crs)
    n, e = grid(10_000.0)
    x, y = f.to_map(n, e)
    lat, lon = reference_geodetic(*origin, n, e)
    rx, ry = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        lon, lat)
    assert np.max(np.hypot(x - rx, y - ry)) <= UM
    glat, glon = f.to_geodetic(n, e)
    assert np.max(geodetic_gap_m(lat, lon, glat, glon)) <= UM


def test_from_geodetic_is_the_vehicle_conversion(frame):
    rng = np.random.default_rng(1)
    lat = LAT0 + rng.uniform(-0.05, 0.05, 500)
    lon = LON0 + rng.uniform(-0.05, 0.05, 500)
    n, e = frame.from_geodetic(lat, lon)
    rn, re = cola2_geodetic2ned(LAT0, LON0, lat, lon)
    assert np.max(np.hypot(n - rn, e - re)) <= UM


def test_cola2_tangent_plane_inverse_departs_only_far_out(frame):
    # cola2_lib's ned2Geodetic(n, e, 0) maps the tangent-plane point: equal to
    # the exact inverse at mission scale (d^3 / 2R^2 = 0.3 um at 283 m),
    # centimetres off at 14 km.
    n, e = grid(200.0)
    lat, lon = frame.to_geodetic(n, e)
    clat, clon = cola2_ned2geodetic(LAT0, LON0, n, e)
    assert np.max(geodetic_gap_m(lat, lon, clat, clon)) <= UM
    n_far = np.array([10_000.0, -10_000.0])
    e_far = np.array([10_000.0, 10_000.0])
    lat, lon = frame.to_geodetic(n_far, e_far)
    clat, clon = cola2_ned2geodetic(LAT0, LON0, n_far, e_far)
    gaps = geodetic_gap_m(lat, lon, clat, clon)
    assert np.all((gaps > 0.02) & (gaps < 0.05))


# ------------------------------------------------------------ round trips

def test_round_trips(frame):
    n, e = grid(10_000.0)
    rn, re = frame.from_map(*frame.to_map(n, e))
    assert np.max(np.hypot(rn - n, re - e)) <= UM
    rn, re = frame.from_geodetic(*frame.to_geodetic(n, e))
    assert np.max(np.hypot(rn - n, re - e)) <= UM


def test_far_points_raise_instead_of_diverging(frame):
    with pytest.raises(ValueError, match="to_geodetic"):
        frame.to_geodetic(1e9, 0.0)


# -------------------------------------------- recorded navigation (dataset)

DAY = Path(os.environ.get(
    "BAGIO_TEST_ANDRATX_DAY",
    "/home/rosuser/dataset/2026/orat/Andratx/2026_07_02",
))


@pytest.mark.skipif(not DAY.is_dir(), reason=f"validation day not mounted at {DAY}")
@pytest.mark.parametrize("mission", ["12_52_41", "13_52_22"])
def test_recorded_navigation_is_reproduced(mission):
    bags = sorted((DAY / mission).glob("sparus2_2*_0.bag"))
    assert len(bags) == 1, f"expected one main bag in {mission}, got {bags}"
    with BagSet([str(bags[0])]) as bag_set:
        nav = NavSeries.from_messages(
            m for _, _, m in bag_set.messages(
                topic_suffixes=["navigator/navigation"]))
    assert len(nav) > 10_000
    f = NavFrame.from_nav(nav, UTM31)
    n, e = f.from_geodetic(nav.lat, nav.lon)
    assert np.max(np.hypot(n - nav.north, e - nav.east)) <= 1e-5
    lat, lon = f.to_geodetic(nav.north, nav.east)
    assert np.max(geodetic_gap_m(nav.lat, nav.lon, lat, lon)) <= 1e-5


# --------------------------------------------------------- at the origin

def test_origin_and_local_factors(frame):
    x, y = frame.to_map(0.0, 0.0)
    ox, oy = Transformer.from_crs("EPSG:4326", UTM31, always_xy=True).transform(
        LON0, LAT0)
    assert math.hypot(x - ox, y - oy) <= 1e-9
    factors = Proj(UTM31).get_factors(LON0, LAT0)
    j = frame.local_affine((0.0, 0.0), 10.0).jacobian
    # Column 1 is the nav east axis in the map; the map is conformal, so the
    # Jacobian is a rotation times the point scale.
    rotation_deg = math.degrees(math.atan2(j[1, 1], j[0, 1]))
    assert abs(rotation_deg - factors.meridian_convergence) <= 1e-6
    assert abs(math.sqrt(abs(np.linalg.det(j))) - factors.meridional_scale) <= 1e-8
    assert abs(j[0, 0] + j[1, 1]) <= 1e-9 and abs(j[1, 0] - j[0, 1]) <= 1e-9


# ------------------------------------------------------- affine bounds

@pytest.mark.parametrize("half, bound", [(300.0, 0.2 * MM), (500.0, 0.5 * MM)])
def test_local_affine_residual_bound(frame, half, bound):
    aff = frame.local_affine((0.0, 0.0), half)
    assert isinstance(aff, LocalAffine)
    assert aff.max_residual_m <= bound
    rng = np.random.default_rng(2)
    n, e = rng.uniform(-half, half, (2, 200))
    ax, ay = aff.apply(n, e)
    x, y = frame.to_map(n, e)
    assert np.max(np.hypot(ax - x, ay - y)) <= aff.max_residual_m + 1e-9
    assert aff.apply(0.0, 0.0) == frame.to_map(0.0, 0.0)


def test_local_affine_refuses_beyond_tolerance(frame):
    with pytest.raises(ValueError, match="departs"):
        frame.local_affine((0.0, 0.0), 3000.0)


def _affine_xy(coeffs, col, row):
    a, b, c, d, e, f = coeffs
    return a * col + b * row + c, d * col + e * row + f


@pytest.mark.parametrize("nw, shape, offset", [
    ((150.0, -150.0), (600, 600), None),
    ((150.0, -150.0), (600, 600), (0.6027, -2.3969)),
    ((2000.0, 3000.0), (200, 800), (0.8887, -1.5396)),   # strip, off-origin
])
def test_raster_affine_corners(frame, nw, shape, offset):
    pixel = 0.5
    coeffs = frame.raster_affine(nw, pixel, shape, offset_ne_m=offset)
    rows, cols = shape
    for col, row in ((0, 0), (cols, 0), (0, rows), (cols, rows),
                     (cols // 3, rows // 2)):
        x, y = _affine_xy(coeffs, col, row)
        ex, ey = frame.to_map(nw[0] - row * pixel, nw[1] + col * pixel,
                              offset_ne_m=offset)
        assert math.hypot(x - ex, y - ey) <= 0.2 * MM
    a, b, _, d, e, _ = coeffs
    assert b != 0.0 and d != 0.0                      # rotation terms
    # The pixel side on the map is the point scale at the raster's centre
    # (UTM's scale varies with the distance from the central meridian), times
    # the tangent plane's own radial scale, 1 + d^2 / 2R^2 (1.3e-7 at the
    # strip, 3.8 km out).
    t = offset or (0.0, 0.0)
    clat, clon = frame.to_geodetic(nw[0] + t[0] - rows * pixel / 2,
                                   nw[1] + t[1] + cols * pixel / 2)
    k = Proj(UTM31).get_factors(clon, clat).meridional_scale
    assert math.hypot(a, d) == pytest.approx(pixel * k, rel=5e-7)


def test_raster_affine_checks_the_footprint_not_a_square(frame):
    # A 2 km x 40 m east-west strip departs from its affine less than its
    # 2 km bounding square does: a limit between the two passes the strip.
    square = frame.local_affine((0.0, 0.0), 1000.0, tol_m=1.0).max_residual_m
    limit = 0.75 * square
    frame.raster_affine((20.0, -1000.0), 0.5, (80, 4000), tol_m=limit)
    with pytest.raises(ValueError, match="local_affine"):
        frame.local_affine((0.0, 0.0), 1000.0, tol_m=limit)


# ------------------------------------------------------------ placement

def test_offset_is_applied_in_the_nav_frame(frame):
    t = (0.6027, -2.3969)
    n, e = grid(200.0)
    assert np.array_equal(np.stack(frame.to_map(n, e, offset_ne_m=t)),
                          np.stack(frame.to_map(n + t[0], e + t[1])))
    p = (37.0, -12.0)
    shift = frame.map_shift(t, p)
    x1, y1 = frame.to_map(p[0] + t[0], p[1] + t[1])
    x0, y0 = frame.to_map(*p)
    assert shift == (x1 - x0, y1 - y0)
    back = frame.from_map(*frame.to_map(n, e, offset_ne_m=t), offset_ne_m=t)
    assert np.max(np.hypot(back[0] - n, back[1] - e)) <= UM


def test_offset_on_map_axes_is_wrong_by_grid_convergence(frame):
    # Record: a (3, -3) m translation added on the UTM axes instead of in the
    # nav frame misses by 29 mm at 39.5 N, 2.4 E.
    dx, dy = frame.map_shift((3.0, -3.0), (0.0, 0.0))
    assert math.hypot(dx - (-3.0), dy - 3.0) == pytest.approx(29.36 * MM, abs=0.1 * MM)


# ---------------------------------- the two shortcuts on two recorded surveys

# Record of how far the two shortcuts this module replaces put the same point,
# on the north-west corners of two photomosaics of the Andratx 2026-07-02
# validation day (corner and placement as recorded in their products):
#
# - "spherical": the corner taken to lat/lon with a spherical metres-per-degree
#   constant (longitude scaled by the cosine of the origin latitude), the
#   placement applied the same way, then projected exactly;
# - "axes": the nav metres (placement included) added to the map coordinates
#   of the origin.
M_PER_DEG = 111319.5

SURVEYS = {
    # recorded corner (lat, lon) under the spherical shortcut, placement
    # (north, east) m, and the expected gaps (easting, northing) m: axes -
    # spherical (its length), spherical - exact, axes - exact
    "12_52_41": dict(corner=(39.5435243644, 2.3777336837), t=(0.6027, -2.3969),
                     gap=0.256, spherical_exact=(-0.021, 0.063),
                     axes_exact=(0.159, -0.119)),
    "13_52_22": dict(corner=(39.5435202130, 2.3776935835), t=(0.8887, -1.5396),
                     gap=0.279, spherical_exact=(-0.026, 0.064),
                     axes_exact=(0.161, -0.143)),
}


@pytest.mark.parametrize("survey", sorted(SURVEYS))
def test_shortcuts_disagree_as_recorded(frame, survey):
    c = SURVEYS[survey]
    lat_c, lon_c = c["corner"]
    tn, te = c["t"]
    cos0 = math.cos(math.radians(LAT0))
    # The spherical shortcut inverted: the unplaced corner, then the corner
    # in nav metres.
    nw_n = (lat_c - tn / M_PER_DEG - LAT0) * M_PER_DEG
    nw_e = (lon_c - te / (M_PER_DEG * cos0) - LON0) * M_PER_DEG * cos0
    to_utm = Transformer.from_crs("EPSG:4326", UTM31, always_xy=True)
    spherical = np.array(to_utm.transform(lon_c, lat_c))
    axes = (np.array(to_utm.transform(LON0, LAT0))
            + np.array([nw_e + te, nw_n + tn]))
    exact = np.array(frame.to_map(nw_n, nw_e, offset_ne_m=c["t"]))
    assert np.hypot(*(axes - spherical)) == pytest.approx(c["gap"], abs=0.005)
    assert np.allclose(spherical - exact, c["spherical_exact"], atol=0.002)
    assert np.allclose(axes - exact, c["axes_exact"], atol=0.002)


# ------------------------------------------------------------- map CRS

@pytest.mark.parametrize("crs, match", [
    ("EPSG:4326", "not a projected"),
    ("EPSG:2227", "not in metres"),            # California zone 3, US feet
    ("EPSG:22275", "not east and north"),      # South Africa Lo15: west/south
    ("EPSG:23031", "not WGS 84 or ETRS89"),    # ED50 / UTM 31N
    ("not-a-crs", "pyproj understands"),
])
def test_unsuitable_map_crs_raises(crs, match):
    with pytest.raises(ValueError, match=match):
        NavFrame(LAT0, LON0, crs)


def test_etrs89_label_moves_nothing_measurable(frame):
    n, e = grid(100.0, 5)
    x1, y1 = frame.to_map(n, e)
    x2, y2 = NavFrame(LAT0, LON0, "EPSG:25831").to_map(n, e)
    assert np.max(np.hypot(x1 - x2, y1 - y2)) <= 0.2 * MM


def test_frame_without_map_crs():
    f = NavFrame(LAT0, LON0)
    assert f.map_crs is None
    assert f.from_geodetic(LAT0, LON0) == pytest.approx((0.0, 0.0), abs=1e-9)
    for call in (lambda: f.to_map(1.0, 1.0), lambda: f.from_map(1.0, 1.0),
                 lambda: f.local_affine((0.0, 0.0), 10.0),
                 lambda: f.raster_affine((0.0, 0.0), 1.0, (2, 2))):
        with pytest.raises(ValueError, match="without a map CRS"):
            call()
    assert NavFrame(LAT0, LON0, UTM31).map_crs == UTM31


# ----------------------------------------------------- shapes and values

def test_scalars_give_floats(frame):
    results = [frame.to_map(1.0, 2.0), frame.to_geodetic(1.0, 2.0),
               frame.from_geodetic(LAT0, LON0), frame.from_map(446_600.0,
                                                               4_377_300.0),
               frame.map_shift((1.0, 1.0), (0.0, 0.0)),
               frame.local_affine((0.0, 0.0), 10.0).apply(1.0, 2.0)]
    for pair in results:
        assert all(type(v) is float for v in pair)


def test_arrays_keep_their_shape(frame):
    lat, lon = frame.to_geodetic(np.zeros(5), 3.0)
    assert lat.shape == lon.shape == (5,)
    x, y = frame.to_map(np.zeros((3, 4)), np.ones((3, 4)))
    assert x.shape == y.shape == (3, 4) and x.dtype == np.float64


def test_non_finite_inputs_give_nan(frame):
    n = np.array([0.0, np.nan, 5.0, np.inf])
    e = np.array([1.0, 1.0, np.nan, 0.0])
    for out in (frame.to_map(n, e), frame.to_geodetic(n, e),
                frame.from_geodetic(LAT0 + n * 1e-5, LON0 + e * 1e-5),
                frame.from_map(*frame.to_map(n, e))):
        a, b = out
        assert np.isfinite(a[0]) and np.isfinite(b[0])
        assert np.isnan(a[1:]).all() and np.isnan(b[1:]).all()


def test_latitude_out_of_range_raises(frame):
    with pytest.raises(ValueError, match="latitude"):
        frame.from_geodetic(95.0, LON0)


def test_placement_info_is_an_offset(frame):
    placement = PlacementInfo(translation_ne_m=(0.6027, -2.3969), method=None,
                              refine_params_sha256=None)
    assert (frame.to_map(3.0, 4.0, offset_ne_m=placement)
            == frame.to_map(3.0, 4.0, offset_ne_m=(0.6027, -2.3969)))
    assert (frame.map_shift(placement, (0.0, 0.0))
            == frame.map_shift((0.6027, -2.3969), (0.0, 0.0)))


def test_pickle_round_trip(frame):
    before = frame.to_map(12.0, -7.0)
    clone = pickle.loads(pickle.dumps(frame))
    assert clone.to_map(12.0, -7.0) == before
    assert (clone.origin_lat, clone.origin_lon, clone.map_crs) == \
        (frame.origin_lat, frame.origin_lon, frame.map_crs)


def test_origin_and_crs_are_read_only(frame):
    for name in ("origin_lat", "origin_lon", "map_crs"):
        with pytest.raises(AttributeError):
            setattr(frame, name, 0.0)


def test_from_nav_uses_the_series_origin():
    s = NavSample(t_ns=1, north=0.0, east=0.0, depth=0.0, altitude=0.0,
                  roll=0.0, pitch=0.0, yaw=0.0, yaw_rate=0.0, vel_x=0.0,
                  lat=LAT0, lon=LON0)
    f = NavFrame.from_nav(NavSeries([s], LAT0, LON0), UTM31)
    assert (f.origin_lat, f.origin_lon, f.map_crs) == (LAT0, LON0, UTM31)


def test_longitude_is_periodic(frame):
    shifted = NavFrame(LAT0, LON0 + 360.0, UTM31)
    assert shifted.origin_lon == LON0 + 360.0
    x1, y1 = frame.to_map(50.0, -20.0)
    x2, y2 = shifted.to_map(50.0, -20.0)
    assert math.hypot(x1 - x2, y1 - y2) <= UM


@pytest.mark.parametrize("lat, lon", [(float("nan"), 2.0), (95.0, 2.0),
                                      (True, 2.0), ("39", 2.0), (10**400, 2.0),
                                      (39.0, float("inf"))])
def test_invalid_origin_raises(lat, lon):
    with pytest.raises(ValueError, match="origin"):
        NavFrame(lat, lon, UTM31)


@pytest.mark.parametrize("offset", [(1.0, 2.0, 3.0), (float("nan"), 0.0),
                                    "ab", (True, 0.0), 5.0, (10**400, 0.0)])
def test_invalid_offset_raises(frame, offset):
    with pytest.raises(ValueError, match="offset_ne_m"):
        frame.to_map(0.0, 0.0, offset_ne_m=offset)


@pytest.mark.parametrize("kwargs, match", [
    (dict(shape=(0, 5)), "shape"),
    (dict(shape=(1.5, 2)), "shape"),
    (dict(shape=5), "shape"),
    (dict(pixel_m=0.0), "pixel_m"),
    (dict(nw_ne_m=(1.0,)), "nw_ne_m"),
    (dict(tol_m=float("nan")), "tol_m"),
])
def test_raster_affine_rejects_bad_arguments(frame, kwargs, match):
    args = dict(nw_ne_m=(0.0, 0.0), pixel_m=0.5, shape=(10, 10))
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        frame.raster_affine(**args)


@pytest.mark.parametrize("half, tol, match", [
    (0.0, 1e-3, "half_extent_m"), (10.0, -1.0, "tol_m"),
    (float("inf"), 1e-3, "half_extent_m")])
def test_local_affine_rejects_bad_arguments(frame, half, tol, match):
    with pytest.raises(ValueError, match=match):
        frame.local_affine((0.0, 0.0), half, tol)


def test_local_affine_validates_and_freezes_its_jacobian():
    aff = LocalAffine(center_ne_m=(0.0, 0.0), center_map=(10.0, 20.0),
                      jacobian=[[0.0, 1.0], [1.0, 0.0]], max_residual_m=0.0)
    assert aff.apply(2.0, 3.0) == (13.0, 22.0)
    for a in (aff, pickle.loads(pickle.dumps(aff))):
        with pytest.raises(ValueError):
            a.jacobian[0, 0] = 5.0
    with pytest.raises(ValueError, match="2x2"):
        LocalAffine((0.0, 0.0), (0.0, 0.0), [[1.0, 0.0]], 0.0)
