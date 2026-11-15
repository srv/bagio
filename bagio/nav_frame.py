"""Exact conversion between the navigation frame and a map CRS.

The **navigation frame** is the ``north``/``east`` metres of
``cola2_msgs/NavSts`` relative to the mission's NED origin
(``NavSts.origin``). On the vehicle, COLA2 computes them from the INS
latitude/longitude with ``cola2_lib``'s ``NED::geodetic2Ned(lat, lon, 0)``:
the point at height 0 on the WGS84 ellipsoid, expressed in the plane tangent
to the ellipsoid at the origin (an ECEF local tangent plane, geodetic
latitude). ``navigator/gps_ned`` uses the same conversion, so trajectory
corrections and placement translations measured against it live in this
frame too.

:class:`NavFrame` implements that conversion, its exact inverse, and the link
to a projected **map CRS** (e.g. a UTM zone), all through PROJ:

- nav ``(north, east)`` -> geodetic ``(lat, lon)`` -> map
  ``(easting, northing)``, point by point;
- a nav-frame translation (a placement) is applied **in the nav frame**,
  before converting;
- a local affine of the conversion, with its residual against the exact
  conversion measured and capped, for rasters.

Neither shortcut is exact. Adding nav metres to the map axes ignores the grid
convergence (the angle between true north and the map's grid north) and the
map's point scale: a rotation (-0.396 deg at 39.5 N, 2.4 E in UTM 31N) and a
scale (0.99964 there) about the origin, together about 0.7 % of the distance
from the origin. A spherical metres-per-degree constant misplaces latitude by
about 0.26 % of the north distance there.

The inverse is exact: ``cola2_lib``'s own ``NED::ned2Geodetic(n, e, 0)`` maps
the point of the tangent plane, which lies above the ellipsoid by about
d^2 / 2R (d the distance from the origin, R the Earth radius), and so departs
from the inverse of ``geodetic2Ned(lat, lon, 0)`` by about d^3 / 2R^2
(12 nm at 100 m, about 3.5 cm at 14 km). :meth:`NavFrame.to_geodetic` moves
the point along the origin's vertical until it is on the ellipsoid.

Conventions: nav-frame inputs are **north first** (as ``NavSts``,
:class:`~bagio.nav_series.PlacementInfo` and ``navigator/gps_ned``); map
outputs are **(easting, northing)**, the x, y order of pyproj and GDAL.
Inputs are scalars or array-likes that broadcast together: scalars give
Python floats, arrays give float64 arrays of the broadcast shape. A non-finite
input gives NaN for that point; a finite input that cannot be converted
raises ``ValueError``.
"""

import math
import numbers
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection
from pyproj.exceptions import CRSError, ProjError

# The datum of the navigation's latitude/longitude, and the ellipsoid of
# COLA2's NED class (WGS84 constants).
_GEODETIC_CRS = "EPSG:4326"
_ELLIPSOID = "WGS84"

# The geodetic CRSs a map CRS may be built on: WGS 84 itself, and ETRS89,
# which EPSG ties to WGS 84 by a null transformation ("ETRS89 to WGS 84 (1)").
# Any other datum needs a real datum transformation, which this module does
# not choose on the caller's behalf.
_MAP_GEODETIC_EPSG = frozenset({4326, 4258})

# The inverse moves a point along the origin's vertical until its geodetic
# height is within this tolerance of 0. Each correction shrinks the remaining
# height by about (d/R)^2 / 2 (d the distance from the origin, R the Earth
# radius), so one correction suffices within about 6 km of the origin and two
# within about 60 km; the step bound limits the frame to about 200 km.
_HEIGHT_TOL_M = 1e-6
_MAX_HEIGHT_STEPS = 4

# Samples per side of the rectangle over which an affine is fitted and checked.
_AFFINE_SAMPLES = 101

_METRE_UNITS = frozenset({"metre", "meter"})


def _finite_real(value):
    """True for a finite real number (Python or NumPy), excluding ``bool``."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:      # an int too large for a double
        return False


def _pair(value, what, order="(north, east)"):
    """``value`` as a pair of finite floats."""
    try:
        first, second = value
    except (TypeError, ValueError):
        raise ValueError(
            f"{what} must be a {order} pair, got {value!r}") from None
    if not (_finite_real(first) and _finite_real(second)):
        raise ValueError(
            f"{what} must hold two finite numbers, got {value!r}")
    return float(first), float(second)


def _offset(offset_ne_m):
    """A nav-frame translation ``(dn, de)``; ``None`` is no translation.

    Accepts a ``(north, east)`` pair or anything carrying one as
    ``translation_ne_m`` (a :class:`~bagio.nav_series.PlacementInfo`).
    """
    if offset_ne_m is None:
        return 0.0, 0.0
    return _pair(getattr(offset_ne_m, "translation_ne_m", offset_ne_m),
                 "offset_ne_m")


def _positive(value, what):
    if not (_finite_real(value) and value > 0):
        raise ValueError(f"{what} must be a positive finite number, "
                         f"got {value!r}")
    return float(value)


def _broadcast(*values):
    """Flat float64 copies of ``values`` broadcast together, their common
    shape, and whether every input was a scalar."""
    try:
        arrays = [np.asarray(v, dtype=np.float64) for v in values]
        scalar = all(a.ndim == 0 for a in arrays)
        arrays = np.broadcast_arrays(*arrays)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"coordinates must be numbers or numeric arrays "
                         f"that broadcast together: {exc}") from exc
    shape = arrays[0].shape
    return [a.reshape(-1).copy() for a in arrays], shape, scalar


def _shaped(flat, shape, scalar):
    return float(flat[0]) if scalar else flat.reshape(shape)


def _number_text(value):
    """``value`` as a PROJ parameter: the shortest decimal that round-trips,
    without an exponent (PROJ's angle parser could read an ``e`` as East)."""
    return np.format_float_positional(value, unique=True, trim="-")


def _normalize_map_crs(map_crs):
    if map_crs is None:
        return None
    try:
        crs = CRS.from_user_input(map_crs)
    except CRSError as exc:
        raise ValueError(
            f"map_crs {map_crs!r} is not a CRS pyproj understands: {exc}"
        ) from exc
    name = crs.to_string()
    if not crs.is_projected:
        raise ValueError(f"map_crs {name} is not a projected CRS")
    axes = crs.axis_info[:2]
    units = {axis.unit_name for axis in axes}
    if not units <= _METRE_UNITS:
        raise ValueError(
            f"map_crs {name} is not in metres (axis units {sorted(units)})")
    directions = sorted(axis.direction.lower() for axis in axes)
    if directions != ["east", "north"]:
        raise ValueError(
            f"map_crs {name} has axes {directions}, not east and north")
    geodetic = crs.geodetic_crs
    if geodetic is None or geodetic.to_epsg() not in _MAP_GEODETIC_EPSG:
        raise ValueError(
            f"map_crs {name} is on "
            f"{geodetic.name if geodetic is not None else 'no geodetic CRS'}, "
            "not WGS 84 or ETRS89: reaching it from the navigation's WGS 84 "
            "needs a datum transformation this module does not choose")
    return crs


def _transform(transformer, *arrays, **kwargs):
    """``transformer.transform`` on equal-length flat float64 arrays,
    returning float64 arrays.

    pyproj sends inputs it can convert to scalars down its single-point path,
    and NumPy deprecates converting a one-element array to a scalar, so
    one-element inputs are passed as floats.
    """
    if arrays[0].size == 1:
        out = transformer.transform(*(float(a[0]) for a in arrays), **kwargs)
        return tuple(np.array([v], dtype=np.float64) for v in out)
    return tuple(np.asarray(v, dtype=np.float64)
                 for v in transformer.transform(*arrays, **kwargs))


def _require_finite(arrays, caller):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise ValueError(
                f"{caller}: {int(np.sum(~np.isfinite(a)))} point(s) could not "
                "be converted (outside the domain of the projection or of the "
                "conversion)")


@dataclass(frozen=True, eq=False)
class LocalAffine:
    """An affine stand-in for :meth:`NavFrame.to_map` over a rectangle.

    ``apply(north, east)`` returns ``center_map + jacobian @ (north - n0,
    east - e0)`` with ``(n0, e0) = center_ne_m``. ``jacobian`` is a read-only
    2x2 array: rows easting, northing; columns north, east. Its slopes are the
    least-squares fit to the exact conversion over the rectangle, and the
    affine is then pinned to pass through ``center_map``, the exact
    ``(easting, northing)`` of the centre. ``max_residual_m`` is the largest distance between
    the affine and the exact conversion over a 101 x 101 sample of the
    rectangle.
    """

    center_ne_m: tuple
    center_map: tuple
    jacobian: np.ndarray
    max_residual_m: float

    def __post_init__(self):
        jacobian = np.array(self.jacobian, dtype=np.float64)
        if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
            raise ValueError(
                f"jacobian must be a finite 2x2 array, got {self.jacobian!r}")
        jacobian.setflags(write=False)
        if not (_finite_real(self.max_residual_m)
                and self.max_residual_m >= 0):
            raise ValueError("max_residual_m must be a finite number >= 0, "
                             f"got {self.max_residual_m!r}")
        object.__setattr__(self, "center_ne_m",
                           _pair(self.center_ne_m, "center_ne_m"))
        object.__setattr__(self, "center_map",
                           _pair(self.center_map, "center_map",
                                 "(easting, northing)"))
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "max_residual_m",
                           float(self.max_residual_m))

    def __reduce__(self):
        return (type(self), (self.center_ne_m, self.center_map,
                             np.array(self.jacobian), self.max_residual_m))

    def apply(self, north, east):
        """Map ``(easting, northing)`` of nav ``(north, east)`` through the
        affine."""
        (north, east), shape, scalar = _broadcast(north, east)
        dn = north - self.center_ne_m[0]
        de = east - self.center_ne_m[1]
        j = self.jacobian
        x = self.center_map[0] + j[0, 0] * dn + j[0, 1] * de
        y = self.center_map[1] + j[1, 0] * dn + j[1, 1] * de
        return _shaped(x, shape, scalar), _shaped(y, shape, scalar)


class NavFrame:
    """The navigation frame of one mission and its link to a map CRS.

    ``origin_lat``/``origin_lon``: the NED origin in decimal degrees, as
    recorded in ``NavSts.origin`` (any finite longitude; the conversion is
    periodic in it). ``map_crs``: anything pyproj accepts as a CRS (e.g.
    ``"EPSG:32631"``), projected, in metres, with east and north axes, and
    built on WGS 84 or ETRS89; ``None`` builds a frame for the geodetic
    conversions only, whose map methods then raise. The caller supplies the
    map CRS: this module holds no CRS of its own.

    The origin and the CRS are read-only. A frame pickles: it is rebuilt from
    its origin and CRS. Raises ``ValueError`` on a non-finite origin, a
    latitude outside (-90, 90), and an unsuitable CRS.
    """

    def __init__(self, origin_lat, origin_lon, map_crs=None):
        if not (_finite_real(origin_lat) and _finite_real(origin_lon)):
            raise ValueError(
                "NavFrame origin must be two finite numbers, got "
                f"({origin_lat!r}, {origin_lon!r})")
        if not -90.0 < origin_lat < 90.0:
            raise ValueError(
                f"NavFrame origin latitude {origin_lat!r} is outside (-90, 90)")
        self._origin_lat = float(origin_lat)
        self._origin_lon = float(origin_lon)
        crs = _normalize_map_crs(map_crs)
        self._map_crs = None if crs is None else crs.to_string()

        # IEEE remainder is exact: it brings the longitude into [-180, 180]
        # without changing a longitude already there.
        topo = (f"+proj=topocentric +ellps={_ELLIPSOID} "
                f"+lat_0={_number_text(self._origin_lat)} "
                f"+lon_0={_number_text(math.remainder(self._origin_lon, 360.0))}"
                " +h_0=0")
        try:
            self._topo_inverse = Transformer.from_pipeline(
                "+proj=pipeline "
                f"+step +inv {topo} "
                f"+step +inv +proj=cart +ellps={_ELLIPSOID} "
                "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
            self._topo_forward = Transformer.from_pipeline(
                "+proj=pipeline "
                "+step +proj=unitconvert +xy_in=deg +xy_out=rad "
                f"+step +proj=cart +ellps={_ELLIPSOID} "
                f"+step {topo}")
            self._map_transformer = None if crs is None else (
                Transformer.from_crs(_GEODETIC_CRS, crs, always_xy=True,
                                     allow_ballpark=False, only_best=True))
        except ProjError as exc:
            raise ValueError(
                f"cannot build the conversions of {self!r}: {exc}") from exc

    @classmethod
    def from_nav(cls, nav, map_crs=None):
        """The frame of a :class:`~bagio.nav_series.NavSeries`."""
        return cls(nav.origin_lat, nav.origin_lon, map_crs)

    @property
    def origin_lat(self):
        """Latitude of the NED origin, degrees."""
        return self._origin_lat

    @property
    def origin_lon(self):
        """Longitude of the NED origin, degrees, as given."""
        return self._origin_lon

    @property
    def map_crs(self):
        """The map CRS as pyproj's string (e.g. ``"EPSG:32631"``), or None."""
        return self._map_crs

    def __reduce__(self):
        return (type(self), (self._origin_lat, self._origin_lon, self._map_crs))

    def __repr__(self):
        return (f"NavFrame(origin_lat={self._origin_lat!r}, "
                f"origin_lon={self._origin_lon!r}, map_crs={self._map_crs!r})")

    # ----------------------------------------------------------- flat kernels

    def _require_map(self, caller):
        if self._map_transformer is None:
            raise ValueError(
                f"{caller}: this NavFrame was built without a map CRS; pass "
                "map_crs to use the map conversions")
        return self._map_transformer

    def _geodetic_of(self, north, east, caller):
        """(lat, lon) of finite flat nav arrays: the ellipsoid point (height 0)
        whose tangent-plane position is (north, east)."""
        up = np.zeros_like(north)
        for _ in range(_MAX_HEIGHT_STEPS):
            lon, lat, height = _transform(self._topo_inverse, east, north, up)
            _require_finite((lon, lat, height), caller)
            if np.max(np.abs(height), initial=0.0) <= _HEIGHT_TOL_M:
                return lat, lon
            # Along the origin's vertical, up and height change together to
            # first order (the verticals differ by distance / Earth radius).
            up = up - height
        raise ValueError(
            f"{caller}: the inverse did not converge; the points are farther "
            "than about 200 km from the frame's origin")

    def _nav_of(self, lat, lon, caller):
        """(north, east) of finite flat geodetic arrays, at height 0."""
        if np.any(np.abs(lat) > 90.0):
            raise ValueError(f"{caller}: latitude outside [-90, 90]")
        east, north, _ = _transform(self._topo_forward, lon, lat,
                                    np.zeros_like(lat))
        _require_finite((north, east), caller)
        return north, east

    def _map_of(self, north, east, caller):
        """(easting, northing) of finite flat nav arrays."""
        transformer = self._require_map(caller)
        lat, lon = self._geodetic_of(north, east, caller)
        x, y = _transform(transformer, lon, lat)
        _require_finite((x, y), caller)
        return x, y

    @staticmethod
    def _apply(kernel, a, b):
        """Run ``kernel`` on the finite points of ``a``/``b``; NaN elsewhere."""
        (a, b), shape, scalar = _broadcast(a, b)
        out1 = np.full(a.shape, np.nan)
        out2 = np.full(a.shape, np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            out1[ok], out2[ok] = kernel(a[ok], b[ok])
        return _shaped(out1, shape, scalar), _shaped(out2, shape, scalar)

    # ------------------------------------------------------------- geodetic

    def to_geodetic(self, north, east):
        """``(lat, lon)`` in degrees of nav ``(north, east)`` metres: the
        exact inverse of the vehicle's ``geodetic2Ned(lat, lon, 0)``."""
        return self._apply(
            lambda n, e: self._geodetic_of(n, e, "to_geodetic"), north, east)

    def from_geodetic(self, lat, lon):
        """``(north, east)`` metres of ``(lat, lon)`` degrees: the vehicle's
        ``geodetic2Ned(lat, lon, 0)``, horizontal part."""
        return self._apply(
            lambda la, lo: self._nav_of(la, lo, "from_geodetic"), lat, lon)

    # ------------------------------------------------------------------ map

    def to_map(self, north, east, offset_ne_m=None):
        """``(easting, northing)`` in the map CRS of nav ``(north, east)``.

        ``offset_ne_m``: a nav-frame translation ``(dn, de)`` — or a
        :class:`~bagio.nav_series.PlacementInfo` — added to the nav
        coordinates **before** converting, which is what a placement is.
        """
        self._require_map("to_map")
        dn, de = _offset(offset_ne_m)
        return self._apply(
            lambda n, e: self._map_of(n + dn, e + de, "to_map"), north, east)

    def from_map(self, easting, northing, offset_ne_m=None):
        """Nav ``(north, east)`` of map ``(easting, northing)``: the inverse
        of :meth:`to_map`, ``offset_ne_m`` subtracted after inverting."""
        transformer = self._require_map("from_map")
        dn, de = _offset(offset_ne_m)

        def kernel(x, y):
            lon, lat = _transform(transformer, x, y,
                                  direction=TransformDirection.INVERSE)
            _require_finite((lon, lat), "from_map")
            n, e = self._nav_of(lat, lon, "from_map")
            return n - dn, e - de

        return self._apply(kernel, easting, northing)

    def map_shift(self, offset_ne_m, at_ne_m):
        """``(d_easting, d_northing)``: what the nav-frame translation
        ``offset_ne_m`` becomes on the map at nav point ``at_ne_m``, i.e.
        ``to_map(at + offset) - to_map(at)``. For moving a finished raster
        rigidly. Exact at ``at_ne_m``; at another point p it differs from
        ``to_map(p + offset) - to_map(p)`` by the change of the local
        Jacobian between the two points times the offset — about 0.1 mm for
        a 3 m offset 300 m away at 39.5 N in UTM 31N.
        """
        dn, de = _offset(offset_ne_m)
        an, ae = _pair(at_ne_m, "at_ne_m")
        x1, y1 = self.to_map(an + dn, ae + de)
        x0, y0 = self.to_map(an, ae)
        return x1 - x0, y1 - y0

    # --------------------------------------------------------------- affines

    def _fit_affine(self, center, half_n, half_e, tol, caller):
        """The :class:`LocalAffine` over the rectangle of half-sides
        ``half_n`` (north) and ``half_e`` (east) around ``center``."""
        cn, ce = center
        dn, de = (g.reshape(-1) for g in np.meshgrid(
            np.linspace(-half_n, half_n, _AFFINE_SAMPLES),
            np.linspace(-half_e, half_e, _AFFINE_SAMPLES), indexing="ij"))
        x, y = self._map_of(cn + dn, ce + de, caller)
        xc, yc = self._map_of(np.array([cn]), np.array([ce]), caller)
        xc, yc = float(xc[0]), float(yc[0])
        design = np.column_stack([dn, de, np.ones_like(dn)])
        fit_x, *_ = np.linalg.lstsq(design, x - xc, rcond=None)
        fit_y, *_ = np.linalg.lstsq(design, y - yc, rcond=None)
        jacobian = np.array([[fit_x[0], fit_x[1]], [fit_y[0], fit_y[1]]])
        residual = float(np.max(np.hypot(
            x - (xc + jacobian[0, 0] * dn + jacobian[0, 1] * de),
            y - (yc + jacobian[1, 0] * dn + jacobian[1, 1] * de))))
        if not residual <= tol:
            raise ValueError(
                f"{caller}: the affine departs from the exact conversion by "
                f"{residual * 1e3:.3f} mm over +-{half_n} m north, "
                f"+-{half_e} m east around ({cn}, {ce}), above the "
                f"{tol * 1e3:.3f} mm limit; use a smaller extent or convert "
                "point by point")
        return LocalAffine(center_ne_m=(cn, ce), center_map=(xc, yc),
                           jacobian=jacobian, max_residual_m=residual)

    def local_affine(self, center_ne_m, half_extent_m, tol_m=1e-3):
        """The affine standing in for :meth:`to_map` over the square of
        half-side ``half_extent_m`` metres (positive, finite) around nav
        ``center_ne_m``.

        Returns a :class:`LocalAffine`. Raises ``ValueError`` when its
        ``max_residual_m`` against the exact conversion exceeds ``tol_m``
        (metres, default 1 mm).
        """
        self._require_map("local_affine")
        center = _pair(center_ne_m, "center_ne_m")
        half = _positive(half_extent_m, "half_extent_m")
        tol = _positive(tol_m, "tol_m")
        return self._fit_affine(center, half, half, tol, "local_affine")

    def raster_affine(self, nw_ne_m, pixel_m, shape, offset_ne_m=None,
                      tol_m=1e-3):
        """Geotransform of a raster gridded in the nav frame.

        The raster's rows run south and its columns east in the nav frame;
        ``nw_ne_m`` is the nav position of its north-west corner (the outer
        corner of pixel (0, 0)), ``pixel_m`` its square pixel side in nav
        metres, ``shape`` its ``(rows, cols)``. ``offset_ne_m`` is a nav-frame
        translation applied first (a placement). Returns the six coefficients
        ``(a, b, c, d, e, f)`` in ``affine.Affine`` order, mapping pixel-edge
        ``(col, row)`` to map ``easting = a*col + b*row + c``,
        ``northing = d*col + e*row + f``; they carry rotation terms. Fitted
        over the raster's footprint, with the residual check of
        :meth:`local_affine` (``tol_m``, metres, default 1 mm).
        """
        self._require_map("raster_affine")
        nw_n, nw_e = _pair(nw_ne_m, "nw_ne_m")
        pixel = _positive(pixel_m, "pixel_m")
        tol = _positive(tol_m, "tol_m")
        try:
            rows, cols = shape
        except (TypeError, ValueError):
            raise ValueError(
                f"shape must be (rows, cols), got {shape!r}") from None
        if not all(isinstance(v, numbers.Integral) and not isinstance(v, bool)
                   and v > 0 for v in (rows, cols)):
            raise ValueError(
                f"shape must hold two positive integers, got {shape!r}")
        dn, de = _offset(offset_ne_m)
        n0, e0 = nw_n + dn, nw_e + de
        half_h, half_w = rows * pixel / 2.0, cols * pixel / 2.0
        affine = self._fit_affine((n0 - half_h, e0 + half_w), half_h, half_w,
                                  tol, "raster_affine")
        j = affine.jacobian
        c, f = affine.apply(n0, e0)
        return (float(j[0, 1] * pixel), float(-j[0, 0] * pixel), float(c),
                float(j[1, 1] * pixel), float(-j[1, 0] * pixel), float(f))
