"""An independent reference for the navigation-frame conversions.

A numpy port of ``cola2_lib``'s C++ ``NED`` class (``ned.cpp``, WGS84
constants from ``ned.h``): ``geodetic2Ned(lat, lon, 0)``, the conversion the
vehicle applies to produce ``NavSts`` north/east, inverted by fixed-point
iteration; and ``ned2Geodetic(n, e, 0)`` as cola2_lib itself implements it.
No PROJ in the forward conversion or its inverse, so it checks
:class:`bagio.NavFrame` from outside.
"""

import math

import numpy as np
from pyproj import Transformer

A = 6378137.0
ESQ = 6.69437999014e-3


def _ecef(lat_deg, lon_deg):
    """``NED::geodetic2Ecef`` at height 0, vectorized; shape (3, N)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    xi = np.sqrt(1.0 - ESQ * np.sin(lat) ** 2)
    return np.stack([A / xi * np.cos(lat) * np.cos(lon),
                     A / xi * np.cos(lat) * np.sin(lon),
                     A / xi * (1.0 - ESQ) * np.sin(lat)])


def _nre(lat_deg, lon_deg):
    """``NED::nRe``: the ECEF -> NED rotation (geodetic latitude)."""
    sl, cl = math.sin(math.radians(lat_deg)), math.cos(math.radians(lat_deg))
    so, co = math.sin(math.radians(lon_deg)), math.cos(math.radians(lon_deg))
    return np.array([[-sl * co, -sl * so, cl],
                     [-so, co, 0.0],
                     [cl * co, cl * so, sl]])


def radii(lat_deg):
    """Meridian and prime-vertical radii of curvature (m) at ``lat_deg``."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    return A * (1.0 - ESQ) / (1.0 - ESQ * s2) ** 1.5, A / np.sqrt(1.0 - ESQ * s2)


def cola2_geodetic2ned(lat0, lon0, lat, lon):
    """(north, east) of ``geodetic2Ned(lat, lon, 0)`` with origin (lat0, lon0)."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    d = _ecef(lat.ravel(), lon.ravel()) - _ecef(lat0, lon0)[:, None]
    ned = _nre(lat0, lon0) @ d
    return ned[0].reshape(lat.shape), ned[1].reshape(lat.shape)


def reference_geodetic(lat0, lon0, north, east):
    """(lat, lon) whose ``geodetic2Ned(lat, lon, 0)`` is (north, east)."""
    north, east = np.broadcast_arrays(np.asarray(north, float),
                                      np.asarray(east, float))
    m0, n0 = radii(lat0)
    lat = lat0 + np.degrees(north / m0)
    lon = lon0 + np.degrees(east / (n0 * math.cos(math.radians(lat0))))
    for _ in range(30):
        fn, fe = cola2_geodetic2ned(lat0, lon0, lat, lon)
        m, n = radii(lat)
        lat = lat - np.degrees((fn - north) / m)
        lon = lon - np.degrees((fe - east) / (n * np.cos(np.radians(lat))))
    fn, fe = cola2_geodetic2ned(lat0, lon0, lat, lon)
    assert np.max(np.hypot(fn - north, fe - east)) < 1e-8   # float floor
    return lat, lon


def cola2_ned2geodetic(lat0, lon0, north, east):
    """cola2_lib's own ``NED::ned2Geodetic(north, east, 0)``: the tangent-plane
    point, not the ellipsoid point (ECEF -> geodetic by pyproj)."""
    north, east = np.broadcast_arrays(np.asarray(north, float),
                                      np.asarray(east, float))
    ned = np.stack([north.ravel(), east.ravel(), np.zeros(north.size)])
    ecef = _nre(lat0, lon0).T @ ned + _ecef(lat0, lon0)[:, None]
    lon, lat, _ = Transformer.from_crs(
        "EPSG:4978", "EPSG:4979", always_xy=True).transform(*ecef)
    return lat.reshape(north.shape), lon.reshape(north.shape)


def geodetic_gap_m(lat1, lon1, lat2, lon2):
    """Horizontal distance (m) between two nearby geodetic points."""
    m, n = radii(lat1)
    return np.hypot(np.radians(lat2 - lat1) * m,
                    np.radians(lon2 - lon1) * n * np.cos(np.radians(lat1)))
