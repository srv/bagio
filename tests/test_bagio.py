"""bagio tests against a real mission recording.

These tests read (never write) the validation mission's bags. They are
skipped when the dataset is not mounted, so the suite stays runnable on any
checkout; full runs happen inside the py311 container where
``/home/rosuser/dataset`` exists. Override the mission with the
``OM_TEST_MISSION`` environment variable.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bagio import (
    BagSet,
    NavSeries,
    TfBuffer,
    decode_image,
    stamp_to_ns,
)

MISSION = Path(os.environ.get(
    "OM_TEST_MISSION",
    "/home/rosuser/dataset/2026/orat/Andratx/2026_07_02/13_52_22",
))

pytestmark = pytest.mark.skipif(
    not MISSION.is_dir(), reason=f"validation mission not mounted at {MISSION}"
)

# Ground truth from the production ROS run of this mission
# (mosaic/homographies/HM/1.yaml) — the first accepted keyframe.
GOLDEN_STAMP_NS = 1782993228175963648
GOLDEN_NAV_X = -27.564414250376863
GOLDEN_NAV_Y = -0.24980336217132498
GOLDEN_RPY = (0.003875457426072775, 0.09724902186866097, -1.7762994835162433)
GOLDEN_ALT = 3.4179458618164062
# The ROS run paired each image with the *latest* TF at the moment its 2 Hz
# timer fired — up to ~0.5 s away from the image stamp — so the golden pose
# corresponds to some instant near the stamp, not to the stamp itself. The
# pose test therefore checks (a) coarse agreement at the stamp and (b) tight
# agreement somewhere within the timer window.
TOL_POS_M = 0.5
TOL_ANG_RAD = 0.3
WINDOW_S = 0.6
TOL_ANG_IN_WINDOW = 0.02


@pytest.fixture(scope="module")
def camera_bagset():
    paths = sorted(MISSION.glob("sparus2_stereo_camera_*.bag"))
    assert paths, "no stereo camera bags in mission folder"
    with BagSet(paths) as bags:
        yield bags


@pytest.fixture(scope="module")
def nav(camera_bagset):
    msgs = (m for _, _, m in camera_bagset.messages(
        topic_suffixes=("navigator/navigation",)))
    return NavSeries.from_messages(msgs)


@pytest.fixture(scope="module")
def tf_buffer(camera_bagset):
    buf = TfBuffer()
    for _, topic, msg in camera_bagset.messages(topics=("/tf", "/tf_static")):
        buf.add_tf_message(msg, static=(topic == "/tf_static"))
    return buf


def test_messages_are_time_ordered(camera_bagset):
    last = -1
    for t_ns, _, _ in camera_bagset.messages(
            topic_suffixes=("port/camera_info",)):
        assert t_ns >= last
        last = t_ns
    assert last > 0


def test_message_count_matches_connections(camera_bagset):
    suffix = ("port/camera_info",)
    counted = camera_bagset.message_count(topic_suffixes=suffix)
    iterated = sum(1 for _ in camera_bagset.messages(topic_suffixes=suffix))
    assert counted == iterated > 0


def test_compressed_image_decodes(camera_bagset):
    for _, _, msg in camera_bagset.messages(
            topic_suffixes=("port/image_raw/compressed",)):
        img = decode_image(msg, "sensor_msgs/msg/CompressedImage")
        assert img.shape == (1464, 1936, 3)
        assert img.dtype == np.uint8
        break


def test_nav_series_basics(nav):
    assert len(nav) > 10_000
    assert np.all(np.diff(nav.t_ns) >= 0)
    assert nav.origin_lat == pytest.approx(39.5437387, abs=1e-6)
    assert nav.origin_lon == pytest.approx(2.3779181, abs=1e-6)


def test_nav_nearest_matches_golden_altitude(nav):
    sample = nav.nearest(GOLDEN_STAMP_NS)
    assert sample.altitude == pytest.approx(GOLDEN_ALT, abs=0.15)


def test_nav_yaw_interpolation_wraps(nav):
    yaw = nav.interp(int(nav.t_ns[len(nav) // 2]), "yaw")
    assert -np.pi <= yaw <= np.pi


def test_tf_static_chain_present(tf_buffer):
    frames = tf_buffer.frames()
    assert "sparus2/camera_port" in frames
    assert "sparus2/base_link" in frames


def test_tf_lookup_matches_golden_pose(tf_buffer):
    T = tf_buffer.lookup("world_ned", "sparus2/camera_port", GOLDEN_STAMP_NS)
    # Rotation block must be orthonormal.
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    # Position vs the production run's NAV_X/NAV_Y.
    assert T[0, 3] == pytest.approx(GOLDEN_NAV_X, abs=TOL_POS_M)
    assert T[1, 3] == pytest.approx(GOLDEN_NAV_Y, abs=TOL_POS_M)
    # Euler decomposition (sxyz, as the extractor uses) vs golden rpy —
    # coarse at the stamp itself (timer-window skew is expected)...
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = Rotation.from_matrix(R).as_euler("xyz")
    for got, want in zip((roll, pitch, yaw), GOLDEN_RPY):
        assert got == pytest.approx(want, abs=TOL_ANG_RAD)
    # ...and tight somewhere inside the timer window: the golden pose must
    # exist on our reconstructed trajectory near the stamp.
    offsets_ns = (np.arange(-12, 13) * 50_000_000).astype(np.int64)
    best = np.inf
    for off in offsets_ns:
        Tw = tf_buffer.lookup(
            "world_ned", "sparus2/camera_port", GOLDEN_STAMP_NS + int(off))
        yw = Rotation.from_matrix(Tw[:3, :3]).as_euler("xyz")[2]
        best = min(best, abs(float(yw) - GOLDEN_RPY[2]))
    assert best < TOL_ANG_IN_WINDOW, (
        f"golden yaw not found within ±{WINDOW_S}s of the stamp "
        f"(best |Δyaw|={best:.4f} rad)")


def test_tf_lookup_is_deterministic(tf_buffer):
    T1 = tf_buffer.lookup("world_ned", "sparus2/camera_port", GOLDEN_STAMP_NS)
    T2 = tf_buffer.lookup("world_ned", "sparus2/camera_port", GOLDEN_STAMP_NS)
    assert np.array_equal(T1, T2)


def test_stamp_to_ns_roundtrip():
    class Stamp:
        sec = 1782993228
        nanosec = 175963648

    assert stamp_to_ns(Stamp()) == GOLDEN_STAMP_NS


# --- synthetic TfBuffer tests (no dataset needed) -------------------------

class _Stamp:
    def __init__(self, ns):
        self.sec, self.nanosec = divmod(ns, 1_000_000_000)


class _V:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=None):
        self.x, self.y, self.z = x, y, z
        if w is not None:
            self.w = w


class _Tf:
    def __init__(self, parent, child, t_ns, xyz=(0, 0, 0), quat=(0, 0, 0, 1)):
        self.header = type("H", (), {})()
        self.header.frame_id = parent
        self.header.stamp = _Stamp(t_ns)
        self.child_frame_id = child
        self.transform = type("T", (), {})()
        self.transform.translation = _V(*xyz)
        self.transform.rotation = _V(*quat[:3], w=quat[3])


class _TfMsg:
    def __init__(self, transforms):
        self.transforms = transforms


BASE_NS = 1_782_993_228_000_000_000  # 2026-epoch magnitude on purpose


@pytest.mark.parametrize("dt_ns", [100, 1])
def test_tf_subfloat_spacing_does_not_crash(dt_ns):
    # Distinct int64 stamps closer than float64 granularity at 2026-epoch
    # magnitude (~256 ns) must not produce tied Slerp knots.
    buf = TfBuffer()
    for k in range(3):
        buf.add_tf_message(_TfMsg([_Tf(
            "world", "base", BASE_NS + k * dt_ns, xyz=(float(k), 0, 0))]))
    T = buf.lookup("world", "base", BASE_NS + dt_ns)
    assert np.isfinite(T).all()
    assert T[0, 3] == pytest.approx(1.0, abs=1e-6)


def test_tf_extrapolation_raises_beyond_tolerance():
    buf = TfBuffer()
    for k in range(2):
        buf.add_tf_message(_TfMsg([_Tf(
            "world", "base", BASE_NS + k * 1_000_000_000)]))
    # Within tolerance: clamps to the boundary sample.
    buf.lookup("world", "base", BASE_NS - 400_000_000)
    with pytest.raises(ValueError, match="extrapolation"):
        buf.lookup("world", "base", BASE_NS - 600_000_000)


def test_tf_reparenting_raises():
    buf = TfBuffer()
    buf.add_tf_message(_TfMsg([_Tf("world", "base", BASE_NS)]))
    with pytest.raises(ValueError, match="reparented"):
        buf.add_tf_message(_TfMsg([_Tf("odom", "base", BASE_NS + 1)]))
