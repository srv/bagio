"""Transform tree built from a bag's ``/tf_static`` + ``/tf`` messages.

Replaces ``tf.TransformListener``. The original extractor looked up the
*latest* transform available at processing time (``rospy.Time(0)``), which is
timing-dependent under playback; this buffer instead resolves the chain at an
explicit timestamp, interpolating dynamic edges (linear translation, slerp
rotation). Deterministic and time-correct by construction.

TF semantics preserved: each child frame has exactly one parent, so the
transform graph is a forest; ``lookup(target, source, t)`` composes the two
root chains and returns ``T_target_source`` (4x4) such that
``p_target = T @ p_source``.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .typestore import stamp_to_ns


def _matrix(translation, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


class _DynamicEdge:
    """Time series of transforms for one (parent -> child) dynamic edge.

    Interpolation runs on nanosecond *offsets from the first sample*, not on
    absolute epoch nanoseconds: at 2026-epoch magnitudes (~1.8e18 ns) float64
    granularity is ~256 ns, so distinct absolute stamps could collapse to the
    same float and make ``Slerp`` raise on tied knots. Offsets stay exact in
    float64 for any realistic mission length.
    """

    def __init__(self):
        self.t_ns = []
        self.trans = []
        self.quats = []
        self._finalized = False

    def add(self, t_ns, translation, quat_xyzw):
        self.t_ns.append(t_ns)
        self.trans.append(translation)
        self.quats.append(quat_xyzw)
        self._finalized = False

    def _finalize(self):
        order = np.argsort(np.array(self.t_ns, dtype=np.int64), kind="stable")
        self._t = np.array(self.t_ns, dtype=np.int64)[order]
        self._trans = np.array(self.trans, dtype=np.float64)[order]
        quats = np.array(self.quats, dtype=np.float64)[order]
        # Drop duplicate timestamps (repeated-TF-data, cf. the geometry2 fork
        # throttling exactly this): keep the last occurrence.
        keep = np.append(self._t[1:] != self._t[:-1], True)
        self._t, self._trans, quats = self._t[keep], self._trans[keep], quats[keep]
        self._t0 = int(self._t[0])
        self._t_rel = (self._t - self._t0).astype(np.float64)
        self._rots = Rotation.from_quat(quats)
        self._slerp = Slerp(self._t_rel, self._rots) if len(self._t) > 1 else None
        self._finalized = True

    def at(self, t_ns, max_extrapolation_ns):
        if not self._finalized:
            self._finalize()
        t_ns = int(t_ns)
        if (t_ns < int(self._t[0]) - max_extrapolation_ns
                or t_ns > int(self._t[-1]) + max_extrapolation_ns):
            raise ValueError(
                f"TF extrapolation beyond {max_extrapolation_ns} ns: query "
                f"{t_ns} outside edge range [{self._t[0]}, {self._t[-1]}]"
            )
        t_rel = float(min(max(t_ns, int(self._t[0])), int(self._t[-1])) - self._t0)
        if self._slerp is None:
            rot = self._rots[0]
            trans = self._trans[0]
        else:
            rot = self._slerp(t_rel)
            trans = np.array([
                np.interp(t_rel, self._t_rel, self._trans[:, k])
                for k in range(3)
            ])
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = trans
        return T


class TfBuffer:
    """Static + dynamic transform storage with chained lookup."""

    def __init__(self):
        self._static = {}   # child -> (parent, 4x4)
        self._dynamic = {}  # child -> (parent, _DynamicEdge)

    def add_tf_message(self, msg, static=False):
        """Ingest one ``tf2_msgs/TFMessage``."""
        for tr in msg.transforms:
            parent = tr.header.frame_id.lstrip("/")
            child = tr.child_frame_id.lstrip("/")
            t = tr.transform.translation
            q = tr.transform.rotation
            translation = (t.x, t.y, t.z)
            quat = (q.x, q.y, q.z, q.w)
            if static:
                self._static[child] = (parent, _matrix(translation, quat))
            else:
                parent_known, edge = self._dynamic.get(child, (None, None))
                if edge is None:
                    edge = _DynamicEdge()
                    self._dynamic[child] = (parent, edge)
                elif parent_known != parent:
                    # A silent reparenting would corrupt every lookup before
                    # the change (history discarded) — unexpected on this
                    # vehicle rig, so fail loudly instead.
                    raise ValueError(
                        f"TF frame {child!r} reparented from {parent_known!r} "
                        f"to {parent!r} mid-recording; unsupported"
                    )
                edge.add(stamp_to_ns(tr.header.stamp), translation, quat)

    def frames(self):
        return sorted(set(self._static) | set(self._dynamic))

    def _parent_transform(self, child, t_ns, max_extrapolation_ns):
        """(parent, T_parent_child) or None if ``child`` is a root."""
        if child in self._dynamic:
            parent, edge = self._dynamic[child]
            return parent, edge.at(t_ns, max_extrapolation_ns)
        if child in self._static:
            parent, T = self._static[child]
            return parent, T
        return None

    def _chain_to_root(self, frame, t_ns, max_extrapolation_ns):
        """(root_name, T_root_frame) walking up the parent links."""
        T = np.eye(4)
        current = frame
        seen = set()
        while True:
            if current in seen:
                raise ValueError(f"TF loop detected at frame {current!r}")
            seen.add(current)
            up = self._parent_transform(current, t_ns, max_extrapolation_ns)
            if up is None:
                return current, T
            parent, T_parent_current = up
            T = T_parent_current @ T
            current = parent

    def lookup(self, target_frame, source_frame, t_ns,
               max_extrapolation_ns=500_000_000):
        """Return ``T_target_source`` (4x4) at time ``t_ns``.

        Equivalent to ``tf.TransformListener.lookupTransform(target, source)``
        but resolved at an explicit timestamp. Queries falling more than
        ``max_extrapolation_ns`` outside a dynamic edge's recorded range
        raise ``ValueError`` (the offline analogue of tf's
        ``ExtrapolationException``); within the tolerance the boundary
        sample is used.
        """
        target_frame = target_frame.lstrip("/")
        source_frame = source_frame.lstrip("/")
        root_t, T_root_target = self._chain_to_root(
            target_frame, t_ns, max_extrapolation_ns)
        root_s, T_root_source = self._chain_to_root(
            source_frame, t_ns, max_extrapolation_ns)
        if root_t != root_s:
            raise ValueError(
                f"Frames {target_frame!r} and {source_frame!r} are not "
                f"connected (roots {root_t!r} vs {root_s!r})"
            )
        return np.linalg.inv(T_root_target) @ T_root_source
