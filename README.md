# bagio — ROS1 bag access without ROS

A small Python 3.11 library for reading recorded ROS1 `.bag` files without
any ROS installation, built on the pure-Python `rosbags` library. It is the
shared bag-access layer of the AI4MAP offline pipelines
(`offline_mosaicing`, `offline_sonar_processing`).

Install (editable, from a checkout):

```bash
pip install -e .
```

## Modules

- **`typestore.py`** — message-type registration. Types are registered from
  each bag's **embedded message definitions** (`connection.msgdef.data`),
  not from `rosbags`' bundled stores: the bundled ROS1 store lacks some
  standard types (e.g. `tf2_msgs/TFMessage`) and knows nothing about
  `cola2_msgs`, and the embedded definitions stay correct even if a message
  definition changes between campaigns. `stamp_to_ns()` converts header
  stamps to integer nanoseconds.

- **`bag_reader.py`** — `BagSet` opens one or more bag parts
  (`*_0.bag`, `*_1.bag`, …) with a shared typestore and yields
  deserialized messages merged in timestamp order. `decode_image()`
  converts `sensor_msgs/Image` (bgr8/rgb8/mono8/bayer, row padding
  handled) and `CompressedImage` payloads to BGR numpy arrays.

- **`nav_series.py`** — the `cola2_msgs/NavSts` stream as sorted arrays
  with nearest-sample and interpolated access. Yaw interpolates on an
  unwrapped copy and re-wraps, so the ±π seam cannot produce a bogus
  midpoint. `median_filtered(field, window_s)` returns a centered
  sliding-median copy of a field (e.g. DVL altitude bottom-lock hop
  removal; refuses `yaw`), and `interp_values(t_ns, values)` linearly
  interpolates a precomputed per-sample array (e.g. the output of
  `median_filtered`) at an arbitrary timestamp.

- **`tf_buffer.py`** — transform tree from `/tf_static` + `/tf`.
  `lookup(target, source, t_ns)` composes the chain at an explicit
  timestamp, interpolating dynamic edges (linear translation + slerp
  rotation). Queries beyond `max_extrapolation_ns` (default 0.5 s) outside
  an edge's recorded range raise, mirroring tf's `ExtrapolationException`.
  A frame reparented mid-recording raises rather than silently discarding
  history.

## Numerical note

All time interpolation runs on nanosecond **offsets from the first sample**
of each series, never on absolute epoch nanoseconds: at 2026-epoch
magnitudes (~1.8e18 ns) float64 resolves only ~256 ns, which can collapse
distinct timestamps (and, e.g., make `scipy` `Slerp` reject tied knots).
Offsets are exact in float64 for any realistic mission length.
