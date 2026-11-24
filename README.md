# bagio — ROS1 bag access without ROS

A small Python 3.11 library for reading recorded ROS1 `.bag` files without
any ROS installation, built on the pure-Python `rosbags` library. It is the
shared bag-access layer of the AI4MAP offline pipelines, and their one
conversion between the vehicle's navigation frame and geodetic or map
coordinates (`nav_frame.py`).

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
  definition changes between campaigns. `build_typestore(readers)` returns a
  typestore able to decode every connection of open `rosbags.rosbag1.Reader`
  instances; `register_connection_types(ts, connections)` registers the
  embedded definitions of `connections` into an existing typestore.
  `stamp_to_ns()` converts header stamps to integer nanoseconds.
  `header_stamp_ns(msg, fallback_ns)` returns a message's header stamp in
  nanoseconds, or `fallback_ns` (typically the bag record time) when the
  stamp is unset (non-positive) or the message has no header.

- **`bag_reader.py`** — `BagSet` opens one or more bag parts
  (`*_0.bag`, `*_1.bag`, …) with a shared typestore and yields
  deserialized messages merged in timestamp order:
  `BagSet.messages(topic_suffixes=None, topics=None)` yields
  `(t_ns, topic, msg)`, where `t_ns` is the bag record time in nanoseconds
  (filtering by topic suffix or exact topic). `decode_image()`
  converts `sensor_msgs/Image` (bgr8/rgb8/mono8/bayer, row padding
  handled) and `CompressedImage` payloads to BGR numpy arrays.
  `bags_with_topics(bag_paths, topic_suffixes)` returns the subset of bags
  (input order kept) whose connection index carries any of the topic
  suffixes. It reads the index only, never the messages, so a mission's
  unrelated bags (e.g. camera bags) are not scanned. `topic_suffixes` must be
  a sequence of suffixes; a bare string raises `TypeError`.

- **`nav_series.py`** — the `cola2_msgs/NavSts` stream as sorted arrays
  with nearest-sample and interpolated access.
  `NavSeries.from_messages(messages)` builds the series from deserialized
  NavSts messages; access is `nearest(t_ns)` and `interp(t_ns, field)`;
  timestamps are the messages' header stamps (not the bag record time) —
  on the INS clock, as recorded — unless the series is built with an
  `InsClock` (see `ins_clock.py`), which moves them to the vehicle PC's
  clock. `NavSeries.from_records(records, clock=None)` takes the
  `(record_t_ns, NavSts)` pairs that `BagSet.messages` yields (topic
  dropped): without a clock it equals `from_messages`; with one, each stamp
  is corrected, then sorted and deduplicated as usual. Record times must be
  integer nanoseconds. `NavSeries.ins_clock` holds the provenance of the
  applied correction, or `None`; the constructor accepts only `None` or an
  applied provenance (`ValueError` otherwise, `TypeError` if it is not
  JSON-serialisable) and keeps its own copy. Repeated stamps are dropped,
  keeping the first. Yaw interpolates on
  an unwrapped copy and re-wraps, so the ±π seam cannot produce a bogus
  midpoint. `median_filtered(field, window_s)` returns a centered
  sliding-median copy of a field (e.g. DVL altitude bottom-lock hop
  removal; refuses `yaw`), and `interp_values(t_ns, values)` linearly
  interpolates a precomputed per-sample array (e.g. the output of
  `median_filtered`) at an arbitrary timestamp. A mission has one
  physical AUV trajectory; every sensor that rode it consumes one
  GPS-free correction solved from the camera pose-graph.
  `load_trajectory(path)` loads and validates a `trajectory_refined.json`
  correction artifact — checking `schema_version`, `status`, a
  `"relative"` frame, and a keyframe list matching `n_keyframes` — and
  raises `ValueError` if any check fails.
  `apply_trajectory_correction(nav, artifact, origin_tol_m=1.0)` returns a
  corrected copy of a `NavSeries`: the artifact's per-keyframe
  `delta_n`/`delta_e`/`delta_yaw` are interpolated at each sample's
  timestamp (linear between keyframes, held flat beyond the first/last)
  and added to `north`/`east`/`yaw` (yaw re-wrapped to (-π, π]); the
  artifact's NED origin must be within `origin_tol_m` metres of the nav
  origin, or it raises (guards against a mis-paired artifact). The gap is
  measured exactly in the nav frame (see `nav_frame.py`), not with a
  metres-per-degree approximation. `ValueError` is also raised when the
  artifact origin is not a pair of finite numbers, when the two origins
  cannot be compared (non-finite nav origin, latitude out of range), and
  when `origin_tol_m` is not a finite number >= 0.
  `NavSeries.with_correction(path, origin_tol_m=1.0)` is a convenience
  wrapper for the two calls. The artifact's keyframe stamps are image
  stamps, on the PC clock; the series' stamps are on that clock when it was
  built with an `InsClock` (or when the mission's INS was synchronised by
  its PPS pulse), and otherwise lag it by the INS clock error. Both sides
  must be in the same state, so the correction raises `ValueError` when the
  artifact's `ins_clock_correction` state differs from the series'
  (`ins_clock`; a missing key means not corrected). The corrected copy keeps
  the series' `ins_clock`.
  `load_placement(artifact)` reads the separate `placement` slot that the
  mosaicing pipeline writes into the same artifact — the per-mission rigid
  translation, measured at a surface park or taken from the recorded INS
  position, that places a mission's GPS-free (floating) products onto the
  ground — accepting either the dict `load_trajectory` returns or a path
  to the JSON. It returns a `PlacementInfo` with the `(north, east)`
  translation in metres (north first), the `method`/`refine_params_sha256`
  provenance, and `level` — what the translation was measured from, one of
  `PLACEMENT_LEVELS` = `("start_park", "end_park", "ins_as_recorded")`, or
  `None`. The level is derived from the slot: `method`
  `park_navavg_median` (`PLACEMENT_METHOD_PARK`) with `park.which`
  `"start"`/`"end"` gives `start_park`/`end_park`; `method`
  `ins_as_recorded` (`PLACEMENT_METHOD_INS`) gives `ins_as_recorded`;
  anything else gives `None`. The level is only reported, never a reason to
  refuse a slot, and constructing a `PlacementInfo` with a `level` outside
  `PLACEMENT_LEVELS` (other than `None`) raises `ValueError`. The placement
  is translation-only (no rotation) and is expressed in the **navigation
  frame**, not on a map's axes: apply it before converting to a map CRS
  (`NavFrame.to_map(..., offset_ne_m=placement)`). `load_placement` raises
  `ValueError` if the slot is missing, not yet computed, or fails
  validation, including when the slot's `ins_clock_correction` state
  differs from the artifact header's.

- **`tf_buffer.py`** — transform tree from `/tf_static` + `/tf`. A `TfBuffer`
  is filled message by message with
  `add_tf_message(msg, static=False, record_t_ns=None)`;
  `lookup(target, source, t_ns)` returns the 4x4 `T_target_source`
  (`p_target = T @ p_source`), composing the chain at an explicit
  timestamp and interpolating dynamic edges (linear translation + slerp
  rotation). Queries beyond `max_extrapolation_ns` (default 0.5 s) outside
  an edge's recorded range raise, mirroring tf's `ExtrapolationException`.
  A frame reparented mid-recording raises rather than silently discarding
  history. `TfBuffer(clock=None, clock_child_suffixes=TF_CHILD_SUFFIXES)`
  takes an optional `InsClock`: the navigator's dynamic edge to the vehicle
  frame (child `<robot>/base_link`) is stamped on the INS clock, and with a
  clock the dynamic edges whose child frame ends with a listed suffix take
  corrected stamps. `record_t_ns` (the message's bag record time, integer
  nanoseconds) is then required for every dynamic message; static and
  unlisted edges are unchanged. `clock_child_suffixes` must be a sequence of
  suffixes; a bare string raises `TypeError`.

- **`captain_log.py`** — the mission captain node's log on `/rosout`. Only
  messages from a node whose name ends in `/captain` count, so the vehicle
  namespace is not hardcoded. Stamps are the messages' header stamps, falling
  back to the bag record time when a stamp is unset. Warnings are passed to
  the `log` callable (default `print`), prefixed `[bagio]`.
  `park_intervals(bag_paths, log=print)` returns the surface parks as a list
  of `(t_start_ns, t_end_ns, declared_s)`, each a matched
  `Executing mission park wait for N seconds` /
  `Mission park wait finalized` pair; `declared_s` is the announced duration
  in seconds, or `None` when the start line gives none. The logged pair, not
  the announced duration, delimits the interval. A start followed by another
  start before any finalize is discarded with a warning, as is a finalize not
  later than its start. The result is validated to be strictly ascending and
  non-overlapping.
  `trigger_calls(bag_paths, log=print)` returns `(t_ns, service)` for every
  `Calling trigger service with id <service>` line, `service` being the full
  service id as logged (match it by suffix), sorted by stamp (stable for
  equal stamps); a line without a single whitespace-free id is skipped with a
  warning. `CaptainLogError` (a `RuntimeError`) is raised by both functions
  when no bag carries `/rosout`, and by `park_intervals` when the intervals
  overlap or are out of order. The matched texts are module constants
  (`ROSOUT_SUFFIX`, `CAPTAIN_SUFFIX`, `PARK_START_PREFIX`, `PARK_END_PREFIX`,
  `PARK_DURATION_RE`, `TRIGGER_PREFIX`).

- **`surface_gps.py`** — the mission navigation together with the surface
  GPS fixes. The vehicle is geolocalized only at the surface: the INS driver
  republishes each accepted fix of the GPS receiver on `navigator/gps_ned`
  already converted into the mission's NED frame (the frame of
  `navigator/navigation`), with the receiver's own stamps, and the
  recorded navigation is the INS solution, conditioned by GPS while the
  vehicle is at the surface and purely inertial once it submerges.
  `load_mission_nav(bag_paths, nav_topic_suffix="navigator/navigation",
  gps_topic_suffix="navigator/gps_ned", clock=None)` makes one pass over the
  bags and returns `(nav, gps_t_ns, gps_ne)`: a `NavSeries`, the fixes'
  header stamps (int64 nanoseconds) and their north/east positions (Nx2
  float64), both time-sorted; the fix arrays are empty when the bags hold no
  fix, and `ValueError` is raised when they hold no navigation message.
  With an `InsClock` as `clock` the navigation stamps are corrected (as in
  `NavSeries.from_records`); the fixes' stamps are the receiver's
  `navigator/gps` stamps, already on the PC clock, and are never corrected.
  `gps_quality_reader(bag_paths)` reads the receiver's own view of the fixes
  from `navigator/gps` (`sensor_msgs/NavSatFix`) once and returns a callable
  `(t0_ns, t1_ns) -> dict | None` that summarizes the samples in the
  inclusive window: `cov_xx_mean`/`cov_xx_min`/`cov_xx_max` (the `xx`
  position covariance, rounded to 6 digits), the sample count `n`, and the
  sorted set of fix `status` values; `None` when the window holds no sample.
  NO_FIX samples (status < 0) and non-finite covariances are skipped. The
  covariance is corroborating evidence only: it is coarsely quantized and
  spans a far smaller range than the actual fix errors, so it suits a record
  rather than the selection of fixes. The default topic suffixes are module
  constants (`NAV_SUFFIX`, `GPS_NED_SUFFIX`, `GPS_STATUS_SUFFIX`).

- **`nav_frame.py`** — exact conversion between the navigation frame and a
  projected map CRS, through `pyproj`. The navigation frame is the
  `north`/`east` metres of `NavSts` (and `navigator/gps_ned`) relative to the
  mission's NED origin; COLA2 computes them as `cola2_lib`'s
  `NED::geodetic2Ned(lat, lon, 0)`: the point at height 0 on the WGS 84
  ellipsoid, expressed in the plane tangent to the ellipsoid at the origin.
  `NavFrame(origin_lat, origin_lon, map_crs=None)` implements that
  conversion, its exact inverse, and the link to `map_crs`, which must be
  projected, in metres, with east/north axes, and built on WGS 84 or ETRS89
  (any other datum needs a transformation the module does not choose);
  `None` allows the geodetic conversions only, and the map methods raise.
  `NavFrame.from_nav(nav, map_crs=None)` builds the frame of a `NavSeries`.
  `origin_lat`, `origin_lon` and `map_crs` are read-only properties; any
  finite longitude is accepted, and a frame pickles.
  `to_geodetic(north, east)` returns `(lat, lon)` and is the exact inverse of
  the vehicle's conversion. (`cola2_lib`'s own `ned2Geodetic(n, e, 0)`
  instead returns the point of the tangent plane, which lies above the
  ellipsoid and departs from that inverse by about d³/2R² for a distance d
  from the origin and Earth radius R: about 3.5 cm at 14 km.)
  `from_geodetic(lat, lon)` returns `(north, east)`.
  `to_map(north, east, offset_ne_m=None)` returns `(easting, northing)`, and
  `from_map(easting, northing, offset_ne_m=None)` is its inverse.
  `offset_ne_m` is a **nav-frame** translation `(dn, de)` (a `PlacementInfo`
  is accepted) applied **before** converting, which is how a placement is
  applied. Adding nav metres to map axes instead ignores the grid
  convergence and the map's point scale: a rotation (−0.396° at 39.5 N,
  2.4 E in UTM 31N) and a scale (0.99964 there) about the origin, together
  about 0.7 % of the distance from the origin.
  `map_shift(offset_ne_m, at_ne_m)` returns the `(d_easting, d_northing)`
  that a nav-frame translation becomes on the map at a given nav point, for
  moving a finished raster rigidly; it is exact at that point only.
  `local_affine(center_ne_m, half_extent_m, tol_m=1e-3)` returns a
  `LocalAffine` standing in for `to_map` over a square: `center_ne_m`,
  `center_map` (exact at the centre), `jacobian` (read-only 2x2, rows
  easting/northing, columns north/east), `max_residual_m` (largest distance
  to the exact conversion over a 101 x 101 sample of the square) and
  `apply(north, east)`. `raster_affine(nw_ne_m, pixel_m, shape,
  offset_ne_m=None, tol_m=1e-3)` returns the six coefficients, in
  `affine.Affine` order, of a raster gridded in the nav frame (rows run
  south, columns east; `nw_ne_m` is the outer corner of pixel (0, 0)),
  rotation terms included. Both affines raise `ValueError` when their
  residual (over the square, or over the raster's footprint) exceeds
  `tol_m`.
  Conventions: nav inputs are **north first**; map outputs are
  **(easting, northing)**. Inputs are scalars or array-likes that broadcast
  together; scalars give floats, arrays give float64 arrays of the broadcast
  shape. A non-finite input gives NaN for that point; a finite input that
  cannot be converted raises `ValueError`.
  Accuracy: point conversions agree with an independent implementation of
  the vehicle's conversion to within 1 micrometre within 10 km of the
  origin, and the inverse is supported up to about 200 km. The affine
  residual is reported and capped (default 1 mm); at 39.5 N in UTM 31N it is
  about 0.12 mm over ±300 m and 0.33 mm over ±500 m.

- **`ins_clock.py`** — per-message correction of the INS clock in the
  navigation stamps. The INS stamps `navigator/navigation`, the navigator's
  `/tf` edge to the vehicle frame (child `<robot>/base_link`), its raw data
  and its republications of the same message (odometry, throttled copies)
  with its own clock; the cameras, sonars, DVL, pressure and GPS are stamped
  on the vehicle PC's clock. Without a PPS pulse the INS clock drifts from
  the PC clock by tens to about 250 ms within a mission and can step by about
  0.1 s, so the navigation stamps sit behind every sensor stamp by that
  error. The bag record time is on the PC clock, so for each navigation
  message `lag = record time - stamp = INS clock error + transport`
  (transport about 3.8 ms).

  The correction is measured on the main recorder's navigation (the
  recording that carries both `navigator/navigation` and `/rosout`):
  `lag_f` is the median of the lag over the 2m+1 messages centred on each
  one, with m the messages in half of `WINDOW_S` (5 s; m = 50 at 20 Hz), and
  `corrected stamp = header stamp + (lag_f - TRANSPORT_NS)`
  (`TRANSPORT_NS` = 3.8 ms). The window is counted in messages, so a clock
  step stays at its own message; near the ends of the recording it shrinks
  symmetrically, so the very first and last messages are not filtered.
  Recorder stalls shorter than half the window are removed.

  `InsClock` is built with `InsClock.from_mission(bag_paths)` (picks the
  main bag among a mission's bags with `main_bags`),
  `InsClock.from_bags(bag_paths)` (the parts of one recording) or
  `InsClock.from_records(records)` (integer-ns `(record_t_ns, stamp_ns)`
  pairs of one recorder's navigation, in any order); all take
  `transport_ns`, `window_s` and `source` (names of the bags, for the
  provenance), and the bag builders take `nav_topic_suffix`. Read-only
  arrays: `record_t_ns`, `stamp_ns`, `lag_ns`, `lag_filtered_ns`,
  `corrections_ns`; also `half_window_messages` and `repeated_stamps`.
  `main_bags(bag_paths, nav_topic_suffix=...)` returns the main bag's parts;
  a part named `…_0.bag`, or a bag without a split suffix, starts a
  recording, and it raises `ValueError` when no bag carries both topics or
  they belong to more than one recording.

  `clock.correction_ns(stamp_ns, record_t_ns, max_hold_s=MAX_HOLD_S)` and
  `clock.correct(...)` (stamp plus correction) are vectorised over integer
  nanoseconds and serve messages recorded anywhere: a stamp carried by
  exactly one main-bag message takes its correction; a stamp carried by
  several (a clock step of a whole number of message periods re-issues
  stamps) takes the correction of the one nearest in record time, and is
  refused when none is clearly nearest; any other stamp takes the correction
  of the main-bag message nearest in record time, refused beyond
  `max_hold_s` (most likely a clock from another recording). Corrections are
  never interpolated over stamps, which step back where the INS clock does.

  Using a clock in a consumer:

  ```python
  import json

  from bagio import (INS_CLOCK_TAG, BagSet, InsClock, NavSeries, TfBuffer,
                     ins_clock_applied, ins_clock_tag_text, load_mission_nav)
  from bagio.ins_clock import NOT_APPLIED, PROVENANCE_KEY

  clock = InsClock.from_mission(bag_paths)

  nav, gps_t_ns, gps_ne = load_mission_nav(bag_paths, clock=clock)

  # Readers that yield (t_ns, topic, msg), as BagSet.messages does:
  with BagSet(bag_paths) as bags:
      nav = NavSeries.from_records(
          ((t, m) for t, _, m in bags.messages(topic_suffixes=("navigator/navigation",))),
          clock)
  tf = TfBuffer(clock)
  # tf.add_tf_message(msg, static=False, record_t_ns=t_ns)  # t_ns required

  meta = {PROVENANCE_KEY: clock.provenance()}   # clock applied
  meta = {PROVENANCE_KEY: dict(NOT_APPLIED)}    # clock not applied

  # A raster that records the state in its own tags:
  tags = {INS_CLOCK_TAG: ins_clock_tag_text(clock.provenance())}
  ins_clock_applied({PROVENANCE_KEY: json.loads(tags[INS_CLOCK_TAG])})  # True
  ```

  Record times must be **integer nanoseconds**: floats are refused
  (`TypeError`), not truncated. A reader that yields float seconds must be
  converted first, e.g. `int(round(t * 1e9))`. The readers apply a clock
  only to `navigator/navigation` and the `base_link` edge; another stream on
  the INS clock (the navigator's odometry, its raw data) is corrected with
  `clock.correct(stamp_ns, record_t_ns)`. The `navigator/gps_ned` fixes
  carry the GPS receiver's stamps on the PC clock and are never corrected.

  `clock.provenance()` returns a JSON-ready dict with `applied`, `method`,
  `transport_ms`, `window_s`, `window_messages`, `source`, `bagio_version`,
  `messages`, `correction_ms` (`min`, `p50`, `max`), `steps` (jumps over
  10 ms between consecutive filtered lags; a long stall shows as a pair of
  opposite steps), `residual` (`max_abs_ms`, `count_above_threshold`,
  `threshold_ms`), `end_departure_ms` (`start`, `end`), `repeated_stamps` and
  `gaps` (record-time gaps over 1 s). A producer stores it under
  `PROVENANCE_KEY` (`"ins_clock_correction"`), or stores `dict(NOT_APPLIED)`
  (`{"applied": False}`; `NOT_APPLIED` itself is a read-only mapping that
  `json` cannot serialise). `ins_clock_applied(record)` reads the state back
  from a product's metadata, an artifact or a placement slot: `False` when
  the key is absent, `ValueError` when the block is malformed, `TypeError`
  when `record` is not a mapping. The trajectory artifact and the placement
  slot must agree on the state (see `nav_series.py`).
  A raster that records the state in its own metadata holds the block in the
  GeoTIFF tag named `INS_CLOCK_TAG` (`"INS_CLOCK_CORRECTION"`);
  `ins_clock_tag_text(block)` gives the tag's text for `clock.provenance()` or
  `dict(NOT_APPLIED)`: canonical JSON (ASCII, sorted keys, no whitespace
  between tokens), so it does not depend on key order. It does not check the
  block, and raises `TypeError` for anything `json` cannot serialise
  (`NOT_APPLIED` itself included) and `ValueError` for a NaN or an infinity.
  Read the tag back with `ins_clock_applied({PROVENANCE_KEY:
  json.loads(text)})`, never `ins_clock_applied(tags)`: a raster's tags do not
  hold `PROVENANCE_KEY`, so that would return `False` for a corrected raster,
  without an error.

  Refusals: `TypeError` on float times; `ValueError` on fewer than two
  messages, anything but pairs, an unusable window, a stamp <= 0, a lag
  beyond 10 s (`MAX_LAG_NS`; stamps or times in other units, or mispaired),
  the navigation of more than one recorder (more than 1 % repeated stamps,
  `MAX_REPEATED_SHARE`), and arrays not in ascending record-time order given
  to the constructor. The constants (`TRANSPORT_NS`, `WINDOW_S`,
  `PROVENANCE_KEY`, `NOT_APPLIED`, `TF_CHILD_SUFFIXES`, `STEP_MS`,
  `RESIDUAL_MS`, `MAX_REPEATED_SHARE`, `MAX_LAG_NS`, `MAX_HOLD_S`, `GAP_S`)
  are imported from `bagio.ins_clock`; of this module, the package exports
  `InsClock`, `ins_clock_applied`, `ins_clock_tag_text`, `main_bags` and one
  constant, `INS_CLOCK_TAG`.

- **`ins_clock_survey.py`** — command line:
  `python -m bagio.ins_clock_survey <mission folder or bags>...` prints each
  mission's clock provenance as JSON. Each folder argument is a mission (its
  `*.bag`); the bag-file arguments together form one more. A mission that
  fails is reported with its error and the others continue; the exit status
  is 1 if any failed. It is its own module because the package imports
  `bagio.ins_clock`, and running that one with `-m` would load it twice.

## Numerical note

All time interpolation runs on nanosecond **offsets from the first sample**
of each series, never on absolute epoch nanoseconds: at 2026-epoch
magnitudes (~1.8e18 ns) float64 resolves only ~256 ns, which can collapse
distinct timestamps (and, e.g., make `scipy` `Slerp` reject tied knots).
Offsets are exact in float64 for any realistic mission length.
