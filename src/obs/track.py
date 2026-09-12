"""A track, fitted from a driven lap, and the observation every world shares.

This is the spine described in SPEC.md. A `Track` is just a smoothed centreline
with arc length along it — no Mario Kart in it anywhere — and `observe()` turns
(kart state, track) into the same fixed vector whether the kart is on an N64, in
Unreal, or on a street in London. Only the code that *produces* the kart state
differs per world.

The centreline comes from a lap that was searched out frame by frame
(src/mk64/ghost.py), so it is a line someone actually drove, not a racing line.
That is the right starting point: checkpoints every few metres along it give
dense reward without anyone having to describe the course.

Everything works in the XZ plane. Y is elevation and is carried for gradient but
never used for projection — tracks cross over themselves in 3D far less often
than they appear to.
"""

import json

import numpy as np

#: Roughly 33 world units per metre in MK64: a kart at full speed covers ~11
#: units per frame at 60 fps (~660 units/s) and moves at something like 20 m/s.
#: Only used to keep checkpoint spacing human-readable.
UNITS_PER_METRE = 33.0


def _resample(points, spacing):
    """Walk a polyline and drop a point every `spacing` units of arc length."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    want = np.arange(0.0, total, spacing)
    out = np.empty((len(want), 3))
    for axis in range(3):
        out[:, axis] = np.interp(want, s, pts[:, axis])
    return out, want, total


def _smooth(pts, window):
    """Moving average that wraps, because a lap is a loop."""
    if window < 3:
        return pts
    k = np.ones(window) / window
    out = np.empty_like(pts)
    for axis in range(pts.shape[1]):
        padded = np.concatenate([pts[-window:, axis], pts[:, axis], pts[:window, axis]])
        out[:, axis] = np.convolve(padded, k, mode="same")[window:-window]
    return out


class Track:
    """A closed centreline with arc length, tangents and curvature."""

    def __init__(self, centre, spacing, length):
        self.centre = centre                 # (N, 3)
        self.spacing = spacing
        self.length = length
        self.xz = centre[:, [0, 2]]
        d = np.roll(self.xz, -1, axis=0) - self.xz
        self.tangent = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        self.s = np.arange(len(centre)) * spacing
        # Signed curvature: how fast the tangent turns per unit of arc length.
        ang = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        dang = np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))
        self.curvature = dang / spacing

    # --- construction ----------------------------------------------------
    @classmethod
    def from_lap(cls, positions, spacing=150.0, smooth_window=9):
        pts, _, total = _resample(positions, spacing)
        pts = _smooth(pts, smooth_window)
        return cls(pts, spacing, total)

    @classmethod
    def from_ghost_file(cls, path, **kw):
        log = json.loads(open(path).read())
        return cls.from_lap([e["pos"] for e in log], **kw)

    @property
    def n_checkpoints(self):
        return len(self.centre)

    # --- queries ---------------------------------------------------------
    #: See KartSim.LOOK_BACK — a lap is a loop, so two parts of the track can be
    #: physically adjacent while being half a lap apart in arc length. A global
    #: nearest-point search flips between them and reports enormous phantom
    #: progress. Pass the previous index as `hint` to search only nearby.
    LOOK_BACK, LOOK_FWD = 3, 12

    def project(self, pos, hint=None):
        """Nearest point on the centreline: (index, arc length, signed offset).

        Offset is positive to the left of the direction of travel, so it has the
        same meaning on every track and in every world.
        """
        p = np.array([pos[0], pos[2]], dtype=float)
        if hint is None:
            d = self.xz - p
            i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        else:
            n = len(self.xz)
            idx = (hint + np.arange(-self.LOOK_BACK, self.LOOK_FWD + 1)) % n
            d = self.xz[idx] - p
            i = int(idx[np.argmin(np.einsum("ij,ij->i", d, d))])
        t = self.tangent[i]
        rel = p - self.xz[i]
        along = float(rel @ t)
        lateral = float(rel[0] * t[1] - rel[1] * t[0])   # 2D cross product
        return i, self.s[i] + along, -lateral

    def curvature_ahead(self, i, distances=(300.0, 600.0, 1200.0, 2400.0)):
        """Curvature a few fixed distances down the road — the corner preview."""
        out = []
        for dist in distances:
            j = (i + int(round(dist / self.spacing))) % len(self.centre)
            out.append(float(self.curvature[j]))
        return out

    def heading_error(self, i, yaw_vec):
        """Angle between where the kart points and where the track goes."""
        t = self.tangent[i]
        return float(np.arctan2(yaw_vec[0] * t[1] - yaw_vec[1] * t[0],
                                yaw_vec[0] * t[0] + yaw_vec[1] * t[1]))


#: Bump this whenever the vector's meaning changes. A policy file records the
#: version it was trained against and refuses to load against another, because
#: silently reinterpreting input 7 is not a failure anyone notices in time.
OBS_VERSION = 1

OBS_FIELDS = (
    "speed", "lateral_offset", "heading_sin", "heading_cos",
    "curv_300", "curv_600", "curv_1200", "curv_2400",
    "grade", "progress_delta",
)


def observe(state, track, prev_s=None, heading_vec=None, hint=None):
    """(kart state, track) -> the shared observation vector.

    Deliberately contains nothing a policy could use to recognise *which* track
    it is on: no absolute position, no lap number, no course id. A policy that
    cannot tell Luigi Raceway from a London roundabout is one that can be moved
    between them.

    Not yet present, and honestly missing rather than faked: the rangefinders
    and track width from SPEC.md. Both need the track *edges*, and a single
    driven line does not give you edges. They arrive with real geometry in
    stage 3, or from an explicit width probe, and OBS_VERSION goes up when
    they do.
    """
    if heading_vec is None:
        yaw = state["yaw"]
        heading_vec = (np.sin(yaw), np.cos(yaw))
    i, s, lateral = track.project(state["pos"], hint=hint)
    err = track.heading_error(i, heading_vec)
    c = track.curvature_ahead(i)
    j = (i + 8) % len(track.centre)
    grade = float(track.centre[j][1] - track.centre[i][1]) / (8 * track.spacing)
    ds = 0.0 if prev_s is None else float(s - prev_s)
    if ds < -track.length / 2:          # wrapped past the finish line
        ds += track.length
    elif ds > track.length / 2:
        ds -= track.length
    return np.array([
        float(state["speed"]) / 6.0,
        lateral / 300.0,
        np.sin(err), np.cos(err),
        c[0] * 1000.0, c[1] * 1000.0, c[2] * 1000.0, c[3] * 1000.0,
        grade * 10.0,
        ds / 12.0,
    ], dtype=np.float32), s, i
