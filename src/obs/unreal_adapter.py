"""The Unreal side of the same ten-number contract.

Unreal sends raw state; this computes the observation. Do NOT compute the
observation vector in Unreal — the normalisations are easy to get subtly wrong
and nothing will look broken, it will just drive badly.

WHAT UNREAL SENDS
-----------------
Once, on connect:
    {"type": "track",
     "centreline": [[x,y,z], ...],   # driving order, closed loop, world units
     "half_width": 350.0,            # optional; else measured from edges
     "up_axis": "z",                 # "z" for Unreal, "y" for the N64 code
     "units_per_metre": 100.0}       # Unreal = 100 (1 uu = 1 cm)

Every frame:
    {"type": "state",
     "pos": [x,y,z],
     "vel": [x,y,z]}                 # world velocity; heading comes from this

This replies:
    {"steer": -1.0..1.0,"throttle": 0.0..1.0,"brake": 0.0..1.0}

WHY VELOCITY AND NOT A YAW ANGLE
--------------------------------
A vector needs no convention guessed about it. Handing over a heading angle
means agreeing on handedness, zero direction and sign, and getting any of those
backwards produces a car that turns the wrong way — which is exactly the bug
that cost this project a day on the N64 side. Send the velocity vector.

THE TWO SCALES, AND WHY THEY ARE SEPARATE
-----------------------------------------
`src/obs/track.observe` normalises by constants fitted in MK64's own units:
`speed / 6.0` against a top speed of 5.6 units per *tick*, `lateral / 300.0`
and `curvature * 1000` in MK64 length units, `ds / 12.0` against 11.2 units of
progress per *frame*. Feeding it Unreal's centimetres and seconds puts four of
the ten numbers far outside the range the policy ever saw, so two independent
conversions are needed, and conflating them is what made the first version of
this file quietly wrong:

* **length** — `ref_units_per_metre` MK64 units per real metre. Pick it so the
  track's proportions match what the policy trained on rather than by physical
  analogy: Luigi Raceway is 12,646 units round with a 144-unit road, a
  lap-to-width ratio of 87.8. King's Cross `_mk64` is 1,004 m round with a 12 m
  road — ratio 83.6, near enough the same circuit shape — so mapping its
  1,004 m onto 12,646 units gives 12.6 units/m and a half-width of 75 against
  MK64's 72. `LUIGI_LAP_UNITS` is here so that sum can be redone rather than
  trusted.

* **time** — MK64 runs 60 frames a second and two physics ticks a frame, so
  speed is per 1/120 s and progress per 1/60 s. One Unreal step is 100 ms. The
  first version of this adapter passed `cm/s * length_scale` straight in as
  `speed`, which is 120x the intended unit, and a `ds` accumulated over a
  100 ms step, which is 6x. Both are silent: the policy still steers, it just
  behaves as though it were doing 200 km/h everywhere.

  `ref_ticks_per_second` and `ref_frames_per_second` carry that conversion, and
  `seconds_per_step` says how much Unreal time one `observe()` covers.

THE ONE THING THAT WILL STILL BITE
----------------------------------
Handedness. Unreal is left-handed (X forward, Y right, Z up); this code works in
a right-handed ground plane. If `steer = +1` turns the car the wrong way, flip
`handedness` below — and confirm it with an open-loop replay
(scripts/calibrate_unreal.py), never by watching a policy fail. A policy that
fails can always be blamed on the policy.
"""

import numpy as np

from .track import Track, observe

#: Luigi Raceway's lap, in MK64 length units, measured off the centreline the
#: policy was trained against (tracks/luigi_raceway.json, 631 points from the
#: decomp's own CPU-racer path). The reference for choosing a length scale.
LUIGI_LAP_UNITS = 12646.0

#: MK64's time base: 60 frames a second, two physics ticks per frame.
MK64_FRAMES_PER_SECOND = 60.0
MK64_TICKS_PER_FRAME = 2


def units_per_metre_for_lap(lap_length_m, lap_units=LUIGI_LAP_UNITS):
    """Length scale that maps a lap of `lap_length_m` onto the reference lap.

    Use this rather than a physical constant. What the policy is sensitive to is
    how wide the road is relative to how far round the lap is, and how tight the
    corners are in the units its curvature normalisation expects — not how many
    centimetres a MK64 unit "really" is.
    """
    if lap_length_m <= 0:
        raise ValueError("lap length must be positive")
    return lap_units / float(lap_length_m)


class UnrealAdapter:
    """Turns Unreal's raw state into the shared observation vector."""

    def __init__(self, centreline, up_axis="z", units_per_metre=100.0,
                 half_width=None, handedness=+1, spacing_m=4.5,
                 ref_units_per_metre=33.0, seconds_per_step=0.1,
                 ref_ticks_per_second=MK64_FRAMES_PER_SECOND * MK64_TICKS_PER_FRAME,
                 ref_frames_per_second=MK64_FRAMES_PER_SECOND):
        self.up = up_axis.lower()
        self.handedness = handedness

        # Length: world units -> MK64 length units.
        self.scale = ref_units_per_metre / units_per_metre
        # Time: the two rates observe() was normalised against.
        self.seconds_per_step = float(seconds_per_step)
        self.ticks_per_second = float(ref_ticks_per_second)
        self.frames_per_step = self.seconds_per_step * float(ref_frames_per_second)

        pts = np.array([self._ground(p) for p in centreline], dtype=float) * self.scale
        spacing = spacing_m * ref_units_per_metre
        self.track = Track.from_lap(pts, spacing=spacing, smooth_window=5)
        self.half_width = (half_width * self.scale) if half_width else None
        self.prev_s = None
        self.hint = None

    def _ground(self, p):
        """Return (x, elevation, z) with the ground plane in x/z."""
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if self.up == "z":                 # Unreal: X,Y ground, Z up
            return [x * self.handedness, z, y]
        return [x * self.handedness, y, z]  # already Y-up

    def observe(self, pos, vel):
        gp = self._ground(pos)
        gv = self._ground(vel)

        # Velocity arrives per second; the policy's speed normalisation is per tick.
        speed_units_per_second = float(np.hypot(gv[0], gv[2])) * self.scale
        state = {
            "pos": [gp[0] * self.scale, gp[1] * self.scale, gp[2] * self.scale],
            # Direction only is what observe() takes from this, so it can stay per
            # second; the magnitude it uses comes from "speed" below.
            "velocity": [gv[0] * self.scale, gv[1] * self.scale, gv[2] * self.scale],
            "speed": speed_units_per_second / self.ticks_per_second,
            "yaw": 0.0,
        }

        # observe() differences s itself and normalises the result per frame, so hand
        # it a prev_s that makes one call look like one frame of progress.
        obs, s, i = observe(state, self.track, self.prev_s, hint=self.hint)
        if self.prev_s is not None and self.frames_per_step > 0:
            ds = s - self.prev_s
            if ds < -self.track.length / 2:
                ds += self.track.length
            elif ds > self.track.length / 2:
                ds -= self.track.length
            obs[9] = np.float32((ds / self.frames_per_step) / 12.0)
        self.prev_s, self.hint = s, i
        return obs

    def reset(self):
        self.prev_s = None
        self.hint = None
