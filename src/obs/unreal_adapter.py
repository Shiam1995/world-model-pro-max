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
    {"steer": -1.0..1.0, "throttle": 0.0..1.0, "brake": 0.0..1.0}

WHY VELOCITY AND NOT A YAW ANGLE
--------------------------------
A vector needs no convention guessed about it. Handing over a heading angle
means agreeing on handedness, zero direction and sign, and getting any of those
backwards produces a car that turns the wrong way — which is exactly the bug
that cost this project a day on the N64 side. Send the velocity vector.

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


class UnrealAdapter:
    """Turns Unreal's raw state into the shared observation vector."""

    def __init__(self, centreline, up_axis="z", units_per_metre=100.0,
                 half_width=None, handedness=+1, spacing_m=4.5, ref_units_per_metre=33.0):
        self.up = up_axis.lower()
        self.handedness = handedness
        # The policy's normalisations were fitted in MK64 units (~33 per metre).
        # Rather than rescale the policy, rescale the world into the units it was
        # trained in — or, better, retrain in the sim on this track: it takes
        # under two minutes and removes the question entirely.
        self.scale = ref_units_per_metre / units_per_metre
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
        state = {
            "pos": [gp[0] * self.scale, gp[1] * self.scale, gp[2] * self.scale],
            "velocity": [gv[0] * self.scale, gv[1] * self.scale, gv[2] * self.scale],
            "speed": float(np.hypot(gv[0], gv[2]) * self.scale),
            "yaw": 0.0,
        }
        obs, s, i = observe(state, self.track, self.prev_s, hint=self.hint)
        self.prev_s, self.hint = s, i
        return obs

    def reset(self):
        self.prev_s = None
        self.hint = None
