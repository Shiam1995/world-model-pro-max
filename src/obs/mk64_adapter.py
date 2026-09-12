"""The MK64 adapter: the only file that knows both Mario Kart and the vector.

Everything above this line is game-agnostic (src/obs/track.py), everything below
it is emulator-specific (src/mk64/). Unreal and London get their own adapter of
this shape and nothing else changes — that is the whole point of SPEC.md's spine.
"""

import json

import numpy as np

from ..mk64 import player
from .track import Track, observe


class MK64Adapter:
    def __init__(self, env, track_file="tracks/luigi_raceway.json",
                 spacing=150.0, smooth_window=5):
        d = json.loads(open(track_file).read())
        self.track = Track.from_lap(np.array(d["points"], dtype=float),
                                    spacing=spacing, smooth_window=smooth_window)
        self.env = env
        self.prev_s = None

    def reset(self, state_path):
        self.env.load_state_file(state_path)
        self.prev_s = None
        return self.observe()

    def raw(self):
        return player.read(self.env)

    def observe(self, state=None):
        state = state or self.raw()
        obs, s = observe(state, self.track, self.prev_s)
        self.prev_s = s
        return obs

    def step(self, frames=1, **action):
        self.env.step(frames=frames, **action)
        return self.observe()
