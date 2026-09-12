"""Drive a lap by savestate segment search, against real progress.

The earlier attempt (ghost.py) maximised raw displacement and wedged the kart on
turn one: covering ground is not the same as getting round. Now that
`track.project` exists, the objective is the honest one — **arc length gained
along the centreline** — and a corner taken wide scores worse than one taken
properly, instead of merely differently.

This is the "improve by savestate search" step of SPEC.md, doing double duty: it
is also the expert whose choices milestone 5 clones.
"""

import numpy as np

from . import player

#: Steering choices per segment. Odd count so "straight" is always available.
STEERS = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)
#: Tried only when the kart is stuck; reversing is never worth a rollout
#: otherwise, and including it everywhere doubles the search for nothing.
RECOVERY = ({"brake": True, "steer": -1.0}, {"brake": True, "steer": 1.0})


class LapSearch:
    def __init__(self, env, track, horizon=20, hop=False,
                 lateral_soft=60.0, lateral_weight=0.6, speed_weight=2.0):
        self.env = env
        self.track = track
        self.horizon = horizon
        self.hop = hop
        self.lateral_soft = lateral_soft
        self.lateral_weight = lateral_weight
        self.speed_weight = speed_weight

    def _progress(self, s_from, s_to):
        """Arc length gained, handling the wrap at the finish line."""
        d = s_to - s_from
        if d < -self.track.length / 2:
            d += self.track.length
        elif d > self.track.length / 2:
            d -= self.track.length
        return d

    def score(self, s_start, state):
        _, s_end, lat = self.track.project(state["pos"])
        gain = self._progress(s_start, s_end)
        # Off the line is where the walls are. A soft band keeps the search on
        # the road without forbidding a racing line inside it.
        excess = max(0.0, abs(lat) - self.lateral_soft)
        return (gain
                + self.speed_weight * float(state["speed"])
                - self.lateral_weight * excess), gain, lat

    def segment(self, sid, s_start, actions):
        best, best_score, best_info = None, -1e18, None
        for act in actions:
            self.env.load_state(sid)
            self.env.step(frames=self.horizon, accel=not act.get("brake"), **act)
            sc, gain, lat = self.score(s_start, player.read(self.env))
            if sc > best_score:
                best, best_score, best_info = act, sc, (gain, lat)
        return best, best_info

    def drive_lap(self, state_path, max_segments=300, verbose=True, log=None):
        env = self.env
        env.step(frames=20)
        env.load_state_file(state_path)

        st = player.read(env)
        lap0 = st["lapCount"]
        _, s, _ = self.track.project(st["pos"])
        sid = env.save_state()
        total = 0.0
        stuck = 0

        for seg in range(max_segments):
            actions = [{"steer": v, "hop": self.hop} for v in STEERS]
            if stuck >= 2:
                actions = actions + [dict(a) for a in RECOVERY]
            best, (gain, lat) = self.segment(sid, s, actions)

            env.load_state(sid)
            for _ in range(self.horizon):
                env.step(1, accel=not best.get("brake"), **best)
                if log is not None:
                    cur = player.read(env)
                    log.append({"frame": env.frame, "pos": cur["pos"],
                                "yaw": cur["yaw"], "speed": cur["speed"],
                                "steer": best.get("steer", 0.0),
                                "brake": bool(best.get("brake")),
                                "lap": cur["lapCount"]})
            env.drop_state(sid)
            sid = env.save_state()

            st = player.read(env)
            _, s, lat = self.track.project(st["pos"])
            total += gain
            stuck = stuck + 1 if gain < 20 else 0

            if verbose and seg % 5 == 0:
                print(f"  seg {seg:3d} steer {best.get('steer', 0):+.1f}"
                      f"{' BRAKE' if best.get('brake') else '     '} "
                      f"speed {st['speed']:5.2f} lat {lat:7.1f} "
                      f"progress {total:7.0f}/{self.track.length:.0f}", flush=True)

            if st["lapCount"] > lap0:
                if verbose:
                    print(f"  LAP COMPLETE after {seg + 1} segments, "
                          f"{(seg + 1) * self.horizon} frames", flush=True)
                env.drop_state(sid)
                return True, total, seg + 1
            if stuck >= 10:
                if verbose:
                    print(f"  stuck at progress {total:.0f}", flush=True)
                env.drop_state(sid)
                return False, total, seg + 1

        env.drop_state(sid)
        return False, total, max_segments
