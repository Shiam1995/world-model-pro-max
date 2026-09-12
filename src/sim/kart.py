"""A fast, vectorised kart model. Thousands of karts, one numpy array each.

Why this exists: the emulator runs at ~70-110 fps on one core, cannot be run
twice in a process, and a single searched lap costs about five minutes. Training
needs millions of frames. So the emulator stops being the training loop and
becomes the **oracle** — it defines truth, and this defines volume.

What is honest about it:

* The constants are MK64's own, lifted from the decomp's `kart_attributes.c`
  (top speed per engine class, the ten-point acceleration curve, friction,
  gravity). They are not invented.
* The *shape* of the model is not MK64's. `player_controller.c` is 5,096 lines
  of mostly-unnamed functions plus collision and surface handling; this is a
  point-mass kart with a speed-dependent turn rate. It is a fit, not a port.
* Therefore every parameter that is not a decomp constant is **calibrated
  against emulator traces** (`src/sim/calibrate.py`) and the residual is
  reported. A sim nobody compared against the real thing is a story.

Physics lives in one dataclass on purpose. That is stage 3's "physics becomes a
parameter set, not code" arriving early: dial `turn_rate`, `grip`, `accel_scale`
and `off_road` from kart toward hoverboard and the same policy can be re-measured
at each setting.
"""

from dataclasses import dataclass, asdict, replace
import json
from pathlib import Path

import numpy as np

ATTRS = Path(__file__).resolve().parents[2] / "sim/kart_attributes.json"


@dataclass
class Physics:
    """The knob set. Defaults are Mario, 100cc (what Time Trials uses)."""
    top_speed: float = 5.6          # game `speed` units per physics tick
    accel_curve: tuple = (2.0, 2.0, 2.0, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4)
    accel_scale: float = 0.012      # curve units -> speed units per tick
    brake: float = 0.15
    drag: float = 0.002
    turn_rate: float = 0.055        # radians per tick at full lock, at speed
    turn_speed_falloff: float = 0.55  # how much steering authority is lost fast
    grip: float = 1.0               # 1 = no sideslip; < 1 lets the kart drift
    off_road: float = 0.35          # speed multiplier off the track
    # Measured, not guessed: driving straight off turn one, the real kart held
    # 5.14 speed at lateral 70.9 and had collapsed to 3.64 by 75.2. The road
    # edge is therefore around 72, not the 130 originally assumed.
    half_width: float = 72.0
    ticks_per_frame: int = 2        # MK64 runs two physics ticks per frame
    #: Measured acceleration per FRAME as a function of speed, taken straight
    #: from an emulator trace (src/sim/calibrate.py). When present this replaces
    #: accel_curve/accel_scale entirely: forcing the decomp's curve shape onto
    #: the model fitted badly (10.9% RMSE, and top_speed pinned to its grid
    #: edge, which is what a wrong shape looks like). Measuring a(v) directly
    #: makes the match exact by construction.
    accel_table: tuple = ()
    #: Measured |dyaw| per FRAME at full lock, as a function of speed, taken
    #: from emulator traces. This replaces turn_rate/turn_speed_falloff and the
    #: invented low-speed ramp. That ramp was the single thing that broke
    #: sim-to-real: it made the sim kart unable to steer below ~1.4 speed, so a
    #: policy could carry a large constant steering bias for free. The real kart
    #: turns 0.0285 rad/frame at speed 0.2, and that bias drove it straight off
    #: the road in the first two seconds.
    turn_table: tuple = ()
    #: Measured steering response: stick deflection -> fraction of full lock.
    #: MK64's steering is strongly NON-linear with a deadzone — 0.15 produces
    #: exactly zero yaw rate and 0.30 produces 5% of full lock. Assuming it was
    #: linear is what broke sim-to-real: a policy trained on a linear model
    #: steers with gentle +-0.27 corrections, which do almost nothing on the
    #: real kart, and it drifts wide into the barrier every time.
    steer_response: tuple = ()

    @classmethod
    def calibrated(cls, **overrides):
        """Physics fitted to the emulator: measured a(v) plus fitted steering."""
        root = Path(__file__).resolve().parents[2]
        tbl = json.loads((root / "sim/accel_table.json").read_text())
        cal = json.loads((root / "sim/calibration.json").read_text())
        turn = json.loads((root / "sim/turn_table.json").read_text())
        sresp = json.loads((root / "sim/steer_response.json").read_text())
        kw = dict(accel_table=tuple(map(tuple, tbl)),
                  turn_table=tuple(map(tuple, turn)),
                  steer_response=tuple(map(tuple, sresp)),
                  top_speed=float(cal["real_top_speed"]),
                  turn_rate=float(cal["turn_rate"]),
                  turn_speed_falloff=float(cal["turn_speed_falloff"]),
                  # The measured table is NET acceleration — it already has the
                  # game's own drag in it. Applying drag on top double-counts
                  # and the kart never reaches top speed (5.15 against 5.58).
                  drag=0.0)
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def from_decomp(cls, character=0, cc="gTopSpeed100cc", **overrides):
        d = json.loads(ATTRS.read_text())
        name = ["Mario", "Luigi", "Yoshi", "Toad", "DK", "Wario", "Peach", "Bowser"][character]
        curve = tuple(d[f"gKartAcceleration{name}"])
        # The decomp's top speed (310 for Mario at 100cc) is in the game's
        # internal unit; the `speed` field we read from RDRAM tops out near 5.6.
        # calibrate.py measures that ratio rather than assuming it.
        return cls(accel_curve=curve, **overrides)


class KartSim:
    """N independent karts on one track, stepped together."""

    def __init__(self, track, physics=None, n=1024, seed=0):
        self.track = track
        self.p = physics or Physics()
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.xz = np.zeros((n, 2))
        self.yaw = np.zeros(n)
        self.speed = np.zeros(n)
        self.s = np.zeros(n)
        self.prev_s = np.zeros(n)
        #: Last known checkpoint per kart. Projection is searched in a window
        #: around this rather than globally — see _project().
        self.cp = np.zeros(n, dtype=int)
        self.alive = np.ones(n, dtype=bool)
        self.ticks = 0

    # --- setup -----------------------------------------------------------
    def reset(self, jitter_lateral=0.0, jitter_yaw=0.0, start_cp=0):
        """Put every kart on the line. Jitter is what stops a cloned policy
        learning one trajectory by heart instead of how to drive."""
        c = self.track.centre[start_cp]
        t = self.track.tangent[start_cp]
        heading = np.arctan2(t[0], t[1])
        normal = np.array([t[1], -t[0]])
        lat = self.rng.uniform(-jitter_lateral, jitter_lateral, self.n)
        self.xz = np.array([c[0], c[2]]) + normal * lat[:, None]
        self.yaw = heading + self.rng.uniform(-jitter_yaw, jitter_yaw, self.n)
        self.speed[:] = 0.0
        self.alive[:] = True
        self.ticks = 0
        self.cp[:] = start_cp
        self.s = self._project()[1]
        self.prev_s = self.s.copy()
        return self.observe()

    # --- dynamics --------------------------------------------------------
    def _accel(self):
        """Acceleration per tick at the current speed."""
        if self.p.accel_table:
            t = np.asarray(self.p.accel_table, dtype=float)
            # Table is per frame; the sim runs ticks_per_frame ticks per frame.
            return np.interp(self.speed, t[:, 0], t[:, 1]) / self.p.ticks_per_frame
        frac = np.clip(self.speed / max(self.p.top_speed, 1e-6), 0, 0.999)
        idx = (frac * len(self.p.accel_curve)).astype(int)
        return np.asarray(self.p.accel_curve)[idx] * self.p.accel_scale

    def step(self, steer, throttle=None, brake=None):
        """One emulated frame = `ticks_per_frame` physics ticks."""
        steer = np.clip(np.asarray(steer, dtype=float).reshape(self.n), -1, 1)
        throttle = np.ones(self.n) if throttle is None else np.asarray(throttle).reshape(self.n)
        brake = np.zeros(self.n) if brake is None else np.asarray(brake).reshape(self.n)
        p = self.p

        for _ in range(p.ticks_per_frame):
            lat = self._project()[2]
            off = np.abs(lat) > p.half_width
            cap = np.where(off, p.top_speed * p.off_road, p.top_speed)

            self.speed += self._accel() * throttle
            self.speed -= p.brake * brake
            self.speed -= p.drag * self.speed
            # Off the road the kart is dragged down to the off-road cap rather
            # than teleported to it: that is what makes a wide corner cost time
            # instead of instantly ending the episode.
            self.speed = np.where(self.speed > cap,
                                  np.maximum(cap, self.speed * 0.90),
                                  self.speed)
            self.speed = np.clip(self.speed, 0.0, p.top_speed * 1.5)

            if p.turn_table:
                tt = np.asarray(p.turn_table, dtype=float)
                rate = np.interp(self.speed, tt[:, 0], tt[:, 1]) / p.ticks_per_frame
                if p.steer_response:
                    sr = np.asarray(p.steer_response, dtype=float)
                    eff = np.interp(np.abs(steer), sr[:, 0], sr[:, 1]) * np.sign(steer)
                else:
                    eff = steer
                # NEGATIVE on purpose. Replaying one fixed action script in
                # both worlds showed +0.8 steer moving the real kart toward +x
                # and the sim kart toward -x: the turn was mirrored. They agreed
                # to 31 units while straight and were 678 apart after two turns.
                # This is what actually broke sim-to-real.
                self.yaw -= eff * rate
            else:
                auth = 1.0 - p.turn_speed_falloff * (self.speed / max(p.top_speed, 1e-6))
                self.yaw -= steer * p.turn_rate * np.clip(auth, 0.05, 1.0) * \
                    np.clip(self.speed / (0.25 * p.top_speed), 0, 1)

            self.xz += np.stack([np.sin(self.yaw), np.cos(self.yaw)], axis=1) \
                * self.speed[:, None]

        self.ticks += 1
        self.prev_s = self.s
        _, self.s, _ = self._project()
        return self.observe()

    # --- track queries (vectorised projection) ---------------------------
    #: How far forward/back of the last checkpoint to look. A lap is a loop, so
    #: two distant parts of the track can be physically adjacent: at Luigi
    #: Raceway the start straight passes within metres of the return leg. A
    #: global nearest-checkpoint search flips between cp 2 and cp 50 there, and
    #: each flip looks like thousands of units of progress. An ES policy found
    #: that immediately and learned to park in the ambiguous zone farming the
    #: jumps — 16.5 laps in 900 frames, when 0.8 is the physical maximum.
    #: Searching locally makes that impossible to express.
    LOOK_BACK, LOOK_FWD = 3, 12

    def _project(self, update=True):
        n_cp = len(self.track.xz)
        offs = np.arange(-self.LOOK_BACK, self.LOOK_FWD + 1)
        idx = (self.cp[:, None] + offs[None, :]) % n_cp          # (N, W)
        cand = self.track.xz[idx]                                 # (N, W, 2)
        d = cand - self.xz[:, None, :]
        k = np.argmin(np.einsum("nwj,nwj->nw", d, d), axis=1)
        i = idx[np.arange(len(idx)), k]
        if update:
            self.cp = i
        t = self.track.tangent[i]
        rel = self.xz - self.track.xz[i]
        along = np.einsum("nj,nj->n", rel, t)
        lateral = -(rel[:, 0] * t[:, 1] - rel[:, 1] * t[:, 0])
        return i, self.track.s[i] + along, lateral

    def progress(self):
        d = self.s - self.prev_s
        half = self.track.length / 2
        d = np.where(d < -half, d + self.track.length, d)
        d = np.where(d > half, d - self.track.length, d)
        # Belt and braces: nothing can advance further in one frame than the
        # kart can physically travel. Any larger value is a projection artefact,
        # and paying it out is how the first policy learned to cheat.
        cap = self.p.top_speed * self.p.ticks_per_frame * 1.5
        return np.clip(d, -cap, cap)

    def observe(self):
        """The same ten numbers as src/obs/track.py, computed for N karts."""
        i, s, lat = self._project()
        t = self.track.tangent[i]
        head = np.stack([np.sin(self.yaw), np.cos(self.yaw)], axis=1)
        err = np.arctan2(head[:, 0] * t[:, 1] - head[:, 1] * t[:, 0],
                         head[:, 0] * t[:, 0] + head[:, 1] * t[:, 1])
        n = len(self.track.centre)
        def curv(dist):
            j = (i + int(round(dist / self.track.spacing))) % n
            return self.track.curvature[j]
        j8 = (i + 8) % n
        grade = (self.track.centre[j8][:, 1] - self.track.centre[i][:, 1]) / (8 * self.track.spacing)
        return np.stack([
            self.speed / 6.0,
            lat / 300.0,
            np.sin(err), np.cos(err),
            curv(300.) * 1000, curv(600.) * 1000,
            curv(1200.) * 1000, curv(2400.) * 1000,
            grade * 10.0,
            self.progress() / 12.0,
        ], axis=1).astype(np.float32)

    def tune(self, **knobs):
        """Return a copy of this sim's physics with knobs changed (stage 3)."""
        return replace(self.p, **knobs)

    def params(self):
        return asdict(self.p)
