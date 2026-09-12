"""Reading the kart out of RDRAM.

The address below was found by experiment (src/mk64/ram_hunt.py), not by
reading a symbol map: run several trajectories from one savestate and look for
memory that moves with the throttle, stays put without it, and splits the two
ways under left and right steering. The struct *layout* came from the mk64
decomp; the *address* had to be measured, which is the part that will still
work on a game with no decomp.

The find validates itself, which is why it can be trusted: at the winning base,
three independently stored fields agree to the last digit —

    speed == |velocity| == |pos - oldPos|   (4.82 units per frame)

Nothing lands on that by coincidence.
"""

import numpy as np

#: Player 1's struct in RDRAM. Verified on MK64 (USA) in a Luigi Raceway time
#: trial. `check()` guards against it moving in another mode or course.
PLAYER_BASE = 0x0F6990

FIELDS = {
    "lapCount":     (0x008, "<i2", 1),
    "pos":          (0x014, "<f4", 3),
    "oldPos":       (0x020, "<f4", 3),
    "rotation":     (0x02C, "<i2", 3),
    "velocity":     (0x034, "<f4", 3),
    "steerPosition": (0x07C, "<i4", 1),
    "boostPower":   (0x080, "<f4", 1),
    "speed":        (0x094, "<f4", 1),
    "currentSpeed": (0x09C, "<f4", 1),
}

#: N64 angles are s16 turns: -32768..32767 maps onto -pi..pi.
ANGLE_SCALE = np.pi / 32768.0

#: MK64 runs TWO physics ticks per rendered frame. `speed` and `pos - oldPos`
#: are per *tick*, so the kart covers 2 x speed per emulator frame. Measured at
#: 2.00 exactly, over and over, at cruising speed. Getting this wrong puts a
#: silent factor of two into every distance, reward and lookahead.
TICKS_PER_FRAME = 2


def speed_per_frame(state):
    """Units the kart covers per emulated frame (not per physics tick)."""
    return float(state["speed"]) * TICKS_PER_FRAME


def read(env, base=PLAYER_BASE):
    """Snapshot the kart. One RDRAM read, no allocation of 8 MiB."""
    raw = env.ram_bytes(base, 0xA0)
    out = {}
    for name, (off, dt, cnt) in FIELDS.items():
        v = np.frombuffer(raw[off:off + np.dtype(dt).itemsize * cnt], dtype=dt)
        out[name] = v.tolist() if cnt > 1 else v[0].item()
    # Heading in radians, from the yaw word. Rotation is stored as s16 turns.
    out["yaw"] = out["rotation"][0] * ANGLE_SCALE
    return out


def check(state, tol=0.25):
    """Is this really the kart? Returns (ok, reason).

    The three-way agreement between speed, |velocity| and the one-frame step
    is what identified the struct in the first place, so it doubles as a
    runtime guard: if the address ever drifts — another course, two players,
    a different mode — this fails loudly instead of feeding the policy noise.
    """
    pos = np.array(state["pos"], dtype=float)
    old = np.array(state["oldPos"], dtype=float)
    vel = np.array(state["velocity"], dtype=float)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        return False, "non-finite pos/velocity"
    speed = float(state["speed"])
    vmag = float(np.linalg.norm(vel))
    step = float(np.linalg.norm(pos - old))
    if speed < 1e-6:                      # parked: nothing to cross-check
        return (vmag < 1e-3 and step < 1e-3), "parked"
    if abs(vmag - speed) > tol * max(speed, 1.0):
        return False, f"|velocity| {vmag:.3f} != speed {speed:.3f}"
    if abs(step - speed) > tol * max(speed, 1.0):
        return False, f"|pos-oldPos| {step:.3f} != speed {speed:.3f}"
    return True, "consistent"
