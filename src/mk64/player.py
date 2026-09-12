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
    # The game's own progress index along the course path — how MK64 knows
    # where you are for lap counting and rank. Far better than inferring
    # progress from displacement, which cannot tell a corner from a circle.
    "nearestPathPointId": (0x220, "<i2", 1),
    "topSpeed":     (0x214, "<f4", 1),
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
    """Snapshot the kart. One RDRAM read, no allocation of 8 MiB.

    Note the `^ 2` on 16-bit fields. mupen64plus stores RDRAM byte-swapped
    *within each 32-bit word*, so a 32-bit read works little-endian (which is
    why pos/speed were right all along) but a 16-bit field at N64 offset X
    actually lives at host offset X^2. Reading it straight gives the neighbouring
    halfword: rotation[1] and rotation[2] came back as a constant 0, and
    `nearestPathPointId` looked permanently unused when it was simply never
    being read. 8-bit fields would need X^3 by the same rule.
    """
    raw = env.ram_bytes(base, 0x230)
    out = {}
    for name, (off, dt, cnt) in FIELDS.items():
        w = np.dtype(dt).itemsize
        if w == 2:
            vals = []
            for k in range(cnt):
                o = (off + 2 * k) ^ 2
                vals.append(np.frombuffer(raw[o:o + 2], dtype=dt)[0].item())
            out[name] = vals if cnt > 1 else vals[0]
            continue
        v = np.frombuffer(raw[off:off + w * cnt], dtype=dt)
        out[name] = v.tolist() if cnt > 1 else v[0].item()
    # Heading in radians, from the yaw word. Rotation is stored as s16 turns.
    # rotation is (pitch, yaw, roll); index 1 after the 16-bit ^2 fix.
    out["yaw"] = out["rotation"][1] * ANGLE_SCALE
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
