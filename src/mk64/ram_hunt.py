"""Find the kart in RDRAM by controlled experiment, not by disassembly.

The trick is that rollback makes RAM a laboratory. From one savestate we can run
several different trajectories through *the same* starting instant, so anything
that differs between them was caused by the input and nothing else. No timing
noise, no "roughly the same lap", no drift.

Trajectories, all from the identical state:

    still   no input          -- what changes on its own (timers, animation, RNG)
    fwd     throttle          -- what moves when the kart moves
    left    throttle + left   -- what moves the other way from...
    right   throttle + right  -- ...this

A position word must move under `fwd`, stay put under `still`, and split in
opposite directions under `left`/`right`. Very little else in 8 MiB does all
three at once.

This is deliberately the general technique rather than reading the mk64 decomp's
player struct: it is the only one that will still work on Sonic Riders, where
there is no decomp to read. The decomp is used afterwards, as an oracle, to
check the answer.
"""

import numpy as np

# MK64's three framebuffers live at 0x31fa00 / 0x345200 / 0x36aa00, each
# 320*240*2 = 0x25800 bytes, so video memory spans 0x31fa00-0x390200. Anything
# found in there is a picture of the kart, not the kart.
FB_LO, FB_HI = 0x310000, 0x3A0000

# mupen64plus keeps RDRAM in HOST word order, not the N64's big-endian layout.
# On x86 that means a 32-bit word reads as little-endian. Scanning as ">f4"
# finds nothing real -- not even a frame counter, which is the giveaway: a
# running game always has one. Verified at 0x0E9E24, which counts 60,61,62...
WORD = "<u4"
FLOAT = "<f4"
RDRAM_SIZE = 8 << 20

TRAJECTORIES = {
    "still": dict(),
    "fwd":   dict(accel=True),
    "left":  dict(accel=True, steer=-1.0),
    "right": dict(accel=True, steer=+1.0),
}


def collect(env, state_path, samples=(45, 90, 150)):
    """Run every trajectory from the same instant; return RAM at each sample."""
    env.step(frames=30)
    env.load_state_file(state_path)
    sid = env.save_state()

    snaps = {}
    for name, action in TRAJECTORIES.items():
        env.load_state(sid)
        got, prev = [], 0
        for t in samples:
            env.step(frames=t - prev, **action)
            got.append(np.frombuffer(env.ram_bytes(), dtype=np.uint8).copy())
            prev = t
        snaps[name] = got
    env.drop_state(sid)
    return snaps, samples


def _as(snap, dtype):
    return snap[:RDRAM_SIZE].view(dtype)


def find_position(snaps, samples, dtype=FLOAT):
    """Words that move with the kart and split left/right. Returns (addr, info)."""
    n = RDRAM_SIZE // 4
    still = [_as(s, dtype) for s in snaps["still"]]
    fwd   = [_as(s, dtype) for s in snaps["fwd"]]
    left  = [_as(s, dtype) for s in snaps["left"]]
    right = [_as(s, dtype) for s in snaps["right"]]

    with np.errstate(invalid="ignore", over="ignore"):
        finite = np.isfinite(fwd[-1]) & np.isfinite(still[-1])
        # Plausible world coordinates: MK64 tracks are thousands of units across.
        sane = finite & (np.abs(fwd[-1]) > 1e-3) & (np.abs(fwd[-1]) < 1e7)

        # 1. throttle changed it, doing nothing did not
        moved = sane & (np.abs(fwd[-1] - still[-1]) > 1.0)
        parked = np.abs(still[-1] - still[0]) < np.abs(fwd[-1] - fwd[0]) * 0.05

        # 2. it keeps going the same way over time (a position, not a flicker)
        d1 = fwd[1] - fwd[0]
        d2 = fwd[2] - fwd[1]
        monotonic = (np.sign(d1) == np.sign(d2)) & (np.abs(d2) > 0.5)

        # 3. steering splits it, and the two directions land on opposite sides
        splits = (np.abs(left[-1] - right[-1]) > 1.0)
        opposed = np.sign(left[-1] - fwd[-1]) != np.sign(right[-1] - fwd[-1])

        hit = moved & parked & monotonic & splits & opposed

    idx = np.flatnonzero(hit)
    out = []
    for i in idx:
        out.append({
            "addr": int(i) * 4,
            "still": float(still[-1][i]), "fwd": float(fwd[-1][i]),
            "left": float(left[-1][i]),   "right": float(right[-1][i]),
            "traj": [float(f[i]) for f in fwd],
        })
    return out


def drop_video(cands):
    return [c for c in cands if not (FB_LO <= c["addr"] < FB_HI)]


def cluster(cands, gap=0x40):
    """Group neighbouring hits — a position is usually three words in a row."""
    groups, cur = [], []
    for c in sorted(cands, key=lambda c: c["addr"]):
        if cur and c["addr"] - cur[-1]["addr"] <= gap:
            cur.append(c)
        else:
            if cur:
                groups.append(cur)
            cur = [c]
    if cur:
        groups.append(cur)
    return groups


# --- structural search -------------------------------------------------------
#
# The statistical hunt above treats RAM as an undifferentiated soup and asks
# which words behave like a position. It finds mostly noise: 8 MiB reinterpreted
# as floats throws up plenty of values that drift one way for three samples.
#
# The structural search is far sharper. A kart keeps both `pos` and `oldPos`,
# 12 bytes apart, and oldPos at frame t+1 is *bitwise* the pos of frame t. Three
# floats matching exactly, frame after frame, at a fixed stride, is a signature
# that essentially nothing else in memory produces.
#
# The stride comes from the mk64 decomp's Player struct (pos 0x14, oldPos 0x20,
# velocity 0x34, speed 0x94, size 0xDD8). That is layout, not an address --
# the address is still found by experiment, which is the part that has to keep
# working on a game with no decomp. A "previous position" field is near
# universal in game physics, so the trick itself travels.

POS_OFF, OLDPOS_OFF, VEL_OFF, SPEED_OFF = 0x14, 0x20, 0x34, 0x94
OLDPOS_STRIDE = OLDPOS_OFF - POS_OFF          # 0x0C


def find_player_struct(env, state_path, warmup=90, action=None):
    """Locate the player struct by the oldPos == previous pos signature."""
    action = action or dict(accel=True)
    env.step(frames=20)
    env.load_state_file(state_path)
    env.step(frames=warmup, **action)

    frames = []
    for _ in range(4):
        env.step(1, **action)
        frames.append(np.frombuffer(env.ram_bytes(), dtype=np.uint8).copy())

    n = RDRAM_SIZE
    # Bytes that could be a `pos` whose `oldPos` 12 bytes later holds the
    # previous frame's value, and that actually moved.
    hits = None
    for a, b in zip(frames, frames[1:]):
        pos_t = np.lib.stride_tricks.sliding_window_view(a, 12)[: n - 0x20 : 4]
        old_t1 = np.lib.stride_tricks.sliding_window_view(b, 12)[OLDPOS_STRIDE: n - 0x20 + OLDPOS_STRIDE: 4]
        pos_t1 = np.lib.stride_tricks.sliding_window_view(b, 12)[: n - 0x20 : 4]
        same = (old_t1 == pos_t).all(axis=1)          # oldPos carried forward
        moved = (pos_t1 != pos_t).any(axis=1)         # and the kart moved
        ok = same & moved
        hits = ok if hits is None else (hits & ok)

    idx = np.flatnonzero(hits) * 4
    out = []
    for pos_addr in idx:
        if FB_LO <= pos_addr < FB_HI:
            continue
        base = int(pos_addr) - POS_OFF
        last = frames[-1]
        def f3(off):
            return np.frombuffer(last[off:off + 12].tobytes(), dtype=">f4")
        def f1(off):
            return float(np.frombuffer(last[off:off + 4].tobytes(), dtype=">f4")[0])
        out.append({
            "base": base,
            "pos": f3(base + POS_OFF).tolist(),
            "oldPos": f3(base + OLDPOS_OFF).tolist(),
            "velocity": f3(base + VEL_OFF).tolist(),
            "speed": f1(base + SPEED_OFF),
            "lapCount": int(np.frombuffer(last[base + 8:base + 10].tobytes(), dtype=">i2")[0]),
        })
    return out


def find_speed(snaps, lo=0.5, hi=200.0):
    """Floats that sit at exactly 0 while parked and climb under throttle.

    Speed is the cleanest hook in the whole struct: a time-trial kart with no
    throttle is *exactly* stationary, so the still trajectory pins the field to
    0.0 bit-for-bit, while the fwd trajectory has to rise monotonically. Almost
    nothing else in RAM does both.

    Once speed is found at S, the decomp's layout gives the rest of the struct
    for free: base = S - 0x94, pos = base + 0x14.
    """
    still = [_as(s, FLOAT) for s in snaps["still"]]
    fwd = [_as(s, FLOAT) for s in snaps["fwd"]]

    with np.errstate(invalid="ignore", over="ignore"):
        parked = np.ones(still[0].shape, dtype=bool)
        for s in still:
            parked &= (s == 0.0)

        climbing = np.isfinite(fwd[-1]) & (fwd[0] > 0) & (fwd[-1] > lo) & (fwd[-1] < hi)
        for a, b in zip(fwd, fwd[1:]):
            climbing &= (b >= a)                 # never goes backwards
        climbing &= (fwd[-1] > fwd[0])           # and actually got somewhere

    idx = np.flatnonzero(parked & climbing)
    return [{"addr": int(i) * 4, "fwd": [float(f[i]) for f in fwd]} for i in idx]


# --- identification ----------------------------------------------------------

PLAYER_FIELDS = {           # from the mk64 decomp's Player struct (size 0xDD8)
    "lapCount": (0x008, "<i2"),
    "pos":      (0x014, "<f4", 3),
    "oldPos":   (0x020, "<f4", 3),
    "rotation": (0x02C, "<i2", 3),
    "velocity": (0x034, "<f4", 3),
    "speed":    (0x094, "<f4"),
    "currentSpeed": (0x09C, "<f4"),
}


def read_player(snap, base):
    out = {}
    for name, spec in PLAYER_FIELDS.items():
        off, dt = spec[0], spec[1]
        cnt = spec[2] if len(spec) > 2 else 1
        w = np.dtype(dt).itemsize
        raw = snap[base + off: base + off + w * cnt].tobytes()
        v = np.frombuffer(raw, dtype=dt)
        out[name] = v.tolist() if cnt > 1 else v[0].item()
    return out


def score_player(snaps, base):
    """How well does `base` behave like the player struct? Higher is better."""
    still = read_player(snaps["still"][-1], base)
    fwd = read_player(snaps["fwd"][-1], base)
    left = read_player(snaps["left"][-1], base)
    right = read_player(snaps["right"][-1], base)

    def finite(xs):
        return all(np.isfinite(x) and abs(x) < 1e6 for x in np.atleast_1d(xs))

    if not all(finite(d[k]) for d in (still, fwd, left, right)
               for k in ("pos", "oldPos", "velocity", "speed")):
        return 0, None

    s = 0
    # parked with the throttle off, moving with it on
    if still["speed"] == 0.0:            s += 2
    if fwd["speed"] > 1.0:               s += 2
    if abs(still["currentSpeed"]) < 1e-6 and fwd["currentSpeed"] > 1.0: s += 1
    # the ground does not move under a kart on a flat straight
    if abs(fwd["pos"][1] - still["pos"][1]) < 5:   s += 1
    # throttle covers real ground
    d = np.linalg.norm(np.array(fwd["pos"]) - np.array(still["pos"]))
    if 100 < d < 5e4:                    s += 2
    # steering splits the two directions, and they land on opposite sides
    lr = np.linalg.norm(np.array(left["pos"]) - np.array(right["pos"]))
    if lr > 10:                          s += 2
    # oldPos trails pos by about one frame of travel, never equals it exactly
    gap = np.linalg.norm(np.array(fwd["pos"]) - np.array(fwd["oldPos"]))
    if 0 < gap < 100:                    s += 2
    # velocity should be that same one-frame step
    vel = np.linalg.norm(fwd["velocity"])
    if gap > 0 and 0.5 < vel / max(gap, 1e-6) < 2.0: s += 3
    # a time trial starts on lap 0
    if still["lapCount"] in (0, 1):      s += 1
    return s, {"still": still, "fwd": fwd, "left": left, "right": right}
