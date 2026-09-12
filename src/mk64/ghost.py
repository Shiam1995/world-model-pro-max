"""Drive a first lap by search, and log it. This is the "ghost" in ghost -> map.

The bootstrap problem: a policy needs a map to know which way is forward, and
the map is built from a driven lap. Nothing can drive yet and there is no map.

Rollback breaks the deadlock. At each segment, branch the savestate across a few
steering choices, run each for a fraction of a second, and keep whichever ends
up **furthest from where the segment began**. That needs no track knowledge at
all: MK64 punishes leaving the road by slowing the kart, so the greedy choice
is to stay on it. The road is discovered, not described.

The result is not a good lap. It is a *completed* lap, which is all the map
needs — the racing line comes later, from training.
"""

import numpy as np

from . import player

#: Five is a deliberate budget: each extra choice costs a full rollout, and
#: in-game rendering caps emulation near 70 fps.
STEERS = (-1.0, -0.45, 0.0, 0.45, 1.0)


def _score(start_pos, end):
    """How much ground did this choice cover, and is the kart still healthy?"""
    moved = float(np.linalg.norm(np.array(end["pos"]) - start_pos))
    # Speed at the end matters as well as distance covered: two choices can
    # travel equally far while one is about to be stuck against a wall.
    return moved + 4.0 * float(end["speed"])


def drive_lap(env, state_path, horizon=30, max_segments=400, stall_limit=12,
              verbose=True):
    """Search out one complete lap. Returns the per-frame log."""
    env.step(frames=20)
    env.load_state_file(state_path)

    sid = env.save_state()
    log = []
    stalls = 0
    lap0 = player.read(env)["lapCount"]

    for seg in range(max_segments):
        start = player.read(env)
        start_pos = np.array(start["pos"])

        best, best_score = None, -1e18
        for steer in STEERS:
            env.load_state(sid)
            env.step(frames=horizon, accel=True, steer=steer)
            sc = _score(start_pos, player.read(env))
            if sc > best_score:
                best, best_score = steer, sc

        # Commit the winner, logging every frame of it — the log is the ghost.
        env.load_state(sid)
        for _ in range(horizon):
            env.step(1, accel=True, steer=best)
            st = player.read(env)
            log.append({
                "frame": env.frame, "pos": st["pos"], "yaw": st["yaw"],
                "speed": st["speed"], "lap": st["lapCount"], "steer": best,
            })
        env.drop_state(sid)
        sid = env.save_state()

        end = player.read(env)
        travelled = float(np.linalg.norm(np.array(end["pos"]) - start_pos))
        stalls = stalls + 1 if travelled < 20 else 0
        if stalls >= stall_limit:
            raise RuntimeError(f"kart stuck for {stalls} segments at {end['pos']}")

        if verbose and seg % 10 == 0:
            print(f"  seg {seg:3d} steer {best:+.2f} speed {end['speed']:5.2f} "
                  f"lap {end['lapCount']} pos "
                  f"[{end['pos'][0]:8.1f} {end['pos'][2]:8.1f}]", flush=True)

        if end["lapCount"] > lap0:
            if verbose:
                print(f"  lap complete after {seg + 1} segments "
                      f"({len(log)} frames)", flush=True)
            break
    else:
        raise RuntimeError("lap never completed within max_segments")

    env.drop_state(sid)
    return log
