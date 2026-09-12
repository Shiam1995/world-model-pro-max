"""Check that the kart is still where ram_hunt said it was.

Run this before trusting any training run. If MK64 is ever loaded differently —
another course, two players, a different mode — the struct can move, and a
policy fed garbage coordinates fails in ways that look like bad learning.
"""

import sys

import numpy as np

from .env import MK64Env
from . import player

STATE = "states/luigi_raceway_tt_start.st"


def main():
    env = MK64Env()
    env.step(frames=20)
    env.load_state_file(STATE)

    s = player.read(env)
    assert s["lapCount"] == 0, f"expected lap 0 at the start line, got {s['lapCount']}"
    assert s["speed"] == 0.0, f"kart should be parked, speed={s['speed']}"
    ok, why = player.check(s)
    assert ok, f"parked kart failed consistency: {why}"
    print(f"[1] parked at {[round(v, 1) for v in s['pos']]}, lap 0, speed 0")

    env.step(frames=200, accel=True)
    s = player.read(env)
    ok, why = player.check(s)
    assert ok, f"moving kart failed consistency: {why}"
    assert s["speed"] > 4.0, f"should be near top speed, got {s['speed']}"
    print(f"[2] cruising at {s['speed']:.2f}/tick, consistent "
          f"(speed == |velocity| == |pos-oldPos|)")

    prev = np.array(player.read(env)["pos"])
    ratios = []
    for _ in range(5):
        env.step(1, accel=True)
        cur = player.read(env)
        moved = float(np.linalg.norm(np.array(cur["pos"]) - prev))
        ratios.append(moved / cur["speed"])
        prev = np.array(cur["pos"])
    mean = sum(ratios) / len(ratios)
    assert abs(mean - player.TICKS_PER_FRAME) < 0.05, \
        f"ticks per frame drifted: {mean:.3f} != {player.TICKS_PER_FRAME}"
    print(f"[3] {mean:.2f} physics ticks per emulated frame, as calibrated")

    # steering has to actually steer, or the action space is a lie
    env.load_state_file(STATE)
    env.step(frames=120, accel=True, steer=-1.0)
    left = player.read(env)
    env.load_state_file(STATE)
    env.step(frames=120, accel=True, steer=+1.0)
    right = player.read(env)
    sep = float(np.linalg.norm(np.array(left["pos"]) - np.array(right["pos"])))
    assert sep > 50, f"left and right ended up {sep:.1f} apart — steering not reaching the game"
    print(f"[4] left vs right diverge by {sep:.0f} units after 120 frames")

    env.close()
    print("\nPLAYER CHECKS PASSED")


if __name__ == "__main__":
    main()
