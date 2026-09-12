"""Prove the harness, in the order the claims depend on each other.

A training run that silently sits on a frozen emulator looks identical to a
policy that learned nothing, so each of these is an assert, not a print.

What rollback has to guarantee, precisely: *the same state plus the same actions
gives the same result*. That is what segment search spends its life doing. It is
not "the RAM after save_state() equals the RAM after load_state()" — waiting for
a savestate to finish writing takes a variable number of frames, so those two
moments are not the same instant and never were.
"""

import hashlib
import sys
import time

from .env import MK64Env, DEFAULT_ROM


def sha(b):
    return hashlib.sha1(b).hexdigest()[:12]


def main():
    print(f"rom: {DEFAULT_ROM}")
    t0 = time.time()
    env = MK64Env(headless=True, verbose="-v" in sys.argv)
    print(f"[1] booted in {time.time() - t0:.1f}s, frame={env.frame}")

    polls0, f0 = env.pad.polls, env.frame
    t0 = time.time()
    env.step(frames=120, accel=True)
    dt = time.time() - t0
    assert env.frame > f0, f"frame counter stuck at {env.frame}"
    assert env.pad.polls > polls0, "core never polled the input plugin"
    print(f"[2] 120 frames in {dt:.2f}s = {120/dt:.0f} fps "
          f"(frame {f0}->{env.frame}, polls +{env.pad.polls - polls0})")

    env.step(frames=240)
    seen = []
    for _ in range(7):
        env.step(1)
        seen.append(sha(env.game_state_bytes()))
    assert len(set(seen)) > 1, f"RDRAM frozen across 7 frames: {seen[0]}"
    print(f"[3] RDRAM live: {len(set(seen))}/7 distinct states")

    # 4. The claim everything else rests on: replay from a restored state is
    #    reproducible. Diverge hard in between so a no-op load cannot pass.
    sid = env.save_state()

    def run_from(sid, frames=30, **action):
        env.load_state(sid)
        env.step(frames=frames, **action)
        return sha(env.game_state_bytes())

    first = run_from(sid, accel=True)
    env.step(frames=77, accel=True, steer=-1.0, hop=True)   # wander off
    second = run_from(sid, accel=True)
    assert first == second, f"ROLLBACK NOT REPRODUCIBLE: {first} != {second}"
    print(f"[4] rollback reproducible across a divergence: {first}")

    # 5. ...and it is not reproducible merely because nothing is happening:
    #    a different action from the same state must land somewhere else.
    other = run_from(sid, accel=True, steer=1.0)
    assert other != first, "same result from different actions — state is inert"
    print(f"[5] different action -> different state: {other}")

    env.drop_state(sid)
    env.close()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
