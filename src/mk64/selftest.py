"""Prove the harness, in the order the claims depend on each other.

A training run that silently sits on a frozen emulator looks identical to a
policy that learned nothing, so each of these is an assert, not a print.
"""

import hashlib
import sys
import time

from .env import MK64Env, DEFAULT_ROM
from . import core as m64


def sha(b):
    return hashlib.sha1(b).hexdigest()[:12]


def main():
    print(f"rom: {DEFAULT_ROM}")
    t0 = time.time()
    env = MK64Env(headless=True, verbose="-v" in sys.argv)
    print(f"[1] booted headless in {time.time() - t0:.1f}s, frame={env.frame}")

    # 1. frames actually advance, and the core really reads our controller
    polls0, f0 = env.pad.polls, env.frame
    t0 = time.time()
    env.step(frames=120, accel=True)
    dt = time.time() - t0
    assert env.frame > f0, f"frame counter stuck at {env.frame}"
    assert env.pad.polls > polls0, "core never polled the input plugin"
    print(f"[2] 120 frames in {dt:.2f}s = {120/dt:.0f} fps headless "
          f"(frame {f0}->{env.frame}, polls +{env.pad.polls - polls0})")

    # 2. RDRAM is readable and is changing.
    #    Skip the boot logo first — early on, whole megabytes sit untouched and
    #    a too-narrow or too-early check calls a working emulator broken.
    env.step(frames=240, accel=False)
    # The attract screen loops on a 3-frame cycle, so comparing two samples an
    # exact multiple of 3 apart shows no change on a perfectly healthy emulator.
    # Sample several consecutive frames and require that they are not all equal.
    seen = []
    for _ in range(7):
        env.step(1, accel=True)
        seen.append(sha(env.game_state_bytes()))
    assert len(set(seen)) > 1, f"RDRAM frozen across 7 frames: {seen[0]}"
    print(f"[3] RDRAM live across 7 frames: {len(set(seen))} distinct states {seen[:3]}...")

    # 3. THE claim: save, diverge, restore, and land back on the same bytes.
    #    Without this there is no world model and no segment search.
    sid = env.save_state()
    at_save = env.game_state_bytes()
    # 91, not 90: a multiple of the attract screen's 3-frame cycle lands back on
    # the same state. Note this only shows that time moved on — on the title
    # screen the stick does nothing. Proving input *controls* something needs a
    # race running, which is the next milestone (ram_hunt), not this one.
    env.step(frames=91, accel=True, steer=-1.0)
    diverged = env.game_state_bytes()
    assert diverged != at_save, "91 frames changed nothing — emulator stalled"
    env.load_state(sid)
    restored = env.game_state_bytes()
    print(f"[4] save={sha(at_save)} diverged={sha(diverged)} restored={sha(restored)}")
    assert restored == at_save, "ROLLBACK BROKEN: restored RAM != saved RAM"
    print("[4] rollback exact — save_state() is a usable env.clone()")

    # 4. determinism: the same actions from the same state must give the same RAM,
    #    or every search result is noise.
    env.load_state(sid)
    env.step(frames=60, accel=True, steer=0.5)
    first = sha(env.game_state_bytes())
    env.load_state(sid)
    env.step(frames=60, accel=True, steer=0.5)
    second = sha(env.game_state_bytes())
    assert first == second, f"NON-DETERMINISTIC: {first} != {second}"
    print(f"[5] deterministic replay: {first} twice")

    env.drop_state(sid)
    env.close()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
