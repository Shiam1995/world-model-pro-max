"""Record the trained policy driving a real Mario Kart 64 lap, as PNG frames.

    python3 scripts/record_lap.py [out_dir] [policy] [every]

Frames come from the core's own screenshot path (queued, never stepped for), so
the drive is exactly the drive -- no extra frames are emulated to take a picture
and the trajectory is bit-identical to `make demo`.

The core numbers screenshots `<rom>-000.png`..`-999.png` and silently stops
once every index is taken, which is 16.6 seconds of a 26.7-second lap. So the
directory is drained into the output while the kart is still driving.
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mk64.env import MK64Env
from src.mk64 import player
from src.obs.mk64_adapter import MK64Adapter
from src.policy.drive import Driver

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "states/luigi_raceway_tt_start.st"


def record(out_dir, policy="policies/luigi_raceway_v2.npy", every=1,
           max_frames=2500, lead_in=0, tail=90):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    env = MK64Env()
    # Screenshots land here, one per queued frame, named by the core.
    shots = Path(env.shot_dir)
    for p in shots.glob("lap_*.png"):
        p.unlink()
    drv = Driver.load(ROOT / policy)
    ad = MK64Adapter(env, str(ROOT / "tracks/luigi_raceway.json"))
    obs = ad.reset(str(STATE))
    lap0 = max(0, player.read(env)["lapCount"])

    t0 = time.time()
    shot_frames = 0
    kept = 0
    done_at = None

    def drain(leave=2):
        """Move landed screenshots out, freeing the core's 000-999 indices.

        The newest files may still be half-written, so `leave` of them stay.
        """
        nonlocal kept
        got = sorted(shots.glob("*.png"), key=lambda q: q.stat().st_mtime)
        got = [q for q in got if q.stat().st_mtime >= t0 - 1]
        for q in got[:max(0, len(got) - leave)]:
            q.rename(out / f"lap_{kept:05d}.png")
            kept += 1
    for k in range(max_frames):
        if k % every == 0:
            env.core.take_screenshot()
            shot_frames += 1
        s = drv.steer(obs)
        env.step(1, accel=True, steer=s)
        st = player.read(env)
        obs = ad.observe(st)
        if done_at is None and st["lapCount"] > lap0:
            done_at = k
            print(f"  lap complete at frame {k} ({k/60:.2f}s)", flush=True)
        if done_at is not None and k >= done_at + tail:
            break
        if k % 200 == 199:
            drain()
        if k % 300 == 0:
            print(f"  frame {k:5d}  speed {st['speed']:5.2f}  "
                  f"lap {st['lapCount']}", flush=True)
    frames = k + 1
    env.close()

    # The core writes asynchronously; give the last ones a moment to land.
    time.sleep(1.0)
    drain(leave=0)
    print(f"{frames} frames driven, {shot_frames} queued, {kept} captured "
          f"-> {out}  ({time.time()-t0:.0f}s wall)")
    return kept, done_at is not None


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/lap_frames"
    pol = sys.argv[2] if len(sys.argv) > 2 else "policies/luigi_raceway_v2.npy"
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    n, ok = record(out, pol, every)
    sys.exit(0 if n and ok else 1)
