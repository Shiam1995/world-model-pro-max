"""THE LIVE DEMO — watch the AI drive Mario Kart 64 in a real emulator window.

    python3 scripts/demo_live.py            # replay the searched lap
    python3 scripts/demo_live.py policy     # drive live from the trained policy

Replay mode plays back the action sequence the rollback search found, at normal
speed, in the real game. Nothing is faked: the same ROM, the same savestate, and
the controller is driven frame by frame through the shared-memory input plugin.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.mk64.env import MK64Env
from src.mk64 import player

STATE = "states/luigi_raceway_tt_start.st"


def replay(path="runs/expert_lap.json", realtime=True):
    blob = json.loads(Path(path).read_text())
    log = blob["log"]
    print(f"replaying {len(log)} frames of searched driving "
          f"({'lap completed' if blob.get('finished') else 'partial lap'})")

    env = MK64Env()
    env.load_state_file(STATE)
    t0 = time.time()
    for k, e in enumerate(log):
        env.step(1, accel=not e.get("brake"), steer=e["steer"],
                 brake=bool(e.get("brake")))
        if realtime:
            # Hold roughly 60 fps so a human can watch it.
            target = t0 + (k + 1) / 60.0
            now = time.time()
            if target > now:
                time.sleep(target - now)
        if k % 120 == 0:
            st = player.read(env)
            print(f"  frame {k:5d}  speed {st['speed']:5.2f}  lap {st['lapCount']}",
                  flush=True)
    st = player.read(env)
    print(f"done: lap {st['lapCount']}, {len(log)} frames "
          f"({len(log)/60:.1f}s of game time)")
    env.close()


def live(policy="policies/es_robust.npy"):
    from src.obs.mk64_adapter import MK64Adapter
    from src.sim.train_es import act
    theta = np.load(policy).reshape(1, -1)
    env = MK64Env()
    ad = MK64Adapter(env)
    obs = ad.reset(STATE)
    t0 = time.time()
    for k in range(3600):
        steer = float(act(theta, obs.reshape(1, -1), obs.shape[0])[0])
        env.step(1, accel=True, steer=steer)
        obs = ad.observe()
        target = t0 + (k + 1) / 60.0
        if target > time.time():
            time.sleep(target - time.time())
    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "policy":
        live()
    else:
        replay()
