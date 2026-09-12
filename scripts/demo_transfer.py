"""THE DEMO: a policy trained in ~80s of simulation drives the real N64 game.

Nothing about Mario Kart 64 is in the policy. It sees ten numbers — speed,
where it sits across the road, which way it points, and the curvature ahead —
and it learned to drive them against a calibrated model, not against the ROM.
This script is the transfer test: same ten numbers, real emulator, real kart.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.mk64.env import MK64Env
from src.mk64 import player
from src.obs.mk64_adapter import MK64Adapter
from src.sim.train_es import act

STATE = "states/luigi_raceway_tt_start.st"


def main(policy="policies/es_seed0.npy", frames=3600, action_every=1):
    theta = np.load(policy).reshape(1, -1)
    env = MK64Env()
    ad = MK64Adapter(env)
    obs = ad.reset(STATE)
    st0 = player.read(env)
    # lapCount reads -1 while sitting on the line and flips to 0 the moment the
    # kart crosses it for the first time. Comparing against -1 makes the START
    # of the lap look like the end; a "completed lap" in 39 frames was the tell.
    lap0 = max(0, st0["lapCount"])

    log, t0 = [], time.time()
    for k in range(frames // action_every):
        steer = float(act(theta, obs.reshape(1, -1), obs.shape[0])[0])
        env.step(frames=action_every, accel=True, steer=steer)
        st = player.read(env)
        obs = ad.observe(st)
        i, s, lat = ad.track.project(st["pos"], hint=ad.hint)
        log.append({"frame": env.frame, "steer": steer, "speed": st["speed"],
                    "lateral": lat, "cp": int(i), "lap": st["lapCount"],
                    "pos": st["pos"]})
        if st["lapCount"] > lap0:
            break

    lat = np.abs([e["lateral"] for e in log])
    sp = np.array([e["speed"] for e in log])
    cps = [e["cp"] for e in log]
    out = {
        "policy": policy,
        "completed_lap": bool(log[-1]["lap"] > lap0),
        "frames": log[-1]["frame"] - log[0]["frame"],
        "checkpoints_reached": int(max(cps)) if cps else 0,
        "of_checkpoints": ad.track.n_checkpoints,
        "mean_speed": float(sp.mean()), "max_speed": float(sp.max()),
        "mean_abs_lateral": float(lat.mean()), "max_abs_lateral": float(lat.max()),
        "frames_offroad": int((lat > 72).sum() * action_every),
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/transfer_eval.json").write_text(json.dumps({"summary": out, "log": log}))
    print(json.dumps(out, indent=1))
    env.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
