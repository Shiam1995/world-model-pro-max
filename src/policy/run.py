"""Drive with a policy, and record how sure it was while doing it.

Every run logs (confidence, what happened next) so the confidence can be
checked rather than admired. SPEC.md's rule: a confidence number nobody
calibrated is a dial painted on a box.
"""

import numpy as np

from ..mk64 import player


def drive(env, adapter, model, state_path, max_frames=4000, action_every=4,
          record=True, stop_on_lap=True):
    """Run the policy from the start line. Returns a per-decision log."""
    obs = adapter.reset(state_path)
    start = player.read(env)
    lap0 = start["lapCount"]
    log = []
    steer = 0.0

    for step in range(max_frames // action_every):
        steer, conf, spread = model.act(obs)
        env.step(frames=action_every, accel=True, steer=float(np.clip(steer, -1, 1)))
        st = player.read(env)
        obs = adapter.observe(st)
        if record:
            i, s, lat = adapter.track.project(st["pos"])
            log.append({
                "frame": env.frame, "steer": steer, "confidence": conf,
                "spread": spread, "speed": st["speed"], "lateral": lat,
                "s": s, "checkpoint": i, "lap": st["lapCount"],
            })
        if stop_on_lap and st["lapCount"] > lap0:
            break
    return log


def summarise(log, track):
    """Did it get round, how fast, and where was it unsure?"""
    if not log:
        return {}
    finished = log[-1]["lap"] > log[0]["lap"]
    prog = _progress(log, track)
    spreads = np.array([e["spread"] for e in log])
    speeds = np.array([e["speed"] for e in log])
    lats = np.abs([e["lateral"] for e in log])
    worst = int(np.argmax(spreads))
    return {
        "finished": bool(finished),
        "frames": log[-1]["frame"] - log[0]["frame"],
        "progress": float(prog),
        "progress_frac": float(prog / track.length),
        "mean_speed": float(speeds.mean()),
        "mean_abs_lateral": float(lats.mean()),
        "max_abs_lateral": float(lats.max()),
        "mean_spread": float(spreads.mean()),
        "max_spread": float(spreads.max()),
        "least_confident_at": {
            "checkpoint": log[worst]["checkpoint"],
            "spread": float(spreads[worst]),
            "lateral": float(log[worst]["lateral"]),
        },
    }


def _progress(log, track):
    total, prev = 0.0, log[0]["s"]
    for e in log[1:]:
        d = e["s"] - prev
        if d < -track.length / 2:
            d += track.length
        elif d > track.length / 2:
            d -= track.length
        total += d
        prev = e["s"]
    return total
