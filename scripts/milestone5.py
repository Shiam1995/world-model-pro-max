"""Search a lap, clone it, and measure the clone. One job, three artefacts.

  runs/expert_lap.json     the searched lap (observations + chosen steering)
  policies/clone_v1.pt     the ensemble trained on it
  runs/clone_eval.json     how the clone drove, and how sure it was

Run it with: python3 scripts/milestone5.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.mk64.env import MK64Env
from src.mk64.search import LapSearch
from src.obs.mk64_adapter import MK64Adapter
from src.obs.track import Track, OBS_FIELDS, OBS_VERSION, observe
from src.policy.model import Ensemble
from src.policy import clone, run

STATE = "states/luigi_raceway_tt_start.st"
TRACK = "tracks/luigi_raceway.json"


def main():
    t0 = time.time()
    env = MK64Env()
    adapter = MK64Adapter(env, TRACK)
    track = adapter.track
    print(f"track: {track.n_checkpoints} checkpoints, {track.length:.0f} units", flush=True)

    # 1. expert lap by segment search against real arc-length progress
    print("\n[1] searching a lap", flush=True)
    log = []
    searcher = LapSearch(env, track, horizon=20)
    ok, total, segs = searcher.drive_lap(STATE, max_segments=200, log=log)
    print(f"    finished={ok} progress={total:.0f}/{track.length:.0f} "
          f"segments={segs} frames={len(log)} ({time.time()-t0:.0f}s)", flush=True)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/expert_lap.json").write_text(json.dumps(
        {"finished": ok, "progress": total, "segments": segs, "log": log}))
    if not ok:
        print("    lap not completed — stopping before cloning a failure", flush=True)
        env.close()
        return 1

    # 2. rebuild observations along that lap, then clone the steering
    print("\n[2] cloning", flush=True)
    obs_list, act_list, prev_s = [], [], None
    for e in log:
        state = {"pos": e["pos"], "yaw": e["yaw"], "speed": e["speed"]}
        o, prev_s = observe(state, track, prev_s)
        obs_list.append(o)
        act_list.append(e["steer"])
    obs_arr = np.array(obs_list, dtype=np.float32)
    print(f"    {len(obs_arr)} samples, {obs_arr.shape[1]} inputs", flush=True)

    model = Ensemble(len(OBS_FIELDS), n_heads=5, obs_version=OBS_VERSION, world="mk64")
    clone.train(model, obs_arr, np.array(act_list, dtype=np.float32), epochs=400)
    p = model.save("policies/clone_v1.pt",
                   meta={"track": TRACK, "expert": "LapSearch h=20",
                         "samples": len(obs_arr), "fields": list(OBS_FIELDS)})
    print(f"    saved {p}", flush=True)

    # 3. let the clone drive, and record how sure it was
    print("\n[3] evaluating the clone", flush=True)
    ev = run.drive(env, adapter, model, STATE, max_frames=6000)
    summary = run.summarise(ev, track)
    Path("runs/clone_eval.json").write_text(json.dumps({"summary": summary, "log": ev}))
    for k, v in summary.items():
        print(f"    {k}: {v}", flush=True)

    env.close()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
